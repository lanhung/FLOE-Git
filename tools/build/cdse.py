"""Find and download Sentinel-1 products from the Copernicus Data Space Ecosystem (CDSE).

Searching is anonymous; downloading needs a (free) CDSE account. Credentials come from
CDSE_USERNAME / CDSE_PASSWORD or from a two-line file (user, password), by default
$FLOE_ROOT/.cdse (keep it chmod 600).

    # the products behind an existing set of scene files (names are S1 product ids)
    python -m tools.build.cdse fetch --names-from $FLOE_DATA_ORIG --out $FLOE_ROOT/raw/s1
    # every EW GRDM HH/HV product over an area and period
    python -m tools.build.cdse search --wkt "POLYGON((...))" --start 2022-01-01 --end 2026-11-01 \
        --json-out found.json

Downloads go to <out>/<name>.zip via a .part file, are checked against the size and
MD5 in the catalogue, and are skipped when already complete.
"""
import argparse
import hashlib
import json
import os
import os.path as osp
import sys
import threading
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

TOKEN_URL = 'https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token'
CATALOGUE = 'https://catalogue.dataspace.copernicus.eu/odata/v1/Products'
DOWNLOAD = 'https://zipper.dataspace.copernicus.eu/odata/v1/Products({id})/$value'


def _get(params, session=None, retries=6):
    """Catalogue GET with backoff; the catalogue answers bursts with HTTP 429."""
    s = session or requests
    for attempt in range(retries):
        r = s.get(CATALOGUE, params=params, timeout=120)
        if r.status_code != 429:
            r.raise_for_status()
            return r.json()['value']
        time.sleep(int(r.headers.get('Retry-After', 0)) or 20 * (attempt + 1))
    r.raise_for_status()


def lookup(names, session=None, batch=20, pause=1.0):
    """Catalogue entries by product name (with or without .SAFE): {name: entry}."""
    names = [n[:-5] if n.endswith('.SAFE') else n for n in names]
    found = {}
    for i in range(0, len(names), batch):
        chunk = names[i:i + batch]
        flt = ' or '.join(f"Name eq '{n}.SAFE'" for n in chunk)
        for p in _get({'$filter': flt, '$top': batch}, session):
            found[p['Name'][:-5]] = p
        time.sleep(pause)
    return found


def search(wkt, start, end, product_type='EW_GRDM_1S', polarisation='HH&HV', top=1000):
    """All products of a type intersecting `wkt` between two ISO dates."""
    flt = (f"Collection/Name eq 'SENTINEL-1' and "
           f"Attributes/OData.CSC.StringAttribute/any(a:a/Name eq 'productType' and "
           f"a/OData.CSC.StringAttribute/Value eq '{product_type}') and "
           f"Attributes/OData.CSC.StringAttribute/any(a:a/Name eq 'polarisationChannels' and "
           f"a/OData.CSC.StringAttribute/Value eq '{polarisation}') and "
           f"OData.CSC.Intersects(area=geography'SRID=4326;{wkt}') and "
           f"ContentDate/Start ge {start}T00:00:00.000Z and ContentDate/Start lt {end}T00:00:00.000Z")
    out, skip = [], 0
    while True:
        page = _get({'$filter': flt, '$orderby': 'ContentDate/Start asc', '$top': top, '$skip': skip})
        out += page
        if len(page) < top:
            return out
        skip += top


class Token:
    """CDSE access token, refreshed before it expires (they live 10 minutes)."""

    def __init__(self, user, password):
        self.user, self.password = user, password
        self._lock = threading.Lock()
        self._token, self._refresh, self._expires = None, None, 0.0

    def _request(self, data):
        r = requests.post(TOKEN_URL, data={'client_id': 'cdse-public', **data}, timeout=60)
        r.raise_for_status()
        j = r.json()
        self._token, self._refresh = j['access_token'], j.get('refresh_token')
        self._expires = time.time() + j.get('expires_in', 600) - 60

    def get(self):
        with self._lock:
            if self._token is None or time.time() > self._expires:
                try:
                    if self._refresh:
                        self._request({'grant_type': 'refresh_token', 'refresh_token': self._refresh})
                    else:
                        raise RuntimeError
                except Exception:
                    self._request({'grant_type': 'password', 'username': self.user,
                                   'password': self.password})
            return self._token


def credentials(path=None):
    user, pw = os.environ.get('CDSE_USERNAME'), os.environ.get('CDSE_PASSWORD')
    if user and pw:
        return user, pw
    path = path or osp.join(os.environ.get('FLOE_ROOT', '.'), '.cdse')
    if not osp.exists(path):
        sys.exit(f'no CDSE credentials: set CDSE_USERNAME/CDSE_PASSWORD or write {path}')
    lines = [l.strip() for l in open(path) if l.strip()]
    return lines[0], lines[1]


def _md5(product):
    for c in product.get('Checksum', []):
        if c.get('Algorithm', '').upper() == 'MD5':
            return c['Value'].lower()
    return None


def download(product, out_dir, token, chunk=8 << 20, retries=5):
    name = product['Name'].replace('.SAFE', '')
    dst = osp.join(out_dir, name + '.zip')
    size, md5 = int(product.get('ContentLength', 0)), _md5(product)
    if osp.exists(dst) and (not size or osp.getsize(dst) == size):
        return name, 'present'
    part = dst + '.part'
    for attempt in range(retries):
        try:
            h = hashlib.md5()
            with requests.get(DOWNLOAD.format(id=product['Id']), stream=True, timeout=(60, 300),
                              headers={'Authorization': f'Bearer {token.get()}'}) as r:
                r.raise_for_status()
                with open(part, 'wb') as f:
                    for block in r.iter_content(chunk):
                        f.write(block)
                        h.update(block)
            if size and osp.getsize(part) != size:
                raise IOError(f'size {osp.getsize(part)} != {size}')
            if md5 and h.hexdigest() != md5:
                raise IOError('MD5 mismatch')
            os.replace(part, dst)
            return name, 'ok'
        except Exception as e:                    # network errors, 429, expired token
            wait = 30 * (attempt + 1)
            print(f'[cdse] {name}: {e}; retry in {wait}s', flush=True)
            time.sleep(wait)
    return name, 'FAILED'


def names_from(src):
    if osp.isdir(src):
        return sorted(osp.splitext(f)[0] for f in os.listdir(src) if f.startswith('S1') and f.endswith('.nc'))
    with open(src) as f:
        items = json.load(f) if src.endswith('.json') else [l.strip() for l in f if l.strip()]
    return sorted(osp.splitext(osp.basename(x))[0] for x in items)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    f = sub.add_parser('fetch', help='download products by name')
    f.add_argument('--names-from', required=True, help='directory of <product>.nc files, a .json list or a text file')
    f.add_argument('--out', required=True)
    f.add_argument('--workers', type=int, default=3, help='CDSE allows 4 parallel downloads per user')
    f.add_argument('--credentials', default=None)
    f.add_argument('--dry-run', action='store_true', help='only check that every product is in the catalogue')
    s = sub.add_parser('search', help='list products over an area and period')
    s.add_argument('--wkt', required=True)
    s.add_argument('--start', required=True)
    s.add_argument('--end', required=True)
    s.add_argument('--json-out', required=True)
    args = ap.parse_args()

    if args.cmd == 'search':
        found = search(args.wkt, args.start, args.end)
        with open(args.json_out, 'w') as fh:
            json.dump([{k: p.get(k) for k in ('Id', 'Name', 'ContentLength', 'ContentDate', 'Online')}
                       for p in found], fh, indent=1)
        print(f'{len(found)} products -> {args.json_out}')
        return

    names = names_from(args.names_from)
    found = lookup(names, requests.Session())
    products = [found[n] for n in names if n in found]
    missing = [n for n in names if n not in found]
    gb = sum(int(p.get('ContentLength', 0)) for p in products) / 1e9
    print(f'{len(products)}/{len(names)} in the catalogue ({gb:.0f} GB); missing: {missing[:5]}', flush=True)
    offline = [p['Name'] for p in products if not p.get('Online', True)]
    if offline:
        print(f'{len(offline)} products are offline (long-term archive), e.g. {offline[:3]}')
    if args.dry_run:
        return
    os.makedirs(args.out, exist_ok=True)
    token = Token(*credentials(args.credentials))
    token.get()                                       # fail early on bad credentials
    status = {}
    with ThreadPoolExecutor(args.workers) as pool:
        futs = [pool.submit(download, p, args.out, token) for p in products]
        for i, fut in enumerate(as_completed(futs), 1):
            name, st = fut.result()
            status[name] = st
            print(f'[{i}/{len(futs)}] {st:8s} {name}', flush=True)
    bad = [n for n, st in status.items() if st == 'FAILED']
    print(f'done: {sum(st != "FAILED" for st in status.values())} ok, {len(bad)} failed')
    sys.exit(1 if bad or missing else 0)


if __name__ == '__main__':
    main()
