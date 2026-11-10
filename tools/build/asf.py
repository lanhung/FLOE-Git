"""Download Sentinel-1 products from the ASF DAAC (NASA Earthdata login).

ASF and the Copernicus Data Space serve the same Sentinel-1 products. Product URLs,
sizes and MD5s come from the ASF search API (no login); downloads need an Earthdata
account, read from EARTHDATA_USERNAME / EARTHDATA_PASSWORD or a two-line file (user,
password), by default $FLOE_ROOT/ASF_credentials.txt (keep it chmod 600).

    python -m tools.build.asf --names-from $FLOE_DATA_ORIG --out $FLOE_ROOT/raw/s1 --workers 4
"""
import argparse
import hashlib
import os
import os.path as osp
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

import requests

from tools.build.cdse import names_from

SEARCH = 'https://api.daac.asf.alaska.edu/services/search/param'
AUTH_HOST = 'urs.earthdata.nasa.gov'


class EarthdataSession(requests.Session):
    """Sends the credentials only to Earthdata login. ASF redirects datapool ->
    sentinel1.asf.alaska.edu -> urs.earthdata.nasa.gov -> back with a cookie -> the file,
    so they are attached at the login hop and stripped everywhere else."""

    def __init__(self, user, password):
        super().__init__()
        self._creds = (user, password)

    def rebuild_auth(self, prepared_request, response):
        prepared_request.headers.pop('Authorization', None)
        if urlparse(prepared_request.url).hostname == AUTH_HOST:
            prepared_request.prepare_auth(self._creds)


def lookup(names, batch=50):
    """{name: {'url', 'bytes', 'md5'}} for GRD_MD products, from the ASF search API."""
    out = {}
    for i in range(0, len(names), batch):
        chunk = names[i:i + batch]
        for attempt in range(5):
            r = requests.get(SEARCH, params={'granule_list': ','.join(chunk), 'processingLevel': 'GRD_MD',
                                             'output': 'geojson'}, timeout=120)
            if r.status_code == 200:
                break
            time.sleep(15 * (attempt + 1))
        r.raise_for_status()
        for f in r.json().get('features', []):
            p = f['properties']
            name = p.get('sceneName') or p.get('fileID', '').replace('-GRD_MD', '')
            out[name] = {'url': p['url'], 'bytes': int(p.get('bytes') or 0), 'md5': (p.get('md5sum') or '').lower()}
    return out


def credentials(path=None):
    user, pw = os.environ.get('EARTHDATA_USERNAME'), os.environ.get('EARTHDATA_PASSWORD')
    if user and pw:
        return user, pw
    path = path or osp.join(os.environ.get('FLOE_ROOT', '.'), 'ASF_credentials.txt')
    if not osp.exists(path):
        sys.exit(f'no Earthdata credentials: set EARTHDATA_USERNAME/PASSWORD or write {path}')
    lines = [l.strip() for l in open(path) if l.strip()]
    return lines[0], lines[1]


def download(name, info, out_dir, session, chunk=8 << 20, retries=5):
    dst = osp.join(out_dir, name + '.zip')
    if osp.exists(dst) and (not info['bytes'] or osp.getsize(dst) == info['bytes']):
        return name, 'present'
    part = dst + '.part'
    for attempt in range(retries):
        try:
            h = hashlib.md5()
            with session.get(info['url'], stream=True, timeout=(60, 300)) as r:
                r.raise_for_status()
                if 'text/html' in r.headers.get('Content-Type', ''):
                    raise IOError('got a login page instead of the product (credentials or EULA?)')
                with open(part, 'wb') as f:
                    for block in r.iter_content(chunk):
                        f.write(block)
                        h.update(block)
            if info['bytes'] and osp.getsize(part) != info['bytes']:
                raise IOError(f'size {osp.getsize(part)} != {info["bytes"]}')
            if info['md5'] and h.hexdigest() != info['md5']:
                raise IOError('MD5 mismatch')
            os.replace(part, dst)
            return name, 'ok'
        except Exception as e:
            wait = 30 * (attempt + 1)
            print(f'[asf] {name}: {e}; retry in {wait}s', flush=True)
            time.sleep(wait)
    return name, 'FAILED'


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--names-from', required=True, help='directory of <product>.nc files, a .json list or a text file')
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--credentials', default=None)
    ap.add_argument('--limit', type=int, default=0, help='only the first N products (testing)')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    names = names_from(args.names_from)
    if args.limit:
        names = names[:args.limit]
    found = lookup(names)
    missing = [n for n in names if n not in found]
    gb = sum(v['bytes'] for v in found.values()) / 1e9
    print(f'{len(found)}/{len(names)} at ASF ({gb:.0f} GB); missing: {missing[:5]}', flush=True)
    if args.dry_run:
        return
    os.makedirs(args.out, exist_ok=True)
    session = EarthdataSession(*credentials(args.credentials))
    status = {}
    with ThreadPoolExecutor(args.workers) as pool:
        futs = [pool.submit(download, n, found[n], args.out, session) for n in names if n in found]
        for i, fut in enumerate(as_completed(futs), 1):
            n, st = fut.result()
            status[n] = st
            print(f'[{i}/{len(futs)}] {st:8s} {n}', flush=True)
    bad = [n for n, st in status.items() if st == 'FAILED']
    print(f'done: {len(status) - len(bad)} ok, {len(bad)} failed')
    sys.exit(1 if bad or missing else 0)


if __name__ == '__main__':
    main()
