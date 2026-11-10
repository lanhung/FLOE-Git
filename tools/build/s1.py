"""Sentinel-1 EW GRDM (zipped SAFE) -> calibrated, denoised sigma0 at 80 m in SAR geometry.

Follows the AI4Arctic choices (no map projection, no speckle filter, 40 m -> 80 m by
2x2 averaging) with ESA's annotated thermal noise (range x azimuth vectors, IPF >= 2.9):

    sigma0 = (DN^2 - noise) / A_sigma^2          (S1-RS-MDA-52-7441, MPC-0392)

Negative values after noise subtraction are kept through the 2x2 average, which
leaves the mean unbiased; pixels still <= 0 afterwards (HV near the noise floor) are
replaced by a 9x9 local mean and, failing that, a floor. The fraction replaced is
returned for QC. The NERSC denoiser used by AI4Arctic is not used because it has no
coefficients for Sentinel-1C/D or recent IPF versions.

Geolocation and incidence angle come from the annotation tie-point grid; `geo` holds
those points so charts can be projected into the image (tools/build/rasterise.py).
"""
import io
import re
import zipfile
from dataclasses import dataclass, field

import numpy as np
from lxml import etree
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import uniform_filter

POLS = ('hh', 'hv')
FLOOR_DB = -45.0


@dataclass
class Scene:
    name: str
    sigma0_db: dict                  # pol -> float32 [H, W] (NaN invalid)
    valid: np.ndarray                # bool [H, W]
    incidence: np.ndarray            # float32 [H, W], degrees
    geo: dict                        # tie points at 80 m: line, pixel, lat, lon, height, inc
    meta: dict = field(default_factory=dict)


def _xml(z, pattern):
    names = [n for n in z.namelist() if re.search(pattern, n)]
    if len(names) != 1:
        raise ValueError(f'{pattern}: found {names}')
    return etree.parse(io.BytesIO(z.read(names[0]))).getroot()


def _floats(el, tag):
    return np.array(el.findtext(tag).split(), dtype=np.float64)


def _vectors(root, list_tag, vec_tag, value_tag):
    lines, pixels, values = [], [], []
    for v in root.iter(vec_tag):
        lines.append(int(v.findtext('line')))
        pixels.append(_floats(v, 'pixel'))
        values.append(_floats(v, value_tag))
    return np.array(lines), pixels, values


def _row_interpolator(lines, pixels, values, ncols):
    """Values given on (line, pixel) vectors -> function(rows) -> [len(rows), ncols]
    (linear along pixel within each vector, then linear between vector lines)."""
    cols = np.arange(ncols)
    grid = np.stack([np.interp(cols, p, v) for p, v in zip(pixels, values)])   # [K, ncols]
    order = np.argsort(lines)
    lines, grid = lines[order].astype(np.float64), grid[order]

    def at(rows):
        rows = np.asarray(rows, dtype=np.float64)
        k = np.clip(np.searchsorted(lines, rows) - 1, 0, len(lines) - 2)
        w = np.clip((rows - lines[k]) / (lines[k + 1] - lines[k]), 0, 1)[:, None]
        return (1 - w) * grid[k] + w * grid[k + 1]
    return at


def _azimuth_noise(root, nrows, ncols):
    """Noise azimuth scaling blocks (IPF >= 2.9) -> list of (r0, r1, c0, c1, f(rows))."""
    blocks = []
    for v in root.iter('noiseAzimuthVector'):
        r0, r1 = int(v.findtext('firstAzimuthLine')), int(v.findtext('lastAzimuthLine'))
        c0, c1 = int(v.findtext('firstRangeSample')), int(v.findtext('lastRangeSample'))
        lines = _floats(v, 'line')
        lut = _floats(v, 'noiseAzimuthLut')
        blocks.append((r0, r1, c0, c1, lines, lut))
    return blocks


def _tie_points(ann):
    pts = [(int(p.findtext('line')), int(p.findtext('pixel')), float(p.findtext('latitude')),
            float(p.findtext('longitude')), float(p.findtext('height')),
            float(p.findtext('incidenceAngle')))
           for p in ann.iter('geolocationGridPoint')]
    a = np.array(pts)
    lines, pixels = np.unique(a[:, 0]), np.unique(a[:, 1])
    if len(lines) * len(pixels) != len(a):
        raise ValueError('geolocation grid is not rectilinear')
    order = np.lexsort((a[:, 1], a[:, 0]))
    g = a[order].reshape(len(lines), len(pixels), 6)
    return lines, pixels, g


def process(zip_path, block=1024):
    """Read one zipped EW GRDM 1SDH product. Returns a Scene at 80 m."""
    import rasterio
    with zipfile.ZipFile(zip_path) as z:
        name = [n for n in z.namelist() if n.endswith('.SAFE/')][0][:-6] \
            if any(n.endswith('.SAFE/') for n in z.namelist()) else z.namelist()[0].split('.SAFE')[0]
        manifest = etree.parse(io.BytesIO(z.read(f'{name}.SAFE/manifest.safe'))).getroot()
        ipf = manifest.xpath('string(//*[local-name()="software"]/@version)')
        ann = {p: _xml(z, rf'annotation/s1.-ew-grd-{p}-.*\.xml$') for p in POLS}
        cal = {p: _xml(z, rf'calibration/calibration-s1.-ew-grd-{p}-.*\.xml$') for p in POLS}
        noi = {p: _xml(z, rf'calibration/noise-s1.-ew-grd-{p}-.*\.xml$') for p in POLS}
        tif = {p: [n for n in z.namelist() if re.search(rf'measurement/s1.-ew-grd-{p}-.*\.tiff$', n)][0]
               for p in POLS}

    info = ann['hh'].find('.//imageInformation')
    H, W = int(info.findtext('numberOfLines')), int(info.findtext('numberOfSamples'))
    H2, W2 = H // 2, W // 2
    out = {p: np.full((H2, W2), np.nan, np.float32) for p in POLS}
    count = np.zeros((H2, W2), np.uint8)
    replaced = {}

    for p in POLS:
        cl, cp, cv = _vectors(cal[p], 'calibrationVectorList', 'calibrationVector', 'sigmaNought')
        A = _row_interpolator(cl, cp, cv, W)
        if noi[p].find('.//noiseRangeVector') is not None:
            nl, np_, nv = _vectors(noi[p], 'noiseRangeVectorList', 'noiseRangeVector', 'noiseRangeLut')
            az = _azimuth_noise(noi[p], H, W)
        else:                                            # IPF < 2.9: range vectors only
            nl, np_, nv = _vectors(noi[p], 'noiseVectorList', 'noiseVector', 'noiseLut')
            az = []
        N = _row_interpolator(nl, np_, nv, W)
        with rasterio.open(f'/vsizip/{zip_path}/{tif[p]}') as src:
            for r0 in range(0, H2 * 2, block):
                r1 = min(r0 + block, H2 * 2)
                dn = src.read(1, window=((r0, r1), (0, W))).astype(np.float64)
                rows = np.arange(r0, r1)
                noise = N(rows)
                for (a0, a1, c0, c1, lines, lut) in az:
                    s0, s1 = max(r0, a0), min(r1, a1 + 1)
                    if s0 >= s1:
                        continue
                    f = np.interp(np.arange(s0, s1), lines, lut)[:, None]
                    noise[s0 - r0:s1 - r0, c0:c1 + 1] *= f
                sig = (dn ** 2 - noise) / A(rows) ** 2
                valid = dn > 0
                sig = np.where(valid, sig, 0.0)[:, :W2 * 2]
                v = valid[:, :W2 * 2]
                s = sig.reshape(-1, 2, W2, 2).sum((1, 3))
                c = v.reshape(-1, 2, W2, 2).sum((1, 3))
                o0 = r0 // 2
                with np.errstate(invalid='ignore', divide='ignore'):
                    out[p][o0:o0 + len(s)] = np.where(c >= 2, s / np.maximum(c, 1), np.nan)
                if p == 'hh':
                    count[o0:o0 + len(c)] = c

        lin = out[p]
        bad = ~(lin > 0) & np.isfinite(lin)
        if bad.any():
            filled = np.where(np.isfinite(lin), lin, 0.0)
            w = uniform_filter(np.isfinite(lin).astype(np.float64), 9)
            mean = uniform_filter(filled, 9) / np.maximum(w, 1e-9)
            lin = np.where(bad & (mean > 0), mean, lin)
            lin = np.where(np.isfinite(lin) & ~(lin > 0), 10 ** (FLOOR_DB / 10), lin)
        replaced[p] = float(bad.sum() / max(np.isfinite(out[p]).sum(), 1))
        with np.errstate(divide='ignore', invalid='ignore'):
            out[p] = (10 * np.log10(lin)).astype(np.float32)

    valid = (count >= 2) & np.isfinite(out['hh']) & np.isfinite(out['hv'])
    lines, pixels, g = _tie_points(ann['hh'])
    rows80, cols80 = np.arange(H2) * 2 + 0.5, np.arange(W2) * 2 + 0.5
    inc = RegularGridInterpolator((lines, pixels), g[..., 5], bounds_error=False, fill_value=None)
    rr, cc = np.meshgrid(rows80, cols80, indexing='ij')
    incidence = inc(np.stack([rr.ravel(), cc.ravel()], 1)).reshape(H2, W2).astype(np.float32)

    gi = ann['hh'].find('.//generalAnnotation/productInformation')
    hdr = ann['hh'].find('.//adsHeader')
    meta = {'product': name, 'ipf': ipf, 'mission': hdr.findtext('missionId'),
            'start': hdr.findtext('startTime'), 'stop': hdr.findtext('stopTime'),
            'pass': gi.findtext('pass') if gi is not None else '',
            'native_shape': (H, W), 'nonpositive_replaced': replaced}
    geo = {'line': (lines - 0.5) / 2, 'pixel': (pixels - 0.5) / 2,      # in 80 m pixel units
           'lat': g[..., 2], 'lon': g[..., 3], 'height': g[..., 4], 'inc': g[..., 5]}
    return Scene(name, out, valid, incidence, geo, meta)
