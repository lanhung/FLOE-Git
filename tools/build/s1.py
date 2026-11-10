"""Sentinel-1 EW GRDM (zipped SAFE) -> calibrated, denoised sigma0 at 80 m in SAR geometry.

Follows the AI4Arctic choices (no map projection, no speckle filter, 40 m -> 80 m by
2x2 averaging) with ESA's annotated thermal noise (range x azimuth vectors, IPF >= 2.9):

    sigma0 = (DN^2 - noise) / A_sigma^2          (S1-RS-MDA-52-7441, MPC-0392)

The 2x2 mean of DN^2 is calibrated and denoised with the LUTs evaluated at the 80 m
pixel centres (they vary over hundreds of pixels), and negative values are kept, which
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
from scipy.interpolate import RectBivariateSpline
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


def _vector_grid(lines, pixels, values, rows, cols):
    """LUT given on (line, pixel) vectors, evaluated on the grid rows x cols (40 m
    pixel coordinates): linear along pixel within each vector, then between vectors."""
    grid = np.stack([np.interp(cols, p, v) for p, v in zip(pixels, values)]).astype(np.float32)
    order = np.argsort(lines)
    lines, grid = lines[order].astype(np.float64), grid[order]
    k = np.clip(np.searchsorted(lines, rows) - 1, 0, len(lines) - 2)
    w = np.clip((rows - lines[k]) / (lines[k + 1] - lines[k]), 0, 1).astype(np.float32)[:, None]
    return (1 - w) * grid[k] + w * grid[k + 1]


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
    # calibration and noise vary over hundreds of pixels, so they are evaluated once at
    # the centres of the 80 m pixels and applied to the 2 x 2 mean of DN^2
    rows80 = np.arange(H2) * 2 + 0.5
    cols80 = np.arange(W2) * 2 + 0.5
    out = {}
    count = None
    replaced = {}

    for p in POLS:
        with rasterio.open(f'/vsizip/{zip_path}/{tif[p]}') as src:
            dn_full = src.read(1)[:H2 * 2, :W2 * 2]
        dn2 = np.empty((H2, W2), np.float32)
        cnt = np.empty((H2, W2), np.uint8)
        for r0 in range(0, H2 * 2, block):
            d = dn_full[r0:r0 + block].astype(np.float32)
            v = d > 0
            o0 = r0 // 2
            n = len(d) // 2
            dn2[o0:o0 + n] = (d * d).reshape(n, 2, W2, 2).sum((1, 3))
            cnt[o0:o0 + n] = v.reshape(n, 2, W2, 2).sum((1, 3))
        del dn_full
        with np.errstate(invalid='ignore', divide='ignore'):
            dn2 = np.where(cnt >= 2, dn2 / np.maximum(cnt, 1), np.nan).astype(np.float32)

        cl, cp, cv = _vectors(cal[p], 'calibrationVectorList', 'calibrationVector', 'sigmaNought')
        A = _vector_grid(cl, cp, cv, rows80, cols80)
        if noi[p].find('.//noiseRangeVector') is not None:
            nl, np_, nv = _vectors(noi[p], 'noiseRangeVectorList', 'noiseRangeVector', 'noiseRangeLut')
            az = _azimuth_noise(noi[p], H, W)
        else:                                            # IPF < 2.9: range vectors only
            nl, np_, nv = _vectors(noi[p], 'noiseVectorList', 'noiseVector', 'noiseLut')
            az = []
        noise = _vector_grid(nl, np_, nv, rows80, cols80)
        for (a0, a1, c0, c1, lines, lut) in az:
            ri = np.nonzero((rows80 >= a0) & (rows80 <= a1 + 1))[0]
            ci = np.nonzero((cols80 >= c0) & (cols80 <= c1 + 1))[0]
            if len(ri) and len(ci):
                f = np.interp(rows80[ri], lines, lut).astype(np.float32)[:, None]
                noise[ri[0]:ri[-1] + 1, ci[0]:ci[-1] + 1] *= f
        lin = (dn2 - noise) / (A * A)
        del noise, A, dn2
        if count is None:
            count = cnt

        bad = ~(lin > 0) & np.isfinite(lin)
        if bad.any():
            filled = np.where(np.isfinite(lin), lin, 0.0)
            w = uniform_filter(np.isfinite(lin).astype(np.float32), 9)
            mean = uniform_filter(filled, 9) / np.maximum(w, 1e-9)
            lin = np.where(bad & (mean > 0), mean, lin)
            lin = np.where(np.isfinite(lin) & ~(lin > 0), 10 ** (FLOOR_DB / 10), lin)
        replaced[p] = float(bad.sum() / max(np.isfinite(lin).sum(), 1))
        with np.errstate(divide='ignore', invalid='ignore'):
            out[p] = (10 * np.log10(lin)).astype(np.float32)

    valid = (count >= 2) & np.isfinite(out['hh']) & np.isfinite(out['hv'])
    lines, pixels, g = _tie_points(ann['hh'])
    incidence = RectBivariateSpline(lines, pixels, g[..., 5], kx=1, ky=1)(rows80, cols80).astype(np.float32)

    gi = ann['hh'].find('.//generalAnnotation/productInformation')
    hdr = ann['hh'].find('.//adsHeader')
    meta = {'product': str(name), 'ipf': str(ipf), 'mission': str(hdr.findtext('missionId')),
            'start': str(hdr.findtext('startTime')), 'stop': str(hdr.findtext('stopTime')),
            'pass': str(gi.findtext('pass')) if gi is not None else '',
            'native_shape': (H, W), 'nonpositive_replaced': replaced}
    geo = {'line': (lines - 0.5) / 2, 'pixel': (pixels - 0.5) / 2,      # in 80 m pixel units
           'lat': g[..., 2], 'lon': g[..., 3], 'height': g[..., 4], 'inc': g[..., 5]}
    return Scene(name, out, valid, incidence, geo, meta)
