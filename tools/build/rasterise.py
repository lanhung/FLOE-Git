"""Project ice-chart polygons into Sentinel-1 image geometry and rasterise them.

Image (row, col) is interpolated from map coordinates (EPSG:3413, polar stereographic
north) of the annotation tie points with a Clough-Tocher (piecewise cubic, C1) scheme,
which is exact at every tie point; a 3rd-order polynomial covers the few vertices
outside the tie-point hull. Polygons are clipped to the scene, densified to 1 km
segments, mapped to image coordinates and burnt with rasterio, so the SAR is never
resampled. As a conservative QC number the interpolation is refitted on every other
tie point and its error at the held-out sea points is reported (tie spacing doubled,
so the real error is several times smaller; land points are excluded because terrain
displaces them in ground range).
"""
import numpy as np
import shapely
from pyproj import Transformer
from rasterio import features
from rasterio.transform import Affine

_TO_3413 = Transformer.from_crs(4326, 3413, always_xy=True)


def _terms(x, y, deg):
    return np.stack([x ** i * y ** j for i in range(deg + 1) for j in range(deg + 1 - i)], -1)


def _poly(x, y, rows, cols, deg=3):
    mx, my, s = x.mean(), y.mean(), max(x.std(), y.std())
    T = _terms((x - mx) / s, (y - my) / s, deg)
    cr = np.linalg.lstsq(T, rows, rcond=None)[0]
    cc = np.linalg.lstsq(T, cols, rcond=None)[0]

    def f(xq, yq):
        t = _terms((xq - mx) / s, (yq - my) / s, deg)
        return t @ cr, t @ cc
    return f


def _tie_xy(geo):
    x, y = _TO_3413.transform(geo['lon'], geo['lat'])
    rows, cols = np.meshgrid(geo['line'], geo['pixel'], indexing='ij')
    return np.asarray(x), np.asarray(y), rows, cols


def fit_inverse(geo):
    """(EPSG:3413 x, y) -> rasterio pixel space of the 80 m image. Returns
    (function [N,2] -> [N,2], held-out sea error in 80 m pixels, rms)."""
    from scipy.interpolate import CloughTocher2DInterpolator
    x, y, rows, cols = _tie_xy(geo)
    pts = np.stack([x.ravel(), y.ravel()], 1)
    ct_r = CloughTocher2DInterpolator(pts, rows.ravel())
    ct_c = CloughTocher2DInterpolator(pts, cols.ravel())
    poly = _poly(x.ravel(), y.ravel(), rows.ravel(), cols.ravel())

    def to_image_xy(xy):
        r, c = ct_r(xy), ct_c(xy)
        bad = ~(np.isfinite(r) & np.isfinite(c))
        if bad.any():
            r[bad], c[bad] = poly(xy[bad, 0], xy[bad, 1])
        return np.stack([c + 0.5, r + 0.5], 1)

    tr = np.zeros(x.shape, bool)
    tr[::2, ::2] = True
    te = ~tr & (geo['height'] < 50)
    err = np.nan
    if te.any():
        sub = np.stack([x[tr], y[tr]], 1)
        q = np.stack([x[te], y[te]], 1)
        er = CloughTocher2DInterpolator(sub, rows[tr])(q) - rows[te]
        ec = CloughTocher2DInterpolator(sub, cols[tr])(q) - cols[te]
        ok = np.isfinite(er) & np.isfinite(ec)
        if ok.any():
            err = float(np.sqrt((er[ok] ** 2 + ec[ok] ** 2).mean()))
    return to_image_xy, err


def _map_polygons(g, f):
    """Apply a coordinate mapping ring by ring, closing every ring exactly; non-polygonal
    parts (lines, points left by clipping) are dropped."""
    def ring(coords):
        c = f(np.asarray(coords)[:, :2])
        c[-1] = c[0]
        return c
    parts = []
    for p in getattr(g, 'geoms', [g]):
        if p.geom_type == 'Polygon' and not p.is_empty:
            parts.append(shapely.Polygon(ring(p.exterior.coords), [ring(r.coords) for r in p.interiors]))
        elif p.geom_type in ('MultiPolygon', 'GeometryCollection'):
            m = _map_polygons(p, f)
            if m is not None:
                parts.extend(getattr(m, 'geoms', [m]))
    if not parts:
        return None
    return parts[0] if len(parts) == 1 else shapely.MultiPolygon(parts)


def footprint(geo):
    """Scene outline in lon/lat from the tie-point grid border."""
    lon, lat = geo['lon'], geo['lat']
    ring = np.concatenate([np.stack([lon[0], lat[0]], 1), np.stack([lon[:, -1], lat[:, -1]], 1),
                           np.stack([lon[-1, ::-1], lat[-1, ::-1]], 1),
                           np.stack([lon[::-1, 0], lat[::-1, 0]], 1)])
    return shapely.Polygon(ring)


def burn(geoms_lonlat, ids, geo, shape, segment_m=1000.0, fill=-1):
    """Rasterise polygons (lon/lat shapely geometries) with integer ids into an image of
    `shape`; later geometries overwrite earlier ones. Returns (int32 raster, QC error px)."""
    to_img, rms = fit_inverse(geo)
    fp = shapely.transform(footprint(geo), lambda c: np.stack(_TO_3413.transform(c[:, 0], c[:, 1]), 1))
    fp = fp.buffer(2000.0)
    shapes = []
    for g, i in zip(geoms_lonlat, ids):
        if g is None or g.is_empty:
            continue
        g = shapely.transform(g, lambda c: np.stack(_TO_3413.transform(c[:, 0], c[:, 1]), 1))
        if not g.is_valid:
            g = shapely.make_valid(g)
        g = g.intersection(fp)
        if g.is_empty:
            continue
        g = _map_polygons(shapely.segmentize(g, segment_m), to_img)
        if g is None:
            continue
        if not g.is_valid:
            g = shapely.make_valid(g)
        shapes.append((g, int(i)))
    out = features.rasterize(shapes, out_shape=shape, transform=Affine.identity(),
                             fill=fill, dtype='int32', all_touched=False) if shapes \
        else np.full(shape, fill, np.int32)
    return out, rms


def edge_distance(poly, cap=255):
    """Distance (80 m pixels, capped) from each pixel to the nearest change of polygon id."""
    from scipy.ndimage import distance_transform_edt
    edge = np.zeros(poly.shape, bool)
    edge[:-1] |= poly[:-1] != poly[1:]
    edge[1:] |= poly[:-1] != poly[1:]
    edge[:, :-1] |= poly[:, :-1] != poly[:, 1:]
    edge[:, 1:] |= poly[:, :-1] != poly[:, 1:]
    d = distance_transform_edt(~edge)
    return np.minimum(d, cap).astype(np.uint8)
