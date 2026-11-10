"""Project ice-chart polygons into Sentinel-1 image geometry and rasterise them.

Map coordinates (EPSG:3413, polar stereographic north) of the annotation tie points are
related to image (row, col) by a 3rd-order polynomial fitted by least squares (as GDAL's
GCP polynomial; Park et al. 2020). Polygons are densified to 1 km segments in EPSG:3413,
their vertices mapped to image coordinates and burnt with rasterio, so the SAR is never
resampled. The fit residual at the tie points is returned for QC.
"""
import numpy as np
import shapely
from pyproj import Transformer
from rasterio import features
from rasterio.transform import Affine

_TO_3413 = Transformer.from_crs(4326, 3413, always_xy=True)


def _terms(x, y, deg):
    return np.stack([x ** i * y ** j for i in range(deg + 1) for j in range(deg + 1 - i)], -1)


def fit_inverse(geo, deg=3):
    """(lon, lat) -> (row, col) in the 80 m image. Returns (function, rms residual in px)."""
    lon, lat = geo['lon'].ravel(), geo['lat'].ravel()
    rows = np.repeat(geo['line'], len(geo['pixel']))
    cols = np.tile(geo['pixel'], len(geo['line']))
    x, y = _TO_3413.transform(lon, lat)
    mx, my, s = x.mean(), y.mean(), max(x.std(), y.std())
    T = _terms((x - mx) / s, (y - my) / s, deg)
    cr, *_ = np.linalg.lstsq(T, rows, rcond=None)
    cc, *_ = np.linalg.lstsq(T, cols, rcond=None)
    res = np.hypot(T @ cr - rows, T @ cc - cols)

    def to_image_xy(xy):
        """[N, 2] EPSG:3413 -> [N, 2] rasterio pixel space (x = col + 0.5, y = row + 0.5)."""
        t = _terms((xy[:, 0] - mx) / s, (xy[:, 1] - my) / s, deg)
        return np.stack([t @ cc + 0.5, t @ cr + 0.5], 1)
    return to_image_xy, float(np.sqrt((res ** 2).mean()))


def footprint(geo):
    """Scene outline in lon/lat from the tie-point grid border."""
    lon, lat = geo['lon'], geo['lat']
    ring = np.concatenate([np.stack([lon[0], lat[0]], 1), np.stack([lon[:, -1], lat[:, -1]], 1),
                           np.stack([lon[-1, ::-1], lat[-1, ::-1]], 1),
                           np.stack([lon[::-1, 0], lat[::-1, 0]], 1)])
    return shapely.Polygon(ring)


def burn(geoms_lonlat, ids, geo, shape, segment_m=1000.0, fill=-1):
    """Rasterise polygons (lon/lat shapely geometries) with integer ids into an image of
    `shape`; later geometries overwrite earlier ones. Returns (int32 raster, rms px)."""
    to_img, rms = fit_inverse(geo)
    shapes = []
    for g, i in zip(geoms_lonlat, ids):
        if g is None or g.is_empty:
            continue
        g = shapely.transform(g, lambda c: np.stack(_TO_3413.transform(c[:, 0], c[:, 1]), 1))
        g = shapely.segmentize(g, segment_m)
        g = shapely.transform(g, to_img)
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
