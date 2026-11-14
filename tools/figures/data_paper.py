"""Figures of the dataset paper (paper_dataset/), drawn from the rebuilt archive.

    python -m tools.figures.data_paper coverage  --table polygons_v2.npz --data-dir $FLOE_V2_DIR \
                                                 --charts $FLOE_ROOT/raw/dmi --land ne_50m_land.zip
    python -m tools.figures.data_paper season    --table polygons_v2.npz
    python -m tools.figures.data_paper example   --scene $FLOE_V2_DIR/<product>.nc
    python -m tools.figures.data_paper purity    --purity label_purity_v2.npz [--model-cm cm.npz]
    python -m tools.figures.data_paper agreement --agreement chart_agreement_same_image.npz

The table comes from tools.build.polygon_table, the other inputs from
tools.build.label_purity and tools.build.chart_agreement. Each command writes
<name>.pdf and <name>.png to --out (default $FLOE_FIG_DIR/data_paper). `coverage`
needs the build environment (geopandas, pyproj, rasterio).

Colours: floe size is an ordered scale, so the five ice classes share one hue from light
(cake) to dark (vast) and open water is a neutral grey; nothing else is coloured.
"""
import argparse
import json
import os
import os.path as osp

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from scipy.ndimage import distance_transform_edt  # noqa: E402

NAMES = ('Water', 'Cake', 'Small', 'Medium', 'Big', 'Vast')
WATER = '#c3c2b7'
ICE = ('#86b6ef', '#5598e7', '#2a78d6', '#1c5cab', '#0d366b')
CLASS_COLORS = (WATER,) + ICE
RAMP = ('#ffffff', '#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b')
BLUES = LinearSegmentedColormap.from_list('blues', RAMP)
INK, INK2, MUTED, GRID, AXIS, LAND = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#ecebe6'
REGIONS = ('Qaanaaq', 'NorthWest', 'CentralWest', 'SouthWest', 'CapeFarewell',
           'SouthEast', 'CentralEast', 'NorthEast', 'North')
REGION_SHORT = ('Qa', 'NW', 'CW', 'SW', 'CF', 'SE', 'CE', 'NE', 'N')
MONTHS = 'JFMAMJJASOND'
KM2 = 0.08 ** 2                                         # one 80 m pixel

plt.rcParams.update({
    'font.size': 7.5, 'axes.titlesize': 8, 'axes.labelsize': 7.5, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'legend.fontsize': 7, 'figure.dpi': 200, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.edgecolor': AXIS, 'axes.linewidth': 0.6,
    'xtick.color': MUTED, 'ytick.color': MUTED, 'xtick.labelcolor': INK2, 'ytick.labelcolor': INK2,
    'axes.labelcolor': INK2, 'text.color': INK, 'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'legend.frameon': False,
})


def save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(osp.join(out, f'{name}.{ext}'))
    plt.close(fig)
    print(f'wrote {osp.join(out, name)}.pdf/.png')


def load_table(path):
    t = dict(np.load(path))
    stamp = np.array([n.split('_')[4] for n in t['scenes']])
    t['scene_year'] = np.array([int(s[:4]) for s in stamp])
    t['scene_month'] = np.array([int(s[4:6]) for s in stamp])
    return t


def class_legend(ax, with_water=True, **kw):
    names, cols = (NAMES, CLASS_COLORS) if with_water else (NAMES[1:], ICE)
    return ax.legend(handles=[Patch(facecolor=c, label=n) for n, c in zip(names, cols)],
                     handlelength=1.0, handleheight=1.0, columnspacing=1.0, handletextpad=0.4, **kw)


def cell_text(ax, x, y, text, value, vmax):
    ax.text(x, y, text, ha='center', va='center', fontsize=6.5, color='white' if value > 0.55 * vmax else INK)


# ------------------------------------------------------------------ coverage

def coverage(args):
    import geopandas as gpd
    import h5py
    import hdf5plugin  # noqa: F401
    from pyproj import Transformer
    from rasterio import features
    from rasterio.enums import MergeAlg
    from rasterio.transform import from_origin
    from shapely.geometry import Polygon, box

    t = load_table(args.table)
    to_xy = Transformer.from_crs(4326, 3413, always_xy=True).transform
    feet = []
    for n in t['scenes']:
        with h5py.File(osp.join(args.data_dir, n + '.nc'), 'r') as f:
            lat, lon = f['sar_grid_lat'][...], f['sar_grid_lon'][...]
        ring = [(lon[i, j], lat[i, j]) for i, j in ((0, 0), (0, -1), (-1, -1), (-1, 0))]
        feet.append(Polygon(zip(*to_xy(*zip(*ring)))))
    bounds = np.array([p.bounds for p in feet])
    res = 10e3
    x0, y0 = np.floor(bounds[:, :2].min(0) / 1e5) * 1e5 - 1e5
    x1, y1 = np.ceil(bounds[:, 2:].max(0) / 1e5) * 1e5 + 1e5
    shape = (int((y1 - y0) / res), int((x1 - x0) / res))
    count = features.rasterize([(p, 1) for p in feet], out_shape=shape, transform=from_origin(x0, y1, res, res),
                               merge_alg=MergeAlg.add, dtype='int32')

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.0, 3.3), gridspec_kw={'width_ratios': [1.0, 1.15], 'wspace': 0.3})
    for lat_ in (60, 70, 80):
        ax.plot(*to_xy(np.linspace(-180, 180, 361), np.full(361, lat_)), color=GRID, lw=0.4, zorder=0)
    for lon_ in range(-180, 180, 30):
        ax.plot(*to_xy(np.full(50, lon_), np.linspace(50, 90, 50)), color=GRID, lw=0.4, zorder=0)
    im = ax.imshow(np.ma.masked_equal(count, 0), extent=(x0, x1, y0, y1), cmap=BLUES, vmin=0, interpolation='nearest',
                   zorder=1)
    if args.land:
        land = gpd.read_file(args.land).clip(box(-179.9, 45, 179.9, 89.9)).to_crs(3413)
        land.plot(ax=ax, facecolor=LAND, edgecolor=AXIS, linewidth=0.3, zorder=2)
    index = json.load(open(osp.join(args.charts, 'index.json')))
    for reg, short in zip(REGIONS, REGION_SHORT):
        boxes = np.array([r['bbox'] for r in index if r['name'].split('_')[1] == reg])
        n_scenes = len(np.unique(t['scene'][t['region'] == reg]))
        if not len(boxes) or not n_scenes:
            continue
        lon_c, lat_c = np.median((boxes[:, 0] + boxes[:, 2]) / 2), np.median((boxes[:, 1] + boxes[:, 3]) / 2)
        ax.annotate(f'{short}\n{n_scenes}', to_xy(lon_c, lat_c), ha='center', va='center', fontsize=6, color=INK,
                    zorder=4, linespacing=0.95,
                    bbox=dict(boxstyle='round,pad=0.18', facecolor='white', edgecolor='none', alpha=0.8))
    ax.set_xlim(x0, x1), ax.set_ylim(y0, y1), ax.set_aspect('equal')
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
    ax.plot([x0 + 1.0e5, x0 + 6.0e5], [y0 + 1.1e5] * 2, color=INK, lw=1.2, solid_capstyle='butt', zorder=5)
    ax.text(x0 + 3.5e5, y0 + 1.5e5, '500 km', ha='center', va='bottom', fontsize=6, zorder=5)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.outline.set_visible(False), cb.ax.set_title('scenes', fontsize=6.5, color=INK2, loc='left', pad=4)
    ax.set_title(f'(a) Footprints of the {len(feet)} scenes', loc='left')

    years = np.arange(t['scene_year'].min(), t['scene_year'].max() + 1)
    grid = np.zeros((len(years), 12), int)
    np.add.at(grid, (t['scene_year'] - years[0], t['scene_month'] - 1), 1)
    bx.imshow(grid, cmap=BLUES, vmin=0, aspect='auto')
    for i in range(len(years)):
        for j in range(12):
            cell_text(bx, j, i, str(grid[i, j]) if grid[i, j] else '', grid[i, j], grid.max())
    bx.set_xticks(range(12), MONTHS), bx.set_yticks(range(len(years)), years)
    bx.set_xticks(np.arange(13) - 0.5, minor=True), bx.set_yticks(np.arange(len(years) + 1) - 0.5, minor=True)
    bx.grid(which='minor', color='white', lw=1.5), bx.tick_params(which='both', length=0)
    for s in bx.spines.values():
        s.set_visible(False)
    bx.set_title('(b) Scenes per month', loc='left')
    save(fig, args.out, 'coverage')


# -------------------------------------------------------------------- season

def season(args):
    t = load_table(args.table)
    comp = t['composition'].astype(np.float64)
    ok = (t['floe'] >= 1) & (t['floe'] <= 5) & (t['unknown'] <= 0.5) & (comp[:, 1:].sum(1) > 0)
    ice = t['labelled'][:, None] * comp[:, 1:] / comp.sum(1, keepdims=True).clip(1e-12)       # px of each ice class
    month = t['scene_month'][t['scene']] - 1
    region = np.array([REGIONS.index(r) if r in REGIONS else -1 for r in t['region']])
    by_month, by_region = np.zeros((12, 5)), np.zeros((len(REGIONS), 5))
    np.add.at(by_month, month[ok], ice[ok])
    np.add.at(by_region, region[ok & (region >= 0)], ice[ok & (region >= 0)])

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 3.2), sharex='col',
                             gridspec_kw={'height_ratios': [0.5, 1], 'hspace': 0.14, 'wspace': 0.16,
                                          'width_ratios': [1.2, 1]})
    for col, (table, labels, title) in enumerate(((by_month, MONTHS, 'month'), (by_region, REGION_SHORT, 'chart region'))):
        a, b = axes[:, col]
        x = np.arange(len(table))
        a.bar(x, table.sum(1) * KM2 / 1e6, width=0.62, color=MUTED, linewidth=0)
        a.yaxis.grid(True, color=GRID, lw=0.5), a.set_axisbelow(True), a.tick_params(axis='x', length=0)
        a.set_title(f'({"ab"[col]}) Ice by {title}', loc='left')
        mix = table / table.sum(1, keepdims=True).clip(1e-12)
        bottom = np.zeros(len(table))
        for k in range(5):
            b.bar(x, np.maximum(mix[:, k] - 0.008, 0), bottom=bottom + 0.004, width=0.62, color=ICE[k], linewidth=0)
            bottom += mix[:, k]
        b.set_xticks(x, labels), b.set_ylim(0, 1), b.set_yticks([0, 0.5, 1], ['0', '50', '100'])
        b.tick_params(axis='x', length=0)
    axes[0, 0].set_ylabel('ice area\n(10$^6$ km$^2$)'), axes[1, 0].set_ylabel('floe-size mix\nof the ice (%)')
    class_legend(axes[1, 0], with_water=False, loc='upper center', bbox_to_anchor=(0.95, -0.14), ncol=5)
    save(fig, args.out, 'season')


# ------------------------------------------------------------------- example

def example(args):
    import h5py
    import hdf5plugin  # noqa: F401
    s, z = args.step, args.zoom
    with h5py.File(args.scene, 'r') as f:
        full_poly = f['polygon_id'][...]
        comp, unk, pf = f['poly_composition'][...], f['poly_unknown'][...], f['poly_floe'][...]
        npx = np.bincount(full_poly[full_poly >= 0], minlength=len(pf))
        shown = [p for p in np.argsort(-npx) if 0 < pf[p] < 6 and unk[p] <= 0.1 and npx[p] > 0][:args.polygons]
        target = shown[ord(args.zoom_polygon) - 65]
        r0, c0 = _best_window(full_poly == target, z)
        hh = f['nersc_sar_primary'][::s, ::s]
        crop = f['nersc_sar_primary'][r0:r0 + z, c0:c0 + z]
        floe, poly = f['FLOE'][::s, ::s], full_poly[::s, ::s]
    del full_poly
    valid = np.isfinite(hh) & (poly >= 0)
    lo, hi = np.percentile(hh[valid], [1, 99])

    fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.95), gridspec_kw={'width_ratios': [1, 1, 1, 0.8], 'wspace': 0.05})
    ax = axes[0]
    ax.imshow(hh, cmap='gray', vmin=lo, vmax=hi, interpolation='nearest')
    ax.add_patch(plt.Rectangle((c0 / s, r0 / s), z / s, z / s, fill=False, edgecolor='white', linewidth=1.6))
    ax.add_patch(plt.Rectangle((c0 / s, r0 / s), z / s, z / s, fill=False, edgecolor=INK, linewidth=0.7))
    ax.set_title('(a) HH backscatter', loc='left'), ax.axis('off')
    for a in axes[:3]:
        a.set_anchor('N')

    ax = axes[1]
    ax.imshow(crop, cmap='gray', vmin=lo, vmax=hi, interpolation='nearest')
    ax.plot([0.06 * z, 0.06 * z + 125], [0.94 * z] * 2, color='white', lw=3.0, solid_capstyle='butt')
    ax.plot([0.06 * z, 0.06 * z + 125], [0.94 * z] * 2, color=INK, lw=1.4, solid_capstyle='butt')
    ax.text(0.06 * z + 62, 0.915 * z, '10 km', ha='center', va='bottom', fontsize=6, color=INK,
            bbox=dict(boxstyle='round,pad=0.1', facecolor='white', edgecolor='none', alpha=0.8))
    ax.set_title(f'(b) Detail of polygon {args.zoom_polygon}', loc='left'), ax.axis('off')

    ax = axes[2]
    ax.imshow(np.where(floe < 6, floe, 6), cmap=ListedColormap(CLASS_COLORS + ('#ffffff',)), vmin=-0.5, vmax=6.5,
              interpolation='nearest')
    edge = np.zeros(poly.shape, bool)
    edge[:-1] |= poly[:-1] != poly[1:]
    edge[:, :-1] |= poly[:, :-1] != poly[:, 1:]
    ax.imshow(np.where(edge, 1.0, np.nan), cmap=ListedColormap(['#ffffff']), interpolation='nearest')
    for i, p in enumerate(shown):
        depth = distance_transform_edt(np.pad(poly == p, 1))[1:-1, 1:-1]      # letter at the most interior point
        r, c = np.unravel_index(depth.argmax(), depth.shape)
        ax.text(c, r, chr(65 + i), ha='center', va='center', fontsize=6, color=INK,
                bbox=dict(boxstyle='circle,pad=0.1', facecolor='white', edgecolor='none'))
    ax.set_title('(c) Label', loc='left'), ax.axis('off')
    class_legend(ax, loc='upper center', bbox_to_anchor=(-0.55, -0.03), ncol=6)

    ax = axes[3]
    box_ = ax.get_position()
    ax.set_position([box_.x0 + 0.035, box_.y0, box_.width - 0.075, box_.height])
    for i, p in enumerate(shown):
        q = comp[p] / comp[p].sum()
        left = 0.0
        for k in range(6):
            if q[k] > 0:
                ax.barh(i, max(q[k] - 0.012, 0.004), left=left + 0.006, height=0.56, color=CLASS_COLORS[k], linewidth=0)
                left += q[k]
        ax.text(1.04, i, NAMES[pf[p]], va='center', ha='left', fontsize=6.5, color=INK2)
    ax.set_yticks(range(len(shown)), [chr(65 + i) for i in range(len(shown))])
    ax.invert_yaxis(), ax.set_xlim(0, 1), ax.set_xticks([0, 0.5, 1], ['0', '50', '100'])
    ax.tick_params(axis='y', length=0), ax.spines['left'].set_visible(False)
    ax.set_xlabel('share of the polygon (%)', labelpad=1)
    ax.set_title('(d) Egg code and label', loc='left')
    save(fig, args.out, 'example')


def _best_window(inside, z):
    """Top-left corner of the z x z window with the largest share of the polygon."""
    step = max(z // 8, 1)
    c = np.pad(inside, ((1, 0), (1, 0))).cumsum(0, dtype=np.int64).cumsum(1)
    r = np.arange(0, inside.shape[0] - z + 1, step)
    q = np.arange(0, inside.shape[1] - z + 1, step)
    total = c[np.ix_(r + z, q + z)] - c[np.ix_(r, q + z)] - c[np.ix_(r + z, q)] + c[np.ix_(r, q)]
    i, j = np.unravel_index(total.argmax(), total.shape)
    return int(r[i]), int(q[j])


# -------------------------------------------------------------------- purity

def _iou(cm):
    k = cm.shape[0]
    tp = np.diag(cm[:, :k])
    return tp / np.maximum(cm.sum(1) + cm[:, :k].sum(0) - tp, 1e-12)


def purity(args):
    d = np.load(args.purity)
    full, ice = d['full'], d['ice']
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.0, 2.2), gridspec_kw={'width_ratios': [1.3, 1.0], 'wspace': 0.3})

    for i, L in enumerate(range(1, 6)):
        q = full[L] / full[L].sum()
        left = 0.0
        for k in range(6):
            if q[k] > 0.004:
                ax.barh(i, max(q[k] - 0.008, 0.002), left=left + 0.004, height=0.56, color=CLASS_COLORS[k], linewidth=0)
            if k == L:
                ax.text(left + q[k] / 2, i, f'{100 * q[k]:.0f}%', ha='center', va='center', fontsize=6.5,
                        color='white' if k >= 3 else INK)
            left += q[k]
    ax.set_yticks(range(5), [f'{NAMES[L]} label' for L in range(1, 6)])
    ax.invert_yaxis(), ax.set_xlim(0, 1), ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ['0', '25', '50', '75', '100'])
    ax.tick_params(axis='y', length=0), ax.spines['left'].set_visible(False)
    ax.set_xlabel('share of the pixels carrying the label (%)')
    ax.set_title('(a) What the labelled pixels are, by the egg code', loc='left')
    class_legend(ax, loc='upper center', bbox_to_anchor=(0.42, -0.3), ncol=6)

    for j, (name, v, col) in enumerate((('water inside ice polygons counted', _iou(full)[1:], MUTED),
                                        ('floe-size mix of the ice only', _iou(ice)[1:], INK2))):
        y = np.arange(5) + (j - 0.5) * 0.36
        bx.barh(y, np.ones(5), height=0.28, color=GRID, linewidth=0)
        bx.barh(y, v, height=0.28, color=col, linewidth=0, label=name)
        for yy, vv in zip(y, v):
            bx.text(vv + 0.02, yy, f'{vv:.2f}', va='center', ha='left', fontsize=6.5, color=INK2)
    if args.model_cm:
        m = np.load(args.model_cm)['cm']
        m = m.sum(0) if m.ndim == 3 else m
        bx.plot(_iou(m)[1:], np.arange(5) - 0.18, 'o', ms=4.5, color=INK, mec='white', mew=1.0, label='baseline model',
                zorder=5)
    bx.set_yticks(range(5), NAMES[1:])
    bx.invert_yaxis(), bx.set_xlim(0, 1.0)
    bx.tick_params(axis='y', length=0), bx.spines['left'].set_visible(False)
    bx.set_xlabel('IoU of a perfect pixel classifier')
    bx.set_title('(b) Ceiling of the dominant-label score', loc='left')
    bx.legend(loc='upper center', bbox_to_anchor=(0.4, -0.3), ncol=1, handlelength=1.0, handleheight=1.0,
              labelspacing=0.3)
    save(fig, args.out, 'purity')


# ----------------------------------------------------------------- agreement

def agreement(args):
    d = np.load(args.agreement)
    cm = d[f'{args.kind}_floe'][:6, :6]
    cm = (cm + cm.T) / 2
    share = cm / cm.sum(1, keepdims=True).clip(1e-12)
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    im = ax.imshow(share, cmap=BLUES, vmin=0, vmax=1)
    for i in range(6):
        for j in range(6):
            if cm[i].sum() > 0:
                cell_text(ax, j, i, f'{100 * share[i, j]:.0f}', share[i, j], 1.0)
    ax.set_xticks(range(6), NAMES)
    ax.set_yticks(range(6), [f'{n}  {a / 1e3:.1f}' for n, a in zip(NAMES, cm.sum(1))])
    ax.set_xticks(np.arange(7) - 0.5, minor=True), ax.set_yticks(np.arange(7) - 0.5, minor=True)
    ax.grid(which='minor', color='white', lw=1.5), ax.tick_params(which='both', length=0)
    ax.set_xlabel('class in the other chart'), ax.set_ylabel('class in one chart, area (10$^3$ km$^2$)')
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03, ticks=[0, 0.5, 1])
    cb.ax.set_yticklabels(['0', '50', '100']), cb.outline.set_visible(False)
    cb.set_label('share of the row (%)')
    save(fig, args.out, f'agreement_{args.kind}')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out', default=osp.join(os.environ.get('FLOE_FIG_DIR', 'figs'), 'data_paper'))
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('coverage')
    p.add_argument('--table', required=True, help='tools.build.polygon_table output')
    p.add_argument('--data-dir', required=True)
    p.add_argument('--charts', required=True, help='DMI chart root with index.json')
    p.add_argument('--land', default=None, help='Natural Earth land polygons (zip or shp)')
    p = sub.add_parser('season')
    p.add_argument('--table', required=True)
    p = sub.add_parser('example')
    p.add_argument('--scene', required=True)
    p.add_argument('--step', type=int, default=6, help='decimation for display')
    p.add_argument('--polygons', type=int, default=7, help='largest ice polygons to show in (d)')
    p.add_argument('--zoom', type=int, default=500, help='side of the full-resolution crop in pixels')
    p.add_argument('--zoom-polygon', default='A', help='letter of the polygon the crop is taken from')
    p = sub.add_parser('purity')
    p.add_argument('--purity', required=True, help='tools.build.label_purity output')
    p.add_argument('--model-cm', default=None, help='confusion matrices of a model (tools.evaluate --dump-cm)')
    p = sub.add_parser('agreement')
    p.add_argument('--agreement', required=True, help='tools.build.chart_agreement output')
    p.add_argument('--kind', default='neighbours', choices=('neighbours', 'same_region'))
    args = ap.parse_args()
    {'coverage': coverage, 'season': season, 'example': example, 'purity': purity, 'agreement': agreement}[args.cmd](args)


if __name__ == '__main__':
    main()
