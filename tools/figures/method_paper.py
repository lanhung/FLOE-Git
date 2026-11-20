"""Figures of the method paper (paper 2), drawn from the benchmark outputs.

    python -m tools.figures.method_paper stats --figdata paper2_figdata.json --compare paper2_compare.json \
                                               --breakdown paper2_breakdown.json
    python -m tools.figures.method_paper example --figdata paper2_figdata.json --data-dir $FLOE_V2_DIR \
                                               --preds DIR        # DIR/<arm>/<scene>.npz from tools.evaluate --dump-preds

`stats` writes classes, forest, regions, mixed, confusion and curves; `example` writes the scene
panels; `studyarea` (build environment: geopandas, rasterio) the map of the benchmark scenes. Inputs come from tools/analysis/paper2_figdata.py, paper2_compare.py and paper2_breakdown.py.

Colours follow the dataset paper (tools.figures.data_paper): floe size is an ordered scale in one
hue with open water in a neutral grey. The two models are drawn as reference and emphasis: the
baseline in grey, the combined model in blue. Differences use blue (combined model higher) and
orange (lower) around a neutral midpoint.
"""
import argparse
import json
import os
import os.path as osp

import numpy as np

from tools.figures.data_paper import (AXIS, CLASS_COLORS, GRID, ICE, INK, INK2, MUTED, NAMES, REGION_SHORT,
                                      REGIONS, WATER, plt, save)
from matplotlib.colors import LinearSegmentedColormap, ListedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

BASE, ARM = '#898781', '#2a78d6'          # reference (grey) and emphasis (blue)
POS, NEG, MID = '#2a78d6', '#eb6834', '#f0efec'
DIV = LinearSegmentedColormap.from_list('div', [NEG, '#f6c9b5', MID, '#b9d3f3', POS])
BLUES = LinearSegmentedColormap.from_list('blues', ['#ffffff', '#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5',
                                                    '#256abf', '#184f95', '#0d366b'])
CEILING = (0.898, 0.237, 0.290, 0.412, 0.480, 0.887)   # perfect classifier vs dominant labels, dataset paper
SETS = (('folds', 'Held-out folds'), ('2026', 'Held-out year 2026'))
UNLAB = '#ecebe6'


def model_legend(fig, extra=(), **kw):
    h = [Patch(facecolor=BASE, label='Baseline'), Patch(facecolor=ARM, label='Combined model')] + list(extra)
    return fig.legend(handles=h, loc='lower center', ncol=len(h), handlelength=1.0, handleheight=1.0,
                      columnspacing=1.6, handletextpad=0.5, **kw)


# ------------------------------------------------------------------ per-class IoU with ceilings

def classes(fd, out):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True, gridspec_kw={'wspace': 0.06})
    x, w = np.arange(6), 0.34
    for ax, (key, title) in zip(axes, SETS):
        a, b = (np.array(fd['confusion'][key][m]['iou']) for m in fd['arms'])
        ax.bar(x - w / 2 - 0.02, a, w, color=BASE, zorder=2)
        ax.bar(x + w / 2 + 0.02, b, w, color=ARM, zorder=2)
        for i in range(6):
            ax.plot([i - w - 0.06, i + w + 0.06], [CEILING[i]] * 2, color=INK, lw=1.2, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{n}\n{b[i] - a[i]:+.3f}' for i, n in enumerate(NAMES)])
        ax.set_title(title, loc='left')
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', color=GRID, lw=0.5, zorder=0)
        ax.tick_params(axis='x', length=0)
    axes[0].set_ylabel('IoU')
    model_legend(fig, [Line2D([0], [0], color=INK, lw=1.2, label='Classifier right on every pixel (dataset paper)')],
                 bbox_to_anchor=(0.5, -0.17))
    save(fig, out, 'classes')


# ------------------------------------------------------------------ paired differences

def forest(cmp_, out):
    rows = [('comp', 'Composition score'), ('miou', 'Mean IoU'), None,
            ('miou_ice', 'IoU, ice classes'), ('hard', 'IoU, small + medium'), ('wf1', 'Weighted F1')]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=True, sharex=True, gridspec_kw={'wspace': 0.06})
    for ax, (key, title) in zip(axes, SETS):
        y, ticks, labels = 0.0, [], []
        for r in rows:
            if r is None:
                y -= 0.55
                ax.axhline(y + 0.2, color=GRID, lw=0.6)
                continue
            d = cmp_[key]['diff'][r[0]]
            pu = np.array(d['per_unit'])
            jit = (np.linspace(-0.22, 0.22, len(pu)) if len(pu) > 3 else np.linspace(-0.12, 0.12, len(pu)))
            ax.scatter(pu, y + jit, s=7, color=BASE, alpha=0.75, lw=0, zorder=2)
            ax.plot(d['boot95'], [y, y], color=ARM, lw=2.2, solid_capstyle='round', zorder=3)
            ax.plot(d['t95'], [y - 0.3, y - 0.3], color=INK2, lw=0.9, zorder=3)
            ax.scatter([d['mean']], [y], s=30, color=ARM, edgecolor='white', lw=0.8, zorder=4)
            ticks.append(y)
            labels.append(r[1])
            y -= 1.0
        ax.axvline(0, color=INK, lw=0.7, zorder=1)
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels)
        ax.set_title(title, loc='left')
        ax.grid(axis='x', color=GRID, lw=0.5, zorder=0)
        ax.tick_params(axis='y', length=0)
        ax.set_xlabel('Combined model minus baseline')
        ax.set_ylim(y + 0.45, 0.6)
    axes[0].text(0.985, 0.965, 'primary measures', transform=axes[0].transAxes, ha='right', va='top', fontsize=6.5,
                 color=MUTED)
    axes[0].text(0.985, 0.50, 'secondary measures', transform=axes[0].transAxes, ha='right', va='top', fontsize=6.5,
                 color=MUTED)
    units = 'seeds (left) or models (right)'
    fig.legend(handles=[Line2D([0], [0], marker='o', color=ARM, lw=2.2, markersize=5, markeredgecolor='white',
                               label='Mean and 95 % interval over scene groups'),
                        Line2D([0], [0], color=INK2, lw=0.9, label=f'95 % interval over {units}'),
                        Line2D([0], [0], marker='o', color=BASE, lw=0, markersize=3, label='Single pairs')],
               loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.17), columnspacing=1.6, handletextpad=0.5)
    save(fig, out, 'forest')


# ------------------------------------------------------------------ hard-ice IoU by region and season

def regions(bd, out):
    seasons = ('DJF', 'MAM', 'JJA', 'SON')
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.3), sharey=True, sharex=True, gridspec_kw={'wspace': 0.06})
    for ax, (key, title) in zip(axes, SETS):
        y, ticks, labels = 0.0, [], []
        for group, names, short in (('region', REGIONS, REGION_SHORT), ('season', seasons, seasons)):
            for n, s in zip(names, short):
                r = bd[key][group][n]
                ax.plot([r['base'], r['arm']], [y, y], color=AXIS, lw=1.6, zorder=2, solid_capstyle='round')
                ax.scatter([r['base']], [y], s=22, color=BASE, edgecolor='white', lw=0.7, zorder=3)
                ax.scatter([r['arm']], [y], s=22, color=ARM, edgecolor='white', lw=0.7, zorder=4)
                ax.text(max(r['base'], r['arm']) + 0.008, y, f"{r['arm'] - r['base']:+.3f}", va='center', fontsize=6.5,
                        color=INK2)
                ticks.append(y)
                labels.append(s)
                y -= 1.0
            y -= 0.6
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels)
        ax.set_title(title, loc='left')
        ax.grid(axis='x', color=GRID, lw=0.5, zorder=0)
        ax.tick_params(axis='y', length=0)
        ax.set_xlabel('IoU, small + medium floes')
        ax.set_ylim(y + 1.1, 0.7)
        ax.set_xlim(0.05, 0.40)
        ax.set_xticks([0.1, 0.2, 0.3])
    model_legend(fig, bbox_to_anchor=(0.5, -0.1))
    save(fig, out, 'regions')


# ------------------------------------------------------------------ class make-up of mixed polygons

def mixed(fd, out):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 1.75), sharey=True, gridspec_kw={'wspace': 0.1})
    for ax, (key, title) in zip(axes, SETS):
        m = fd['mixed'][key]
        rows = [('Chart', m['baseline']['chart']), ('Baseline', m['baseline']['pred']),
                ('Combined model', m[fd['arms'][1]]['pred'])]
        for i, (name, v) in enumerate(rows):
            left = 0.0
            for c, share in enumerate(v):
                ax.barh(-i, share * 100, left=left * 100, height=0.62, color=CLASS_COLORS[c], edgecolor='white',
                        lw=1.2, zorder=2)
                if share >= 0.06:
                    ax.text((left + share / 2) * 100, -i, f'{share * 100:.0f}', ha='center', va='center', fontsize=6.5,
                            color='white' if c >= 3 else INK)
                left += share
        ax.set_yticks([0, -1, -2])
        ax.set_yticklabels([r[0] for r in rows])
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_title(title, loc='left')
        ax.set_xlabel('Share of the area of mixed polygons (%)')
        ax.tick_params(axis='y', length=0)
        for s in ('left',):
            ax.spines[s].set_visible(False)
    fig.legend(handles=[Patch(facecolor=c, label=n) for n, c in zip(NAMES, CLASS_COLORS)], loc='lower center', ncol=6,
               bbox_to_anchor=(0.5, -0.33), handlelength=1.0, handleheight=1.0, columnspacing=1.2, handletextpad=0.4)
    save(fig, out, 'mixed')


# ------------------------------------------------------------------ confusion matrices (2026)

def confusion(fd, out):
    a, b = (np.array(fd['confusion']['2026'][m]['cm']) for m in fd['arms'])
    ra, rb = (100 * c / c.sum(1, keepdims=True) for c in (a, b))
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.55), gridspec_kw={'wspace': 0.12})
    panels = (('(a) Baseline', ra, BLUES, 0, 100, '{:.0f}'), ('(b) Combined model', rb, BLUES, 0, 100, '{:.0f}'),
              ('(c) Combined minus baseline', rb - ra, DIV, -2.5, 2.5, '{:+.1f}'))
    for ax, (title, m, cmap, lo, hi, fmt) in zip(axes, panels):
        ax.imshow(m, cmap=cmap, vmin=lo, vmax=hi)
        for i in range(6):
            for j in range(6):
                dark = (m[i, j] > 0.55 * hi) if lo == 0 else (abs(m[i, j]) > 0.6 * hi)
                txt = fmt.format(m[i, j])
                if lo != 0 and abs(m[i, j]) < 0.05:
                    txt = '0'
                ax.text(j, i, txt, ha='center', va='center', fontsize=6, color='white' if dark else INK)
        ax.set_xticks(range(6))
        ax.set_xticklabels([n[:3] for n in NAMES])
        ax.set_yticks(range(6))
        ax.set_yticklabels(NAMES if ax is axes[0] else [])
        ax.set_title(title, loc='left')
        ax.set_xlabel('Predicted')
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xticks(np.arange(-0.5, 6), minor=True)
        ax.set_yticks(np.arange(-0.5, 6), minor=True)
        ax.grid(which='minor', color='white', lw=1.2)
        ax.tick_params(which='minor', length=0)
    axes[0].set_ylabel('Label')
    save(fig, out, 'confusion')


# ------------------------------------------------------------------ validation curves

def curves(fd, out):
    fig, axes = plt.subplots(1, 5, figsize=(7.0, 1.9), sharey=True, gridspec_kw={'wspace': 0.08})
    for k, ax in enumerate(axes):
        for arm, col, z in ((fd['arms'][0], BASE, 2), (fd['arms'][1], ARM, 3)):
            for r in fd['runs']:
                if r['arm'] == arm and r['fold'] == k:
                    c = np.array(r['curve'])
                    ax.plot(2 * np.arange(1, len(c) + 1), c, color=col, lw=0.9, alpha=0.9, zorder=z)
        ax.set_title(f'Fold {k}', loc='left')
        ax.set_xlabel('Epoch')
        ax.set_xlim(0, 200)
        ax.set_xticks([50, 150])
        ax.set_ylim(0.30, 0.45)
        ax.grid(axis='y', color=GRID, lw=0.5, zorder=0)
    axes[0].set_ylabel('Validation mean IoU')
    fig.legend(handles=[Line2D([0], [0], color=BASE, lw=1.2, label='Baseline (3 seeds)'),
                        Line2D([0], [0], color=ARM, lw=1.2, label='Combined model (3 seeds)')],
               loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.2), columnspacing=1.6)
    save(fig, out, 'curves')


# ------------------------------------------------------------------ example scenes

def example(args):
    import h5py
    import hdf5plugin  # noqa: F401
    fd = json.load(open(args.figdata))
    step = args.step
    cls_cmap = ListedColormap(list(CLASS_COLORS) + [UNLAB, '#ffffff'])     # 0-5 classes, 6 unlabelled ice, 7 outside
    diff_cmap = ListedColormap(['#ffffff', MID, POS, NEG])                  # outside, same outcome, only arm, only base
    scenes = [(c, fd['examples'][c]) for c in ('east', 'west')]
    fig, axes = plt.subplots(len(scenes), 6, figsize=(7.0, 1.3 * len(scenes) + 0.1), gridspec_kw={'wspace': 0.04, 'hspace': 0.1})
    titles = ('(a) HH', '(b) HV', '(c) Floe-size label', '(d) Baseline', '(e) Combined', '(f) One model right')
    for row, (coast, ex) in enumerate(scenes):
        n = ex['scene']
        with h5py.File(osp.join(args.data_dir, n + '.nc'), 'r') as f:
            hh = f['nersc_sar_primary'][::step, ::step]
            hv = f['nersc_sar_secondary'][::step, ::step]
            lab = f['FLOE'][::step, ::step]
            pid = f['polygon_id'][::step, ::step]
            valid = f['global_valid_mask'][::step, ::step] > 0
        charted = (pid >= 0) & valid
        pa = np.load(osp.join(args.preds, fd['arms'][0], n + '.npz'))['FLOE'][::step, ::step]
        pb = np.load(osp.join(args.preds, fd['arms'][1], n + '.npz'))['FLOE'][::step, ::step]
        has = lab <= 5

        def show(p):
            m = np.full(p.shape, 7, np.uint8)
            m[charted] = p[charted]
            return m
        lab_show = np.full(lab.shape, 7, np.uint8)
        lab_show[charted] = 6
        lab_show[has & valid] = lab[has & valid]
        d = np.zeros(lab.shape, np.uint8)
        d[has & valid] = 1
        d[has & valid & (pb == lab) & (pa != lab)] = 2
        d[has & valid & (pa == lab) & (pb != lab)] = 3
        lo, hi = np.percentile(hh[valid], [2, 98])
        lo2, hi2 = np.percentile(hv[valid], [2, 98])
        ims = ((np.where(valid, hh, np.nan), dict(cmap='gray', vmin=lo, vmax=hi)),
               (np.where(valid, hv, np.nan), dict(cmap='gray', vmin=lo2, vmax=hi2)),
               (lab_show, dict(cmap=cls_cmap, vmin=0, vmax=7)), (show(pa), dict(cmap=cls_cmap, vmin=0, vmax=7)),
               (show(pb), dict(cmap=cls_cmap, vmin=0, vmax=7)), (d, dict(cmap=diff_cmap, vmin=0, vmax=3)))
        for col, (img, kw) in enumerate(ims):
            ax = axes[row, col]
            ax.imshow(img, interpolation='nearest', **kw)
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(True)
                s.set_color(AXIS)
                s.set_linewidth(0.5)
            if row == 0:
                ax.set_title(titles[col], loc='left', fontsize=7)
        date = f'{n[23:25]}.{n[21:23]}.{n[17:21]}'
        axes[row, 0].set_ylabel(f"{ex['region']}\n{date}", fontsize=7, color=INK2)
        ok_b = float(((pb == lab) & has & valid).sum()) / max(float((has & valid).sum()), 1)
        ok_a = float(((pa == lab) & has & valid).sum()) / max(float((has & valid).sum()), 1)
        print(f"{coast}: {n} region {ex['region']} pixel agreement with labels: baseline {ok_a:.3f}, combined {ok_b:.3f}; "
              f"only combined right {float((d == 2).sum()) / max(float((d > 0).sum()), 1):.3f}, "
              f"only baseline right {float((d == 3).sum()) / max(float((d > 0).sum()), 1):.3f}")
    h = [Patch(facecolor=c, label=n_) for n_, c in zip(NAMES, CLASS_COLORS)]
    h += [Patch(facecolor=UNLAB, edgecolor=AXIS, lw=0.4, label='Charted ice, no floe label'),
          Patch(facecolor=POS, label='Panel f: only combined right'),
          Patch(facecolor=NEG, label='Panel f: only baseline right')]
    fig.legend(handles=h, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.11 if len(scenes) > 1 else -0.2),
               handlelength=1.0, handleheight=1.0, columnspacing=1.2, handletextpad=0.4)
    save(fig, args.out, 'example')


# ------------------------------------------------------------------ what the chart gives for one scene

def targets(args):
    import h5py
    import hdf5plugin  # noqa: F401
    # the test scene with the largest charted ice area that has a composition but no floe-size label
    t = np.load(args.table)
    test = {x[:-3] for x in json.load(open(osp.join(args.split, 'test.json')))}
    only = ((t['floe'] == 255) & (np.nan_to_num(t['unknown'], nan=1.0) <= 0.5)
            & (np.nan_to_num(t['composition'][:, 1:].sum(1)) > 0))
    area = np.bincount(t['scene'][only], weights=t['pixels'][only].astype(np.float64), minlength=len(t['scenes']))
    area[[i for i, x in enumerate(t['scenes']) if str(x) not in test]] = -1
    n, step = str(t['scenes'][int(np.argmax(area))]), args.step
    print(f'targets scene: {n} ({area.max() / 1e6:.1f} Mpx of composition-only ice)')
    with h5py.File(osp.join(args.data_dir, n + '.nc'), 'r') as f:
        hh, hv = (f[k][::step, ::step] for k in ('nersc_sar_primary', 'nersc_sar_secondary'))
        floe, sic, sod, pid = (f[k][::step, ::step] for k in ('FLOE', 'SIC', 'SOD', 'polygon_id'))
        valid = f['global_valid_mask'][::step, ::step] > 0
        pfloe, punk, pcomp = f['poly_floe'][()], f['poly_unknown'][()], f['poly_composition'][()]
    charted = (pid >= 0) & valid
    cls_cmap = ListedColormap(list(CLASS_COLORS) + [UNLAB, '#ffffff'])

    def classes_map(v, nmax):
        m = np.full(v.shape, 7, np.uint8)
        m[charted] = 6
        ok = charted & (v <= nmax)
        m[ok] = v[ok]
        return m
    kind = np.full(len(pfloe), 2, np.uint8)                       # 2: no floe-size information
    mix = (np.nan_to_num(punk, nan=1.0) <= 0.5) & (np.nan_to_num(pcomp[:, 1:].sum(1)) > 0)
    kind[mix] = 1                                                 # 1: composition only
    kind[(pfloe >= 1) & (pfloe <= 5)] = 0                         # 0: floe-size label (and composition)
    kind[pfloe == 0] = 3                                          # 3: open water
    info = np.full(pid.shape, 4, np.uint8)
    info[charted] = kind[pid[charted]]
    info_cmap = ListedColormap([ARM, NEG, UNLAB, WATER, '#ffffff'])
    sic_cmap = ListedColormap([WATER] + [BLUES(0.12 + 0.088 * i) for i in range(1, 11)] + [UNLAB, '#ffffff'])
    sic_show = np.full(sic.shape, 12, np.uint8)
    sic_show[charted] = 11
    ok = charted & (sic <= 10)
    sic_show[ok] = sic[ok]

    fig, axes = plt.subplots(2, 3, figsize=(7.0, 5.3), gridspec_kw={'wspace': 0.05, 'hspace': 0.5})
    lo, hi = np.percentile(hh[valid], [2, 98])
    lo2, hi2 = np.percentile(hv[valid], [2, 98])
    sod_names = ('Water', 'New', 'Young', 'Thin first-year', 'Thick first-year', 'Old')
    panels = (
        ('(a) HH backscatter', np.where(valid, hh, np.nan), dict(cmap='gray', vmin=lo, vmax=hi), None),
        ('(b) HV backscatter', np.where(valid, hv, np.nan), dict(cmap='gray', vmin=lo2, vmax=hi2), None),
        ('(c) Floe-size label', classes_map(floe, 5), dict(cmap=cls_cmap, vmin=0, vmax=7),
         [Patch(facecolor=c, label=x) for x, c in zip(NAMES, CLASS_COLORS)] + [Patch(facecolor=UNLAB, label='No label')]),
        ('(d) Total concentration (tenths)', sic_show, dict(cmap=sic_cmap, vmin=0, vmax=12), 'sic'),
        ('(e) Stage of development', classes_map(sod, 5), dict(cmap=cls_cmap, vmin=0, vmax=7),
         [Patch(facecolor=c, label=x) for x, c in zip(sod_names, CLASS_COLORS)] + [Patch(facecolor=UNLAB, label='No label')]),
        ('(f) Floe-size information', info, dict(cmap=info_cmap, vmin=0, vmax=4),
         [Patch(facecolor=ARM, label='Label and composition'), Patch(facecolor=NEG, label='Composition only'),
          Patch(facecolor=UNLAB, label='None'), Patch(facecolor=WATER, label='Open water')]),
    )
    for ax, (title, img, kw, handles) in zip(axes.ravel(), panels):
        ax.imshow(img, interpolation='nearest', **kw)
        ax.set_xticks([]), ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_color(AXIS)
            sp.set_linewidth(0.5)
        ax.set_title(title, loc='left')
        if handles == 'sic':
            cax = ax.inset_axes([0.04, -0.1, 0.92, 0.05])
            cax.imshow(np.arange(11)[None, :], cmap=ListedColormap([sic_cmap(i) for i in range(11)]), aspect='auto')
            cax.set_xticks(range(11)), cax.set_yticks([])
            cax.tick_params(length=0, labelsize=6)
            for sp in cax.spines.values():
                sp.set_visible(False)
        elif handles:
            ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.02), ncol=3 if len(handles) > 4 else 2,
                      fontsize=6, handlelength=0.9, handleheight=0.9, columnspacing=0.8, handletextpad=0.35,
                      labelspacing=0.3)
    area = {k: float((info == k).sum()) for k in range(4)}
    ice = area[0] + area[1] + area[2]
    print(f"{n}: charted ice area with floe label {area[0] / ice:.3f}, composition only {area[1] / ice:.3f}, none {area[2] / ice:.3f}; "
          f"SIC labelled {float((charted & (sic <= 10) & (info < 3)).sum()) / ice:.3f}, SOD labelled {float((charted & (sod <= 5) & (info < 3)).sum()) / ice:.3f}")
    save(fig, args.out, 'targets')


# ------------------------------------------------------------------ model and supervision

def schematic(fd, out):
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    cov = fd['coverage']
    fig, ax = plt.subplots(figsize=(7.0, 2.55))
    ax.set_xlim(0, 100), ax.set_ylim(0, 40), ax.axis('off')
    base_kw = dict(facecolor='#f4f3ef', edgecolor=BASE, lw=0.9)
    add_kw = dict(facecolor='#e6f0fc', edgecolor=ARM, lw=0.9)

    def box(x, y, w, h, text, kw, size=6.8):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.8', **kw))
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=size, color=INK, linespacing=1.25)

    def arrow(p, q, col=INK2):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=7, lw=0.8, color=col, shrinkA=0, shrinkB=0))

    box(0, 10, 17.5, 20, 'Sentinel-1 EW patch\n256 \u00d7 256 px, 80 m\n\nHH, HV,\nincidence angle,\nvalid mask, HH \u2212 HV', base_kw, 6.2)
    box(22, 12, 12, 16, 'Encoder\nMiT-B2\n(shared)', base_kw, 6.4)
    box(38.5, 28.5, 20, 9, 'Floe-size decoder\n6 classes', base_kw, 6.4)
    box(38.5, 15.5, 20, 9, 'Concentration decoder\ntenths', add_kw, 6.4)
    box(38.5, 2.5, 20, 9, 'Stage decoder\n6 classes', add_kw, 6.4)
    box(63, 33.6, 37, 5.2, f"Cross-entropy, pixels with a floe-size label ({100 * cov['floe_label']:.0f} %)", base_kw, 5.9)
    box(63, 26.4, 37, 5.2, f"Composition loss, chart polygons ({100 * cov['composition']:.0f} %)", add_kw, 5.9)
    box(63, 17.4, 37, 5.2, f"0.5 \u00d7 cross-entropy, total concentration ({100 * cov['sic_label']:.0f} %)", add_kw, 5.9)
    box(63, 4.4, 37, 5.2, f"0.5 \u00d7 cross-entropy, dominant stage ({100 * cov['sod_label']:.0f} %)", add_kw, 5.9)
    arrow((17.5, 20), (22, 20))
    for y in (33, 20, 7):
        arrow((34, 20), (38.5, y))
    arrow((58.5, 34.5), (63, 36.2))
    arrow((58.5, 31.5), (63, 29.0), ARM)
    arrow((58.5, 20), (63, 20), ARM)
    arrow((58.5, 7), (63, 7), ARM)
    ax.text(81.5, 12.6, 'in brackets: share of the charted ice area\nthat carries the target', ha='center', va='center',
            fontsize=5.8, color=MUTED, linespacing=1.2)
    fig.legend(handles=[Patch(label='Baseline, and all that is used at inference', **base_kw),
                        Patch(label='Added for training in the combined model', **add_kw)],
               loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.09), handlelength=1.2, handleheight=1.0,
               columnspacing=1.8)
    save(fig, out, 'schematic')


# ------------------------------------------------------------------ study area and split

def studyarea(args):
    import geopandas as gpd
    import h5py
    import hdf5plugin  # noqa: F401
    from pyproj import Transformer
    from rasterio import features
    from rasterio.enums import MergeAlg
    from rasterio.transform import from_origin
    from shapely.geometry import Polygon, box

    fd = json.load(open(args.figdata))
    folds = sorted({n[:-3] for k in range(5) for n in json.load(open(osp.join(args.split, f'fold{k}_val.json')))})
    test = sorted(n[:-3] for n in json.load(open(osp.join(args.split, 'test.json'))))
    to_xy = Transformer.from_crs(4326, 3413, always_xy=True).transform

    def foot(n):
        with h5py.File(osp.join(args.data_dir, n + '.nc'), 'r') as f:
            lat, lon = f['sar_grid_lat'][...], f['sar_grid_lon'][...]
        ring = [(lon[i, j], lat[i, j]) for i, j in ((0, 0), (0, -1), (-1, -1), (-1, 0))]
        return Polygon(zip(*to_xy(*zip(*ring))))
    feet = {n: foot(n) for n in folds + test}
    bounds = np.array([p.bounds for p in feet.values()])
    res = 10e3
    x0, y0 = np.floor(bounds[:, :2].min(0) / 1e5) * 1e5 - 1e5
    x1, y1 = np.ceil(bounds[:, 2:].max(0) / 1e5) * 1e5 + 1e5
    shape = (int((y1 - y0) / res), int((x1 - x0) / res))
    land = gpd.read_file(args.land).clip(box(-179.9, 45, 179.9, 89.9)).to_crs(3413) if args.land else None
    index = json.load(open(osp.join(args.charts, 'index.json')))

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.75), gridspec_kw={'width_ratios': [1, 1, 1.25], 'wspace': 0.16})
    panels = ((folds, '(a) Folds, 2022\u20132025'), (test, '(b) Test year 2026'))
    for ax, (names, title) in zip(axes[:2], panels):
        count = features.rasterize([(feet[n], 1) for n in names], out_shape=shape,
                                   transform=from_origin(x0, y1, res, res), merge_alg=MergeAlg.add, dtype='int32')
        for lat_ in (60, 70, 80):
            ax.plot(*to_xy(np.linspace(-180, 180, 361), np.full(361, lat_)), color=GRID, lw=0.4, zorder=0)
        for lon_ in range(-180, 180, 30):
            ax.plot(*to_xy(np.full(50, lon_), np.linspace(50, 90, 50)), color=GRID, lw=0.4, zorder=0)
        im = ax.imshow(np.ma.masked_equal(count, 0), extent=(x0, x1, y0, y1), cmap=BLUES, vmin=0,
                       interpolation='nearest', zorder=1)
        if land is not None:
            land.plot(ax=ax, facecolor='#ecebe6', edgecolor=AXIS, linewidth=0.3, zorder=2)
        for reg, short in zip(REGIONS, REGION_SHORT):
            boxes = np.array([r['bbox'] for r in index if r['name'].split('_')[1] == reg])
            if len(boxes):
                c = to_xy(np.median((boxes[:, 0] + boxes[:, 2]) / 2), np.median((boxes[:, 1] + boxes[:, 3]) / 2))
                ax.annotate(short, c, ha='center', va='center', fontsize=6, color=INK, zorder=4,
                            bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.8))
        if names is test:
            for coast in ('east', 'west'):
                ax.plot(*feet[fd['examples'][coast]['scene']].exterior.xy, color=NEG, lw=1.0, zorder=5)
        ax.set_xlim(x0, x1), ax.set_ylim(y0, y1), ax.set_aspect('equal')
        ax.set_xticks([]), ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True)
        cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02, orientation='horizontal')
        cb.outline.set_visible(False)
        cb.set_label('scenes covering a place', fontsize=6.5)
        ax.set_title(title, loc='left')
    ax = axes[0]
    ax.plot([x0 + 1.0e5, x0 + 6.0e5], [y0 + 1.1e5] * 2, color=INK, lw=1.2, solid_capstyle='butt', zorder=5)
    ax.text(x0 + 3.5e5, y0 + 1.5e5, '500 km', ha='center', va='bottom', fontsize=6, zorder=5)

    bx = axes[2]
    years = np.arange(2022, 2027)
    grid = np.zeros((len(years), 12), int)
    for n in folds + test:
        grid[int(n[17:21]) - 2022, int(n[21:23]) - 1] += 1
    bx.imshow(grid, cmap=BLUES, vmin=0, aspect='auto')
    for i in range(len(years)):
        for j in range(12):
            if grid[i, j]:
                bx.text(j, i, str(grid[i, j]), ha='center', va='center', fontsize=6,
                        color='white' if grid[i, j] > 0.55 * grid.max() else INK)
    bx.set_xticks(range(12), list('JFMAMJJASOND'))
    bx.set_yticks(range(len(years)), [f'{y}' if y < 2026 else '2026\n(test)' for y in years])
    bx.set_xticks(np.arange(1, 12) - 0.5, minor=True), bx.set_yticks(np.arange(1, len(years)) - 0.5, minor=True)
    bx.grid(which='minor', color='white', lw=1.5), bx.tick_params(which='both', length=0)
    for sp in bx.spines.values():
        sp.set_visible(False)
    bx.set_xlim(-0.5, 11.5)
    bx.set_title('(c) Scenes per month', loc='left')
    save(fig, args.out, 'studyarea')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out', default=osp.join(os.environ.get('FLOE_FIG_DIR', 'figs'), 'method_paper'))
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('stats')
    p.add_argument('--figdata', required=True)
    p.add_argument('--compare', required=True)
    p.add_argument('--breakdown', required=True)
    p = sub.add_parser('example')
    p.add_argument('--figdata', required=True)
    p.add_argument('--data-dir', required=True)
    p.add_argument('--preds', required=True)
    p.add_argument('--step', type=int, default=6)
    p = sub.add_parser('targets')
    p.add_argument('--figdata', required=True)
    p.add_argument('--data-dir', required=True)
    p.add_argument('--table', required=True, help='polygon table of the dataset')
    p.add_argument('--split', required=True)
    p.add_argument('--step', type=int, default=4)
    p = sub.add_parser('studyarea')
    p.add_argument('--figdata', required=True)
    p.add_argument('--data-dir', required=True)
    p.add_argument('--split', required=True)
    p.add_argument('--charts', required=True)
    p.add_argument('--land', default=None)
    args = ap.parse_args()
    if args.cmd == 'example':
        example(args)
        return
    if args.cmd == 'studyarea':
        studyarea(args)
        return
    if args.cmd == 'targets':
        targets(args)
        return
    fd, cmp_, bd = (json.load(open(f)) for f in (args.figdata, args.compare, args.breakdown))
    classes(fd, args.out)
    forest(cmp_, args.out)
    regions(bd, args.out)
    mixed(fd, args.out)
    confusion(fd, args.out)
    curves(fd, args.out)
    schematic(fd, args.out)


if __name__ == '__main__':
    main()
