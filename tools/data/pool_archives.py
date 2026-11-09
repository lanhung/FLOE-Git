"""Pool several scene archives for joint training.

Creates one directory of symlinks (scene names must not collide) and one tile index
that concatenates the per-archive indexes, so the existing loader and tile sampler
work unchanged. Archives must already share the file format and z-score space
(tools.data.convert_ai4arctic does that for AutoICE).

    python -m tools.data.pool_archives \
        --archive $FLOE_DATA_DIR $FLOE_TILE_INDEX \
        --archive $FLOE_A4A_DIR $FLOE_A4A_TILE_INDEX \
        --dst $FLOE_POOL_DIR --index-out $FLOE_POOL_TILE_INDEX
"""
import argparse
import os
import os.path as osp

import numpy as np

PER_TILE = ('row', 'col', 'valid_ratio', 'label_valid_ratio', 'hist', 'hist_w')


def merge_indexes(paths):
    parts = [np.load(p, allow_pickle=False) for p in paths]
    for k in ('tile_size', 'patch_size'):
        vals = {int(p[k]) for p in parts}
        if len(vals) != 1:
            raise ValueError(f'{k} differs between indexes: {sorted(vals)}')
    if len({p['hist'].shape[1] for p in parts}) != 1:
        raise ValueError('class histograms have different widths')

    files, scene_id, offset = [], [], 0
    for p in parts:
        names = [str(f) for f in p['files']]
        files += names
        scene_id.append(p['scene_id'].astype(np.int64) + offset)
        offset += len(names)
    if len(set(files)) != len(files):
        dup = sorted({f for f in files if files.count(f) > 1})
        raise ValueError(f'scene names collide across archives, e.g. {dup[:3]}')

    out = {k: np.concatenate([p[k] for p in parts]) for k in PER_TILE}
    out['scene_id'] = np.concatenate(scene_id).astype(parts[0]['scene_id'].dtype)
    out['files'] = np.array(files)
    out['tile_size'] = parts[0]['tile_size']
    out['patch_size'] = parts[0]['patch_size']
    return out


def link_archives(dirs, dst):
    os.makedirs(dst, exist_ok=True)
    n_new = 0
    for d in dirs:
        for f in sorted(os.listdir(d)):
            if not f.endswith('.nc'):
                continue
            src, link = osp.abspath(osp.join(d, f)), osp.join(dst, f)
            if osp.lexists(link):
                if osp.realpath(link) != osp.realpath(src):
                    raise ValueError(f'{link} already points to {osp.realpath(link)}')
                continue
            os.symlink(src, link)
            n_new += 1
    return n_new


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--archive', nargs=2, action='append', required=True,
                    metavar=('DATA_DIR', 'TILE_INDEX'))
    ap.add_argument('--dst', required=True, help='directory for the symlinks')
    ap.add_argument('--index-out', required=True)
    args = ap.parse_args()

    merged = merge_indexes([idx for _, idx in args.archive])
    n_new = link_archives([d for d, _ in args.archive], args.dst)
    tmp = args.index_out + '.tmp.npz'
    np.savez_compressed(tmp, **merged)
    os.replace(tmp, args.index_out)
    print(f'{len(merged["files"])} scenes, {len(merged["row"])} tiles -> {args.index_out}; '
          f'{n_new} new links in {args.dst}')


if __name__ == '__main__':
    main()
