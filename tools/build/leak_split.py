"""A leaking variant of one fold, to measure what a split by scene would hide.

The grouped split (tools.build.split) keeps the frames of one pass, of one chart and of
one region and week on the same side. This script breaks that rule for one fold in a
controlled way. In every validation group with two or more scenes, every second scene
in time order goes to the training set; as many training scenes, drawn at random, are
taken out, so the training set keeps its size. A model trained on these lists has seen
neighbours of the validation scenes that stay; the model trained on the grouped lists
has not. Both are then scored on the same scenes:

  exposed   validation scenes that stay and whose group now has a scene in training;
  control   validation scenes that are alone in their group, for which nothing changes.

The control scenes show how much of a difference is due to the changed training set
alone. Everything is fixed by the grouped split and --seed; nothing is chosen by score.

    python -m tools.build.leak_split --log $FLOE_ROOT/data_v2/archive_log.jsonl \
        --data-dir $FLOE_ROOT/data_v2 --split datalists/v2_2023 --fold 0 \
        --out datalists/v2_2023_leak

Writes fold<K>_train.json and fold<K>_val.json (exposed and control scenes) for
tools.train, and leak_fold<K>.json with the scenes by role and the group of each
validation scene (for tools.analysis.leak_effect).
"""
import argparse
import json
import os
import os.path as osp

import numpy as np

from tools.build.split import groups_of, load_scenes


def start_time(name):
    return name.split('_')[4]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--log', required=True)
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--split', required=True, help='directory of the grouped split')
    ap.add_argument('--fold', type=int, default=0)
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    def names(kind):
        files = json.load(open(osp.join(args.split, f'fold{args.fold}_{kind}.json')))
        return [f[:-3] for f in files]

    train, val = names('train'), names('val')
    scenes = load_scenes(args.log, args.data_dir)
    missing = [n for n in train + val if n not in scenes]
    if missing:
        raise SystemExit(f'{len(missing)} scenes of the split are not in the archive log, e.g. {missing[:2]}')
    # Groups as the split saw them: over exactly the scenes of the split.
    groups = groups_of({n: scenes[n] for n in train + val})
    in_val = set(val)
    val_groups = [sorted(g, key=start_time) for g in groups if g[0] in in_val]
    if any(n not in in_val for g in val_groups for n in g) or sum(map(len, val_groups)) != len(val):
        raise SystemExit('the validation list does not hold whole groups: not a split of tools.build.split')

    moved, exposed, control, group_of = [], [], [], {}
    for gid, g in enumerate(sorted(val_groups)):
        for n in g:
            group_of[n] = gid
        if len(g) == 1:
            control += g
        else:
            exposed += g[0::2]
            moved += g[1::2]

    rng = np.random.default_rng(args.seed)
    removed = sorted(rng.choice(sorted(train), size=len(moved), replace=False).tolist())
    new_train = sorted((set(train) - set(removed)) | set(moved))
    new_val = sorted(exposed + control)
    assert len(new_train) == len(train) and not set(new_train) & set(new_val)

    os.makedirs(args.out, exist_ok=True)
    for kind, lst in (('train', new_train), ('val', new_val)):
        json.dump([n + '.nc' for n in lst], open(osp.join(args.out, f'fold{args.fold}_{kind}.json'), 'w'), indent=1)
    report = {'split': args.split, 'fold': args.fold, 'seed': args.seed,
              'exposed': sorted(exposed), 'control': sorted(control), 'moved': sorted(moved),
              'removed': removed, 'group': group_of}
    json.dump(report, open(osp.join(args.out, f'leak_fold{args.fold}.json'), 'w'), indent=1)

    sizes = [len(g) for g in val_groups]
    print(f'fold {args.fold} of {args.split}: {len(val)} validation scenes in {len(val_groups)} groups '
          f'({sizes.count(1)} of one scene, largest {max(sizes)})')
    print(f'moved to training {len(moved)}, taken out of training {len(removed)}; '
          f'training {len(new_train)} scenes, validation {len(new_val)} '
          f'({len(exposed)} exposed, {len(control)} control) -> {args.out}')


if __name__ == '__main__':
    main()
