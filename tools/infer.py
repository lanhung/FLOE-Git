r"""Run a trained legacy (UNet/Swin) checkpoint on a scene list via floe.evaluation.test.

    python -m tools.infer CONFIG CHECKPOINT --wandb-project P [--wandb-disabled] \
        [--mode apply|test|val] [--scene-list-file LIST.json] [--work-dir DIR]

'apply' runs on unlabelled scenes; 'test' and 'val' also score against the labels.
"""

__author__ = 'Muhammed Patel'
__contributor__ = 'Xinwwei chen, Fernando Pena Cantu,Javier Turnes, Eddie Park'
__copyright__ = ['university of waterloo']
__contact__ = ['m32patel@uwaterloo.ca', 'xinweic@uwaterloo.ca']
__version__ = '1.0.0'
__date__ = '2024-04-05'

import argparse
import json
import os.path as osp
import shutil
from icecream import ic
import pathlib

import torch
from mmcv import Config, mkdir_or_exist

import wandb
from floe.loaders import get_variable_options, val_scoped_options
from floe.models.legacy.unet import UNet, Sep_feat_dif_stages
from floe.models.legacy.swin_transformer import SwinTransformer
from floe.utils import colour_str
from floe.evaluation import test


def parse_args():
    parser = argparse.ArgumentParser(description='Run a trained FLOE model on scenes (inference or evaluation)')

    parser.add_argument('config', type=pathlib.Path, help='train config file path',)
    parser.add_argument('checkpoint', type=pathlib.Path, help='checkpoint path of the model',)
    parser.add_argument('--wandb-project', required=True, help='Name of wandb project')
    parser.add_argument('--wandb-disabled', action='store_true',
                        help="完全关闭 wandb 记录（本地推理无需上传时推荐）")
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--mode', choices=['val', 'test', 'apply'], default='apply',
                        help="'apply': 真实推理（无标签的 NC）；"
                             "'test'/'val': 带 GT 的评估")
    parser.add_argument('--scene-list-key', default=None,
                        help="配置中存放 JSON 路径的 key；默认 apply/test→'test_path'、val→'val_path'")
    parser.add_argument('--scene-list-file', default=None,
                        help="直接传入场景列表 JSON 文件路径，优先级高于 --scene-list-key")
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    checkpoint_path = args.checkpoint
    ic(args.config)
    cfg = Config.fromfile(args.config)
    train_options = cfg.train_options
    train_options = get_variable_options(train_options)
    id = wandb.util.generate_id()

    # work_dir priority: CLI > config file > config filename
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        if not train_options['cross_val_run']:
            cfg.work_dir = osp.join('./work_dir',
                                    osp.splitext(osp.basename(args.config))[0])
        else:
            run_name = id
            cfg.work_dir = osp.join('./work_dir',
                                    osp.splitext(osp.basename(args.config))[0], run_name)

    ic(cfg.work_dir)
    mkdir_or_exist(osp.abspath(cfg.work_dir))
    shutil.copy(args.config, osp.join(cfg.work_dir, osp.basename(args.config)))
    if torch.cuda.is_available():
        print(colour_str('GPU available!', 'green'))
        print('Total number of available devices: ',
              colour_str(torch.cuda.device_count(), 'orange'))
        device = torch.device(f"cuda:{train_options['gpu_id']}")

    else:
        print(colour_str('GPU not available.', 'red'))
        device = torch.device('cpu')
    print('GPU setup completed!')

    if train_options['model_selection'] == 'unet':
        net = UNet(options=train_options).to(device)
    elif train_options['model_selection'] == 'swin':
        net = SwinTransformer(options=train_options).to(device)
    elif train_options['model_selection'] == 'Separate_decoder':
        net = Sep_feat_dif_stages(options=train_options).to(device)
    elif train_options['model_selection'] in ['UNet_regression', 'unet_regression']:
        from floe.models.legacy.unet import UNet_regression
        net = UNet_regression(options=train_options).to(device)
    elif train_options['model_selection'] in ['UNet_regression_all']:
        from floe.models.legacy.unet import UNet_regression_all
        net = UNet_regression_all(options=train_options).to(device)
    elif train_options['model_selection'] in ['UNet_sep_dec_regression', 'unet_sep_dec_regression']:
        from floe.models.legacy.unet import UNet_sep_dec_regression
        net = UNet_sep_dec_regression(options=train_options).to(device)
    elif train_options['model_selection'] in ['UNet_sep_dec_mse']:
        from floe.models.legacy.unet import UNet_sep_dec_mse
        net = UNet_sep_dec_mse(options=train_options).to(device)
    else:
        raise ValueError('Unknown model selected')

    _wandb_kwargs = dict(
        name=osp.splitext(osp.basename(args.config))[0] + '_inference',
        project=args.wandb_project,
        config=train_options,
        id=id,
        resume="allow",
    )
    if args.wandb_disabled:
        _wandb_kwargs['mode'] = 'disabled'
    wandb.init(**_wandb_kwargs)

    # 模型按完整 charts 建好（多任务 checkpoint 含辅头，strict load 需要它）；此后
    # train_options 只用于取场景列表 key。test() 内部会从 cfg 重新导出 options 并只跑 target chart。
    train_options = val_scoped_options(train_options)

    # 场景列表：优先 --scene-list-file，其次按 key 从配置取 JSON 路径
    if args.scene_list_file is not None:
        list_json_path = args.scene_list_file
    else:
        _default_list_key = {
            'apply': 'test_path',
            'test':  'test_path',
            'val':   'val_path',
        }
        list_key = args.scene_list_key or _default_list_key[args.mode]
        if list_key not in train_options:
            raise KeyError(f"配置中缺少场景列表 key '{list_key}'（mode={args.mode}）")
        list_json_path = train_options[list_key]

    # 配置里也可能直接给出场景列表
    if isinstance(list_json_path, (list, tuple)):
        scene_list = list(list_json_path)
    else:
        # 先尝试 path_to_env 前缀，再尝试原样路径
        _env = train_options.get('path_to_env', '') or ''
        _candidates = [osp.join(_env, list_json_path), list_json_path]
        for _p in _candidates:
            if osp.isfile(_p):
                with open(_p, 'r') as _f:
                    scene_list = json.loads(_f.read())
                print(f"Loaded {len(scene_list)} scene(s) from {_p}")
                break
        else:
            raise FileNotFoundError(
                f"无法定位场景列表 JSON，尝试过: {_candidates}")

    test(args.mode, net, checkpoint_path, device, cfg,
         scene_list, args.mode)

    wandb.finish()


if __name__ == '__main__':
    main()
