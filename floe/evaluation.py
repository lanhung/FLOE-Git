"""Scene-by-scene inference with a trained checkpoint (adapted from the AutoICE test notebook).

test() runs mode 'val', 'test' or 'apply' (no labels), saves one figure per scene under
<work_dir>/inference_<mode>/, and for labelled modes logs metrics to stdout and W&B.
"""

# -- Built-in modules -- #
import os
import os.path as osp

# -- Third-part modules -- #
import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr
from tqdm import tqdm
from mmengine.utils import mkdir_or_exist
import wandb
# --Proprietary modules -- #
from floe.functions import chart_cbar, compute_metrics, water_edge_metric, class_decider
from floe.loaders import AI4ArcticChallengeTestDataset, get_variable_options, val_scoped_options
from floe.functions import slide_inference


def test(mode: str, net: torch.nn.modules, checkpoint: str, device: str, cfg, test_list, test_name, test_list_reference=None):
    """Load `checkpoint` into `net` and run inference over `test_list`.

    Args:
        mode: 'val', 'test' or 'apply' (unlabelled scenes; metrics are skipped).
        net: model built with the full training charts; weights are loaded strictly.
        checkpoint: path to a checkpoint written by save_best_model.
        device: device to run inference on.
        cfg: mmengine Config of the run.
        test_name: prefix for W&B summary keys and the confusion-matrix folder.
    """

    if mode not in ["val", "test", "apply"]:
        raise ValueError("mode must be one of 'val', 'test', 'apply'")

    # 'apply'：无标签推理。dataset 用 mode='test_no_gt'，不读取标签；下面所有依赖标签的
    # 步骤（指标、混淆矩阵、GT 可视化、wandb summary）都跳过。
    _apply_mode = (mode == 'apply')

    train_options = cfg.train_options
    train_options = get_variable_options(train_options)
    # 推理/评估只跑 target chart（aux_charts 只用于训练，见 loaders.val_scoped_options）。
    # net 按完整 charts 构建并 strict load，这里收窄的只是数据读取、前向和指标。
    train_options = val_scoped_options(train_options)
    weights = torch.load(checkpoint, weights_only=False)['model_state_dict']
    # torch.compile 训练的 checkpoint 带 "_orig_mod." 前缀。
    if any(k.startswith('_orig_mod.') for k in weights.keys()):
        weights = {k.replace('_orig_mod.', '', 1): v for k, v in weights.items()}

    net.load_state_dict(weights)
    print('Model successfully loaded.')
    experiment_name = osp.splitext(osp.basename(cfg.work_dir))[0]
    artifact = wandb.Artifact(experiment_name, 'dataset')
    table = wandb.Table(columns=['ID', 'Image'])

    # - Stores the output and the reference pixels to calculate the scores after inference on all the scenes.
    output_class = {chart: torch.Tensor().to(device) for chart in train_options['charts']}
    outputs_flat = {chart: torch.Tensor().to(device) for chart in train_options['charts']}
    inf_ys_flat = {chart: torch.Tensor().to(device) for chart in train_options['charts']}
    # Outputs masked by the train fill value.
    outputs_tfv_mask = {chart: torch.Tensor().to(device) for chart in train_options['charts']}

    if mode == 'test':
        train_options['test_list'] = test_list
        dataset = AI4ArcticChallengeTestDataset(
            options=train_options, files=train_options['test_list'], mode='test')
        asid_loader = torch.utils.data.DataLoader(
            dataset, batch_size=None, num_workers=train_options['num_workers_val'], shuffle=False)
        print('Setup ready')

    elif mode == 'val':
        train_options['test_list'] = test_list
        dataset = AI4ArcticChallengeTestDataset(
            options=train_options, files=train_options['test_list'], mode='train')
        asid_loader = torch.utils.data.DataLoader(
            dataset, batch_size=None, num_workers=train_options['num_workers_val'], shuffle=False)
        print('Setup ready')

    elif mode == 'apply':
        train_options['test_list'] = test_list
        dataset = AI4ArcticChallengeTestDataset(
            options=train_options, files=train_options['test_list'], mode='test_no_gt')
        asid_loader = torch.utils.data.DataLoader(
            dataset, batch_size=None, num_workers=train_options['num_workers_val'], shuffle=False)
        print('Setup ready (apply / no-GT)')

    if mode == 'val':
        inference_name = 'inference_val'
    elif mode == 'test':
        inference_name = 'inference_test'
    elif mode == 'apply':
        inference_name = 'inference_apply'

    # 场景根目录，用于回读 coastline_land_mask 做陆地可视化。
    if mode in ('test', 'apply'):
        _scene_root = train_options.get('path_to_test_data', None)
    else:
        _scene_root = train_options.get('path_to_train_data', None)

    target_chart = train_options.get('target_chart', 'FLOE')

    # 类别用 jet，陆地（< vmin）浅灰，NaN 透明。
    _land_cmap = plt.get_cmap('jet').copy()
    _land_cmap.set_under('lightgray')
    _land_cmap.set_bad(color='white', alpha=0.0)

    os.makedirs(osp.join(cfg.work_dir, inference_name), exist_ok=True)
    net.eval()
    for inf_x, inf_y, cfv_masks, tfv_mask, scene_name, original_size in tqdm(iterable=asid_loader,
                                                               total=len(train_options['test_list']), colour='green', position=0):
        scene_name = osp.splitext(scene_name)[0]  # Remove the .nc extension.
        torch.cuda.empty_cache()

        coastline_mask_full = None
        if _scene_root is not None:
            _nc_path = osp.join(_scene_root, scene_name + '.nc')
            try:
                with xr.open_dataset(_nc_path, engine='h5netcdf') as _ds_land:
                    if 'coastline_land_mask' in _ds_land.variables:
                        coastline_mask_full = _ds_land['coastline_land_mask'].values.astype(bool)
            except Exception as _e:
                # 场景文件读不到时不画陆地，推理照常进行。
                coastline_mask_full = None

        inf_x = inf_x.to(device, non_blocking=True)
        with torch.no_grad(), torch.cuda.amp.autocast():
            _needs_slide = (train_options['model_selection'] == 'swin' or
                            inf_x.shape[2] > train_options['patch_size'] or
                            inf_x.shape[3] > train_options['patch_size'])
            if _needs_slide:
                output = slide_inference(inf_x, net, train_options, 'test')
            else:
                output = net(inf_x)

            # Upsample the masks to the original scene size.
            tfv_mask = torch.nn.functional.interpolate(tfv_mask.type(torch.uint8).unsqueeze(0).unsqueeze(0), size=original_size, mode='nearest').squeeze().squeeze().to(torch.bool)
            if cfv_masks is not None:
                for chart in train_options['charts']:
                    masks_int = cfv_masks[chart].to(torch.uint8)
                    masks_int = torch.nn.functional.interpolate(masks_int.unsqueeze(
                        0).unsqueeze(0), size=original_size, mode='nearest').squeeze().squeeze()
                    cfv_masks[chart] = torch.gt(masks_int, 0)

            if train_options['down_sample_scale'] != 1:
                for chart in train_options['charts']:
                    # Regression head: (N, H, W, 1).
                    if output[chart].size(3) == 1:
                        output[chart] = output[chart].permute(0, 3, 1, 2)
                        output[chart] = torch.nn.functional.interpolate(
                            output[chart], size=original_size, mode='nearest')
                        output[chart] = output[chart].permute(0, 2, 3, 1)
                    else:
                        # 先 argmax 再上采样，显存只需 1/n_classes。
                        output[chart] = class_decider(output[chart], train_options, chart).unsqueeze(0).unsqueeze(0).float()
                        output[chart] = torch.nn.functional.interpolate(
                            output[chart], size=original_size, mode='nearest').squeeze().to(torch.int64)

                    if inf_y is not None:
                        inf_y[chart] = torch.nn.functional.interpolate(inf_y[chart].unsqueeze(dim=0).unsqueeze(dim=0),
                                                                       size=original_size, mode='nearest').squeeze()

        for chart in train_options['charts']:
            # With down_sample_scale != 1 the class map was already computed above.
            if train_options['down_sample_scale'] != 1 and output[chart].dim() == 2:
                output_class[chart] = output[chart].detach()
            else:
                output_class[chart] = class_decider(output[chart], train_options, chart).detach()
            if not _apply_mode:
                outputs_flat[chart] = torch.cat(
                            (outputs_flat[chart], output_class[chart][~cfv_masks[chart]]))
                outputs_tfv_mask[chart] = torch.cat(
                            (outputs_tfv_mask[chart], output_class[chart][~tfv_mask].to(device)))
                inf_ys_flat[chart] = torch.cat(
                            (inf_ys_flat[chart], inf_y[chart][~cfv_masks[chart]].to(device, non_blocking=True)))

        for chart in train_options['charts']:
            if inf_y is not None:
                inf_y[chart] = inf_y[chart].cpu().numpy()
            output_class[chart] = output_class[chart].squeeze().cpu().numpy()

        # Per-scene figure, 2×3:
        #   Row 0: HH | HV | GLCM Contrast
        #   Row 1: GLCM Homogeneity | target-chart prediction | ground truth (apply: HH + land)
        fig, axs2d = plt.subplots(nrows=2, ncols=3, figsize=(18, 11))
        axs = axs2d.flat

        img_data = torch.squeeze(inf_x, dim=0).cpu().numpy()

        def pclip(arr, lo=2, hi=98):
            """Percentile-based vmin/vmax to suppress outliers."""
            finite = arr[np.isfinite(arr)]
            if finite.size == 0:
                return arr.min(), arr.max()
            return np.percentile(finite, lo), np.percentile(finite, hi)

        # --- Row 0, Col 0: HH ---
        ax = axs[0]
        hh = img_data[0]
        vmin, vmax = pclip(hh)
        ax.imshow(hh, cmap='gray', vmin=vmin, vmax=vmax, interpolation='nearest')
        ax.set_title('SAR HH', fontsize=12, fontweight='bold')
        ax.set_xticks([]); ax.set_yticks([])

        # --- Row 0, Col 1: HV ---
        ax = axs[1]
        hv = img_data[1]
        vmin, vmax = pclip(hv)
        ax.imshow(hv, cmap='gray', vmin=vmin, vmax=vmax, interpolation='nearest')
        ax.set_title('SAR HV', fontsize=12, fontweight='bold')
        ax.set_xticks([]); ax.set_yticks([])

        # --- Row 0, Col 2: GLCM Contrast ---
        # Channels are looked up by name: configs without GLCM bands (e.g. the
        # resolution / AutoICE runs) would otherwise index past the end.
        _tv = list(train_options['train_variables'])

        def _panel(ax, var, title):
            if var not in _tv:
                ax.set_title(f'{title} (not an input)', fontsize=12)
                ax.set_xticks([]); ax.set_yticks([])
                return
            arr = img_data[_tv.index(var)]
            vmin, vmax = pclip(arr)
            im = ax.imshow(arr, cmap='viridis', vmin=vmin, vmax=vmax, interpolation='nearest')
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xticks([]); ax.set_yticks([])
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        _panel(axs[2], 'glcm_sigma0_hh_contrast', 'GLCM Contrast (HH)')

        # --- Row 1, Col 0: GLCM Homogeneity ---
        _panel(axs[3], 'glcm_sigma0_hh_homogeneity', 'GLCM Homogeneity (HH)')

        # --- Row 1, Col 1: Target-chart Prediction ---
        ax = axs[4]
        pred_map = output_class[target_chart].astype(float)
        # apply 模式没有 cfv_masks，用 tfv_mask 标无效区。
        _invalid_pred = cfv_masks[target_chart].numpy() if (cfv_masks is not None) else tfv_mask.cpu().numpy()
        pred_map[_invalid_pred] = np.nan
        # 陆地置 -1（< vmin），由 set_under 画成浅灰。
        if coastline_mask_full is not None:
            pred_map[coastline_mask_full] = -1.0
        ax.imshow(pred_map, vmin=0, vmax=train_options['n_classes'][target_chart] - 1,
                  cmap=_land_cmap, interpolation='nearest')
        ax.set_title(f'{target_chart}: Prediction', fontsize=12, fontweight='bold')
        ax.set_xticks([]); ax.set_yticks([])
        chart_cbar(ax=ax, n_classes=train_options['n_classes'][target_chart], chart=target_chart, cmap='jet')

        # --- Row 1, Col 2: Ground truth, or HH + land overlay in apply mode (checks mask alignment) ---
        ax = axs[5]
        if _apply_mode or inf_y is None:
            hh_gray = img_data[0].copy()
            vmin_hh, vmax_hh = pclip(hh_gray)
            ax.imshow(hh_gray, cmap='gray', vmin=vmin_hh, vmax=vmax_hh, interpolation='nearest')
            if coastline_mask_full is not None:
                overlay = np.zeros((*coastline_mask_full.shape, 4), dtype=np.float32)
                overlay[coastline_mask_full] = [1.0, 0.3, 0.3, 0.45]  # translucent red
                ax.imshow(overlay, interpolation='nearest')
            ax.set_title('HH + Land Overlay (apply mode)', fontsize=12, fontweight='bold')
            ax.set_xticks([]); ax.set_yticks([])
        else:
            gt_map = inf_y[target_chart].astype(float)
            gt_map[cfv_masks[target_chart].numpy()] = np.nan
            if coastline_mask_full is not None:
                gt_map[coastline_mask_full] = -1.0
            ax.imshow(gt_map, vmin=0, vmax=train_options['n_classes'][target_chart] - 1,
                      cmap=_land_cmap, interpolation='nearest')
            ax.set_title(f'{target_chart}: Ground Truth', fontsize=12, fontweight='bold')
            ax.set_xticks([]); ax.set_yticks([])
            chart_cbar(ax=ax, n_classes=train_options['n_classes'][target_chart], chart=target_chart, cmap='jet')

        fig.suptitle(f'Scene: {scene_name}', fontsize=13, y=1.01)
        plt.tight_layout()
        fig.savefig(f"{osp.join(cfg.work_dir,inference_name,scene_name)}.png",
                    format='png', dpi=128, bbox_inches="tight")
        plt.close('all')
        table.add_data(scene_name, wandb.Image(f"{osp.join(cfg.work_dir,inference_name,scene_name)}.png"))

    if _apply_mode:
        artifact.add(table, experiment_name + '_apply')
        wandb.log_artifact(artifact)
        return

    combined_score, scores = compute_metrics(true=inf_ys_flat, pred=outputs_flat, charts=train_options['charts'],
                                             metrics=train_options['chart_metric'], num_classes=train_options['n_classes'])

    water_edge_accuarcy = water_edge_metric(outputs_tfv_mask, train_options)
    if train_options['compute_classwise_f1score']:
        from floe.functions import compute_classwise_f1score
        classwise_scores = compute_classwise_f1score(true=inf_ys_flat, pred=outputs_flat,
                                                     charts=train_options['charts'], num_classes=train_options['n_classes'])

    if train_options['plot_confusion_matrix']:
        from torchmetrics.functional.classification import multiclass_confusion_matrix
        import seaborn as sns
        from floe.utils import GROUP_NAMES

        for chart in train_options['charts']:
            cm = multiclass_confusion_matrix(
                preds=outputs_flat[chart], target=inf_ys_flat[chart], num_classes=train_options['n_classes'][chart])
            cm = cm.cpu().numpy()
            cm_percent = np.round(cm / cm.sum(axis=1)[:, np.newaxis] * 100, 2)
            plt.figure(figsize=(10, 8))
            ax = sns.heatmap(cm_percent, annot=True, cmap='Blues')
            # cm 是 n_classes×n_classes（255 已剔除），刻度标签数与之一致。
            class_names = list(GROUP_NAMES[chart].values())
            tick_marks = np.arange(len(class_names)) + 0.5
            plt.xticks(tick_marks, class_names, rotation=45)
            if chart in ['FLOE', 'SOD']:
                plt.yticks(tick_marks, class_names, rotation=45)
            else:
                plt.yticks(tick_marks, class_names)

            plt.xlabel('Predicted Labels')
            plt.ylabel('Actual Labels')
            plt.title('Confusion Matrix')
            cbar = ax.collections[0].colorbar
            cbar.set_ticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
            mkdir_or_exist(f"{osp.join(cfg.work_dir)}/{test_name}")
            plt.savefig(f"{osp.join(cfg.work_dir)}/{test_name}/{chart}_confusion_matrix.png",
                        format='png', dpi=128, bbox_inches="tight")

    wandb.run.summary[f"{test_name}/Best Combined Score"] = combined_score
    print(f"{test_name}/Best Combined Score = {combined_score}")
    for chart in train_options['charts']:
        wandb.run.summary[f"{test_name}/{chart} {train_options['chart_metric'][chart]['func'].__name__}"] = scores[chart]
        print(
            f"{test_name}/{chart} {train_options['chart_metric'][chart]['func'].__name__} = {scores[chart]}")
        if train_options['compute_classwise_f1score']:
            wandb.run.summary[f"{test_name}/{chart}: classwise score:"] = classwise_scores[chart]
            print(
                f"{test_name}/{chart}: classwise score: = {classwise_scores[chart]}")

    wandb.run.summary[f"{test_name}/Water Consistency Accuarcy"] = water_edge_accuarcy
    print(
        f"{test_name}/Water Consistency Accuarcy = {water_edge_accuarcy}")

    if mode == 'test':
        artifact.add(table, experiment_name+'_test')
    elif mode == 'val':
        artifact.add(table, experiment_name+'_val')

    wandb.log_artifact(artifact)
