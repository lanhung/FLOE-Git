# FLOE-Git

基于 Sentinel-1 EW SAR 的海冰浮冰尺寸（floe size）逐像素语义分割。标签来自业务冰图多边形的
主导浮冰尺寸，6 类（water / cake / small / medium / big / vast），ignore index 255，80 m 网格。
包含 SegFormer-B2 基线、PGSG（physics-guided scale gating）及其对照、AutoICE 公开基准的适配、
论文诊断工具和论文源码，以及从 Sentinel-1 原始产品与 DMI 冰图重建数据集的代码（`tools/build`）。

## 目录

```text
floe/            模型、数据读取、损失与指标
  models/sota/   SegFormer（含 PGSG 模块）及对比模型
  models/legacy/ 早期 U-Net / Swin（infer.py 与旧配置仍引用）
configs/         FLOE/：本档案；AI4Arctic/：AutoICE 基准；_base_/：公共默认值
datalists/       cv5/：分组五折名单；ai4arctic/：AutoICE 名单与官方 MD5 清单
tools/           train / evaluate / infer 入口，以及 build、data、analysis、figures、checks、system
preprocessing/   原作者的 SNAP 处理与标注脚本（旧档案的来源，保留备查，不再使用）
docs/            数据手册、计划、论文数字来源、诊断日志
paper/           论文 LaTeX 与插图
```

所有工具从仓库根目录以 `python -m tools.…` 运行。

## 配置

| 配置 | 内容 |
|---|---|
| `configs/FLOE/sota_segformer_in1k_dlr_417.py` | (a) 基线：SegFormer-B2，ImageNet 初始化，编码器 0.1× 学习率 |
| `configs/FLOE/pgsg_evidence_gate.py` | (b) 证据 + 竞争门控 |
| `configs/FLOE/pgsg.py` | (c) 完整 PGSG（另加尺度轴 Transformer） |
| `configs/FLOE/pgsg_gate_noevidence.py`、`pgsg_noevidence.py` | (b′)(c′) 去掉证据的结构对照 |
| `configs/FLOE/res80_noglcm.py`、`res160_noglcm.py` | 分辨率实验：输入 80 m vs 160 m，标签与评估均在 80 m |
| `configs/FLOE/sota_deeplabv3_in1k_dlr.py` | DeepLabV3+ R50 对比 |
| `configs/AI4Arctic/a4a_baseline.py`、`a4a_pgsg.py` | AutoICE（AI4Arctic ready-to-train v3），仅 SAR |
| `configs/Pooled/pooled_noglcm.py` | 本档案 fold-K 训练景 + 全部 AutoICE 训练景（对照为 `res80_noglcm`） |
| `configs/FLOE/comp.py`、`comp_noord.py`、`comp_allce.py` | 冰图组成监督（`floe/composition.py`）及其消融 |

论文实验一律逐折训练：`python -m tools.train <cfg> --fold K`，名单取 `datalists/cv5/fold{K}_{train,val}.json`。
配置里默认的 `train_417`/`val_417` 名单只用于冒烟：42 个验证景中有 14 个与训练景属于同一 datatake 的相邻帧。

## 常用命令

```bash
source tools/system/floe_env.sh                 # 数据路径、HF 镜像、离线 W&B，并激活环境
python -m tools.checks.smoke_pgsg               # CPU，无需数据
python -m tools.checks.smoke_pipeline configs/FLOE/pgsg.py
python -m tools.train configs/FLOE/pgsg.py --fold 0 --seed 42 --work-dir work_dir/pgsg_f0
python -m tools.evaluate --config configs/FLOE/pgsg.py --checkpoint work_dir/pgsg_f0/best_model_*.pth \
    --val-list datalists/cv5/fold0_val.json --dump-cm $FLOE_CM_DIR/cv5/pgsg_s42_f0.npz
python -m tools.analysis.paired_cv --a "$FLOE_CM_DIR/cv5/baseline_s42_f{K}.npz" \
    --b "$FLOE_CM_DIR/cv5/pgsg_s42_f{K}.npz" --labels baseline pgsg
```

服务器上的长任务走 GPU 队列：`tools/system/run_queue.sh <queue>`（`--check` 只看状态），
单个 run 用 `launch_fold.sh`（五折）或 `launch_run.sh`（AutoICE 等非折 run），都在 tmux 中运行并把日志写到持久盘。

## 数据工具

| 工具 | 用途 |
|---|---|
| `tools/data/scan_tiles.py` | 建 tile 索引（训练采样用） |
| `tools/data/recompress.py` | 压缩副本：连续变量量化到 2⁻¹⁰ z 网格 + Blosc-zstd，标签逐位不变，逐文件核对 |
| `tools/data/check_transfer.py` | 检查传输完整性，可逐个打开 NC |
| `tools/data/fetch_ai4arctic.sh`、`convert_ai4arctic.py`、`prepare_ai4arctic.sh` | 下载 AutoICE 并按官方 MD5 核对，统一到本档案的辐射标准化，建名单与 tile 索引 |
| `tools/data/folds_from_cm.py` | 由旧五折逐景 CM 恢复折名单 |
| `tools/data/pool_archives.py` | 两个档案合并训练：符号链接目录 + 合并 tile 索引 |

## 文档

- `docs/PAPER_TODO.md`：当前计划与进度（§6 为组成监督方法与实验设计）。
- `docs/PAPER_NUMBERS.md`：论文中每个数字的来源。
- `docs/DATASET_HANDBOOK.md`、`docs/dataset_distribution.md`、`docs/tile_sampling.md`：数据格式、分布与采样设计。
- `docs/EXPERIMENT_RESULTS.md`：历史 fold-0 结果及其适用范围。
- `docs/ENVIRONMENT.md`：服务器、目录与依赖。
- `docs/logs/`：论文诊断实验的原始日志（只读）。

## 许可

MIT，见 [LICENSE](LICENSE)。`floe/functions.py` 等文件源自滑铁卢大学的 AutoICE 工具包，保留其作者信息。
外部数据（Sentinel-1、冰图、AI4Arctic，CC BY 4.0）与预训练权重按各自许可使用。

## 数据来源

旧档案（417 景）由 `preprocessing/`（SNAP 流程）生成；其问题与替代方案见 `docs/DATA_PIPELINE_AUDIT.md`。新档案由 `tools/build` 生成。

## FLOE v1.0: dataset, benchmark and papers

This branch (`paper-revision`) holds the code behind the FLOE v1.0 dataset (3224 Sentinel-1 EW scenes with DMI
ice-chart floe-size labels around Greenland, 2022-2026) and the two papers that use it.

- **Code archive:** https://doi.org/10.5281/zenodo.24295581 (all versions; MIT licence).
- **Dataset:** deposited in Science Data Bank under CC BY 4.0 (DOI on publication of the dataset paper). Ice charts:
  The Greenland Ice Service, Danish Meteorological Institute. Contains modified Copernicus Sentinel data 2022-2026.
- **Build and release:** `tools/build/` (pairing, labels from the egg codes, `split.py` for the benchmark split in
  `datalists/v2_bench/`, `release.py` for the public netCDF files with checksums).
- **Dataset paper:** `tools/analysis/archive_stats.py`, `leak_effect.py`, `baseline_table.py` (Table 7),
  `tools/figures/data_paper.py`.
- **Method paper:** `tools/analysis/screen_rule.py` (choice of the benchmark model), `paper2_compare.py` (the
  comparison fixed before the results), `paper2_breakdown.py`, `paper2_figdata.py`, `paper2_tables.py`,
  `tools/figures/method_paper.py`.
- **Runs:** `configs/V2/` (baseline, multitask, comp_aux, multitask_comp), `tools/system/launch_fold.sh`,
  `tools/system/eval_bench.sh`, run lists in `tools/system/queues/`.

### Rebuilding the dataset

These commands rebuild FLOE v1.0 from the DMI chart archive and the Alaska Satellite Facility (an Earthdata
account is needed for the downloads), as described in Sect. 4 and Appendix C of the dataset paper. Every step can
be restarted where it stopped. Each product (about 0.26 GB) is read without unpacking and deleted after its scene
is written, so little scratch space is needed; with 8 processes the build handled about 130 frames per hour.

```bash
# DMI charts of 2022-2026 and their index
python -m tools.build.charts fetch --years 2022 2023 2024 2025 2026 --out raw/dmi
python -m tools.build.charts index --dir raw/dmi
# Sentinel-1 EW frames for every chart, one file per year
for y in 2022 2023 2024 2025 2026; do
  python -m tools.build.pair --charts raw/dmi --years $y --out raw/pairs_$y.json
done
# download, calibrate, label and write the scenes
python -m tools.build.run_archive --pairs raw/pairs_*.json --charts raw/dmi \
    --out data --chart-out charts --tmp raw/s1 --workers 8
# benchmark split: grouped folds, held-out year 2026 with a 14-day buffer, charts within 5 min of the image
python -m tools.build.split --log data/archive_log.jsonl --data-dir data --out datalists/v2_bench \
    --folds 5 --max-gap-minutes 5 --test-year 2026 --buffer-days 14
# polygon table, label ceiling and chart agreement (Sect. 6 of the dataset paper)
python -m tools.build.polygon_table --data-dir data --out polygons
python -m tools.build.label_purity --table polygons.npz --out purity
python -m tools.build.chart_agreement --charts raw/dmi --out agreement
python -m tools.analysis.agreement_bootstrap --prefix agreement
# public netCDF files with checksums, scene table and split lists
python -m tools.build.release --data-dir data --table polygons.npz --log data/archive_log.jsonl \
    --split datalists/v2_bench --out release_v1
```
