# FLOE-Git

基于 Sentinel-1 EW SAR 的海冰浮冰尺寸（floe size）逐像素语义分割。标签来自业务冰图多边形的
主导浮冰尺寸，6 类（water / cake / small / medium / big / vast），ignore index 255，80 m 网格。
包含 SegFormer-B2 基线、PGSG（physics-guided scale gating）及其对照、AutoICE 公开基准的适配、
论文诊断工具和论文源码。数据构建（Sentinel-1 预处理与冰图配准）代码不在本仓库。

## 目录

```text
floe/            模型、数据读取、损失与指标
  models/sota/   SegFormer（含 PGSG 模块）及对比模型
  models/legacy/ 早期 U-Net / Swin（infer.py 与旧配置仍引用）
configs/         FLOE/：本档案；AI4Arctic/：AutoICE 基准；_base_/：公共默认值
datalists/       cv5/：分组五折名单；ai4arctic/：AutoICE 名单与官方 MD5 清单
tools/           train / evaluate / infer 入口，以及 data、analysis、figures、checks、system
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
