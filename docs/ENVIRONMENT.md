# 环境

## 服务器（2026-11-09 核实）

- 实例：`ssh -p 2338 root@gpu.chzmark.com`（AutoDL 容器，多人共用：只在 `/root/autodl-tmp/WannianLi/FLOE/` 内读写）。
- 资源：4× NVIDIA RTX 5000 Ada（32 GB），驱动 570.124.06 / CUDA 12.8；cgroup 64 核、约 124 GB 内存；`/root/autodl-tmp` 2.8 TB（约 1.3 TB 可用）。
- 目录：
  - `FLOE/FLOE-Git/`：本仓库（rsync 部署）
  - `FLOE/Data/FLOE_datanc/`：417 景原始 NC（传输中，`FLOE_DATA_ORIG`）；`FLOE/Data/floe_tile_index_417.npz`：tile 索引
  - `FLOE/data_q10/`：压缩副本（训练默认，`FLOE_DATA_DIR`），见 `DATASET_HANDBOOK.md` §11
  - `FLOE/envs/ice-binary/`：conda 环境（`tools/system/setup_env.sh` 构建，`pip-freeze.txt` 为锁定清单）
  - `FLOE/ai4arctic/`：AutoICE 基准（`dl/` 下载、`rtt/` 解包、`floe_format/` 转换后、`floe_tile_index_a4a.npz`）
  - `FLOE/logs/`、`FLOE/cm/`、`FLOE/figs/`：日志、逐景 CM、图
- Figshare/DTU 下载有反爬验证，服务器与本机均无法脚本下载；AutoICE 取自 Hugging Face 上 TorchGeo 的转存（经 hf-mirror），逐文件与官方 MD5 核对。
- 网络：pypi 镜像、download.pytorch.org、github、hf-mirror.com 可达；huggingface.co 不可达（`HF_ENDPOINT` 指向镜像）；无 W&B 登录（`WANDB_MODE=offline`）。

每次使用先 `source tools/system/floe_env.sh`（设置数据路径、镜像与离线 W&B，并激活环境）。

## 归一化统计量（已核实）

417 景 NC 的变量属性 `zscore_mean`/`zscore_std`：HH −13.9018 / 8.3349 dB，HV −27.0105 / 9.4454 dB，与 `DATASET_HANDBOOK.md` 附录一致（已核对前 30 个到达的文件；传输完成后用 `tools.data.check_transfer --open -1` 全量核对）。数据目录中没有 `zscore_stats.json`；`radiometric.py --db-shifts` 直接读 NC 属性并逐景核对。

## 依赖

主要库：PyTorch、torchvision、torchmetrics、segmentation-models-pytorch（0.5.0：`segformer.py` 的分段前向按该版本的 MiT 调用约定编写）、timm、mmcv 1.x（只用 `mmcv.Config`/`mkdir_or_exist`，无 CUDA 算子）、mmengine、numpy、scipy、h5py、xarray、h5netcdf、matplotlib、tqdm、wandb、icecream；可选 seaborn、fvcore、thop。具体版本见 `tools/system/setup_env.sh` 与环境内的 `pip-freeze.txt`。

旧服务器（RTX 5090）上的原 `ice-binary` 环境未迁移；新环境下的 run 与历史 run 不逐位可比（另见 RNG 修复）。
