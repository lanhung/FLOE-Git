# FLOE-Git 工作约定

- 范围：基线、PGSG 及对照、AutoICE 基准、五折评估与论文。数据构建源码不在仓库内。
- 论文实验只用分组五折（`datalists/cv5/`，`tools.train --fold K`）。375/42 名单（`train_417`/`val_417`）有相邻帧泄漏，只作冒烟。
- 本地只编辑与静态检查；训练、评估与 Python 检查在服务器的 `ice-binary` 环境运行（`source tools/system/floe_env.sh`）。服务器多人共用，只在 `/root/autodl-tmp/WannianLi/FLOE/` 内读写。
- 数据路径来自环境变量（`FLOE_DATA_DIR`、`FLOE_TILE_INDEX`、`FLOE_A4A_DIR` 等），由 `floe_env.sh` 设置。
- 保持 6 类、ignore index 255；输入 8 通道（HH、HV、入射角、3 个 GLCM、有效掩膜、z_HH−z_HV）。改物理解释前先核对归一化统计量。
- 训练与验证分辨率一致；训练走 chunk 对齐的 tile 采样，推理走滑窗。按需读只在满足网格对齐前提的验证路径上启用。
- 跑之前看 GPU、cgroup CPU/内存与磁盘；同一张卡训练期间不另起 GPU 任务。长任务放 tmux，日志写持久盘，不用 `tee` 的退出码判断成败。
- 训练入口在模型构造前后各重设一次种子，训练 DataLoader 用独立 generator；改动训练入口后先跑 `tools.checks.seed_stream`。
- checkpoint 不上传 W&B，只记标量和小文件。
- 主要比较指标：macro mIoU、逐类 IoU、hard-ice（small/medium IoU 均值）；macro-F1、weighted-F1、OA 分开报告。
- 历史 fold-0 数字不得与修复后的 run 并列比较；不改写历史日志和论文已有结果；修路径与改模型科学定义分开提交。
- 原始数据（`Data/FLOE_datanc`）未经用户同意不删除。
