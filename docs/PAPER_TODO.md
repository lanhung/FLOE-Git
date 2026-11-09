# 论文执行计划（2026-11-09 起，以此为准）

目标：把单折、单 seed 的 +0.008 macro mIoU 变成可发表的结论。核心是评估，不是再加模块。

## 0. 审计发现（决定了下面的计划）

| # | 问题 | 处理 |
|---|---|---|
| 1 | 375/42 名单泄漏：42 个验证景中 14 个与训练景是同一 datatake 的相邻帧 | 论文只用冻结的分组五折（`datalists/cv5/`）；`tools.train` 不带 `--fold` 时打印警告 |
| 2 | 旧训练入口在模型构造后才建 DataLoader，增加的模块消耗全局 RNG ⇒ 各变体的数据流不同 | 已修：构造前后各重设种子＋训练 DataLoader 独立 generator。历史 run 不满足，需重跑 |
| 3 | (a)→(b) 同时引入门控与证据，无法归因 | 新增对照 (b′) `pgsg_gate_noevidence.py`、(c′) `pgsg_noevidence.py` |
| 4 | +0.0078 与 seed 波动（平滑后约 0.007）同量级 | 五折配对＋fold 0 多 seed |
| 5 | 稿件表述与实现不符：8 通道（含 mask）、极化通道为 z_HH−z_HV、z 偏移的 dB 换算因极化而异、仿射不变性只对特征场成立、证据经门控与 xattn 残差进入主路 | 稿件已修订（见 git 记录）；数字未改 |
| 6 | 辐射探针在 z 空间加同一偏移：0.30 z ≈ +2.50 dB(HH)/+2.83 dB(HV)，不是共模定标漂移 | 新增 dB 空间模式 `radiometric.py --db-shifts` |
| 7 | 逐景 mIoU/hard-ice 把景内不存在的类记 0 | `evaluate.py` 增加 mIoU* 列；`radiometric.py` 逐景改为 NaN |
| 8 | 证据替换的均值在被评分的验证景上估计；sub-all/gate 并非干净移除 | `evidence_ablation.py --mean-list`（用该折训练景）；论文只引 */both |
| 9 | 类别与地域完全混杂：big/vast 全部来自 19 景加拿大北极；big 仅 12 景、vast 仅 9 景 | 写入 Limitations；逐类结论以五折为准 |

## 1. 准备状态（2026-11-09 晚）

- [x] 服务器 `gpu.chzmark.com:2338`：4× RTX 5000 Ada，见 `docs/ENVIRONMENT.md`。仓库部署在 `/root/autodl-tmp/WannianLi/FLOE/FLOE-Git`。
- [x] 环境 `FLOE/envs/ice-binary`（`tools/system/setup_env.sh`；隔离了 root 的 `~/.local`，否则其他项目的包排在环境前面）。ImageNet 权重（mit_b2、resnet50）已缓存，`floe_env.sh` 默认离线。
- [x] CPU 检查全部通过：`smoke_pgsg`（含 [E] 无证据对照：t=0 输出与 Baseline 逐位相同）、`radiometric --self-test`（含 dB 模式）、`paired_cv --self-test`、`evidence_ablation --self-test`。
- [x] GPU 端到端 `smoke_pipeline`：Baseline / PGSG / PGSG 无证据 均 PASS（已到达的场景子集）。
- [x] 数据流一致性 `tools.checks.seed_stream`：修复后三种配置前 3 个 batch 哈希完全相同；`--legacy`（旧行为）三者各不相同，证实了历史消融的混杂。
- [x] 归一化统计量：NC 属性与手册一致（HH −13.9018/8.3349，HV −27.0105/9.4454 dB）。
- [x] fold 0 名单：由三份 fold-0 诊断日志恢复（83 验证景，334 训练景，无共享 datatake），见 `datalists/cv5/README.md`。
- [x] 已布置 `tmux floe-launch`：fold 0 全部 417 景到齐且可读后，按 20 分钟间隔在 GPU 0–3 启动 (a)(c)(c′)(b′)，seed 42。日志 `FLOE/logs/wait_and_launch_f0_s42.log`；取消：`tmux kill-session -t floe-launch`。
- [x] 压缩副本 `FLOE/data_q10`（2.4× 更小，解码更快，误差 ≤0.005 dB，标签/掩膜逐位相同；`DATASET_HANDBOOK.md` §11）。`tmux floe-compress` 随传输逐景转换并逐文件核对；`floe-launch` 等待的是压缩副本。原始文件待全部 run 完成后再决定是否删除（约 360 GB）。
- [ ] 数据传输：rsync 进行中（约 46 景/小时）。`python -m tools.data.check_transfer --data-dir $FLOE_DATA_DIR --open -1`。
- [ ] fold 1–4 名单：原始 `fold*_*.json`，或旧 CM（`Data/cm/cv5/*.npz`，随传输到达）经 `tools.data.folds_from_cm` 恢复；分组文件（场景→组）若有一并放入。

## 2. Phase 1：决定论文定位的实验（GPU）

先过冒烟：

```bash
python -m tools.checks.smoke_pgsg                       # CPU；含新增 [E] 无证据对照检查
python -m tools.analysis.radiometric --self-test        # CPU；含 dB 模式自检
python -m tools.analysis.paired_cv --self-test          # CPU
python -m tools.checks.smoke_pipeline configs/FLOE/pgsg.py
```

训练矩阵（每格一个 run，seed 42）：

| 配置 | 含义 | f0 | f1 | f2 | f3 | f4 |
|---|---|---|---|---|---|---|
| `sota_segformer_in1k_dlr_417.py` | (a) Baseline | ☐ | ☐ | ☐ | ☐ | ☐ |
| `pgsg.py` | (c) 完整 PGSG | ☐ | ☐ | ☐ | ☐ | ☐ |
| `pgsg_noevidence.py` | (c′) 无证据 | ☐ | ☐ | ☐ | ☐ | ☐ |
| `pgsg_gate_noevidence.py` | (b′) 门控无证据 | ☐ | ☐ | ☐ | ☐ | ☐ |
| `pgsg_evidence_gate.py` | (b) 证据＋门控 | ☐ | ☐ | ☐ | ☐ | ☐ |

先跑 fold 0 的 (a)(c)(c′)，尽早看到 (c) 与 (c′) 的差别；再补其余折。另在 fold 0 上对 (a)(c)(c′) 各加 seed 123、2024，估计 seed 方差。

```bash
# 每个 run：训练 + 逐景评估 + CM 导出，独立 tmux 会话；检查数据完整与 GPU 空闲
bash tools/system/launch_fold.sh 0 42 baseline:0 pgsg:1 pgsg_noev:2 gate_noev:3
python -m tools.analysis.paired_cv --a "$FLOE_CM_DIR/cv5/baseline_s42_f{K}.npz" \
    --b "$FLOE_CM_DIR/cv5/pgsg_s42_f{K}.npz" --labels baseline pgsg \
    --groups datalists/cv5/groups.json --json-out docs/results/paired_baseline_pgsg.json
```

需要的配对比较：(c) vs (a)、(c) vs (c′)、(c′) vs (a)、(b) vs (b′)。

判读与定位：

- (c) 在多数折上稳定优于 (c′)：保留“物理证据引导”的主张。
- (c) ≈ (c′)，两者都优于 (a)：贡献是尺度融合结构；改名、改叙事，证据衰减读数作为负结果保留。
- 都不能稳定优于 (a)：转为数据集与诊断论文（medium 跨景泛化、辐射依赖、类别-地域混杂、负结果）。

## 3. Phase 2：加强（Phase 1 有结论后）

- dB 空间辐射探针，(a)(c)(c′) 各一次，五折或至少 fold 0：
  `python -m tools.analysis.radiometric ... --zscore-stats <data>/zscore_stats.json --check-nc-attrs --db-shifts=-4,-2,-1,1,2,4`
- 证据替换用训练景均值重跑：`evidence_ablation.py --mean-list datalists/cv5/fold0_train.json`
- 对比模型：`sota_deeplabv3_in1k_dlr.py`（DeepLabV3+·R50·ImageNet，同协议），五折。
- 可选：重叠滑窗评估对照；按地域分别报告；逐景 mIoU* 分布图。

## 4. 稿件收尾（数字到位后）

- Table 3 用修复后的协议在 fold 0 重做；Table 4 填五折配对结果；Table 2 用 `cv5_report.md`。
- 摘要、结论、致谢、附录；作者要求：无破折号、不用 `\emph{}`。
- OA、macro-F1、weighted-F1 从原始 CM 汇总（`paired_cv.py` 已输出）。
- 稿件尚未报告两项已有结果，须写入（用修复后的 run 重测后）：推理期证据替换（历史 fold 0：sub-all/both macro −0.0009）支撑“证据不再驱动门控”一节；PGSG 与 Baseline 的辐射鲁棒性对比（历史 fold 0，−0.30 z 下 PGSG macro 0.449、Baseline 0.460，hard-ice 退化 PGSG 较小），如实写成“难类退化减小但整体不更稳”。
- 仍空着、但已有来源的 TODO：增强列表（`configs/FLOE/_sota_base.py`）、Table 2 折均衡（Supplement 01 §1.2）、参数/计算量/硬件（Supplement 02 与 `EXPERIMENT_RESULTS.md`）。
- 历史 fold-0 数字不得与五折汇总并列比较；375/42 数字不进论文。


## 5. 发表路线（2026-11-10 起执行）

判断：难点在跨景泛化、标签与分辨率，不在模块数量（历史已试 ordinal、supcon、Lovász、B5、大 patch、纹理通道、EMA、DANN、PBA，均在 seed 噪声量级）。三条线：

1. **AutoICE 公开基准**（进行中）：AI4Arctic ready-to-train（train v3 + test v2，CC BY 4.0）。同类别体系、同 80 m，有独立测试集与已发表对照（冠军 FLOE 加权 F1 约 70.7%，待从原文核对）。
   - 域内：AutoICE train → AutoICE test（20 景）：`configs/AI4Arctic/a4a_{baseline,pgsg}.py`。
   - 跨时期迁移：本档案（2022–2026）↔ AutoICE（2018–2021），双向。
   - 报告时注明：bergs 类忽略；支配阈值 0.65 vs 0.5；AutoICE 无 Refined Lee 滤波、无 GLCM。
2. **分辨率极限**（进行中）：80 m vs 160 m（均无 GLCM，`res80_noglcm`/`res160_noglcm`，fold 0）。若 160 m 使 small/medium 明显下降 ⇒ 80 m 网格是瓶颈 ⇒ 用 AutoICE raw（40 m，261 GB）或本档案的 40 m 重处理做 40 m 实验。
3. **混合标签学习**（待数据）：用冰图部分浓度（Ca/Fa、Cb/Fb、Cc/Fc）做软标签 + 多边形级评估，直接针对 medium↔small 混淆。AutoICE raw 有多边形编码；本档案需要原作者的数据构建代码（不在本仓库）。

GPU 队列：`tools/system/queues/main_2026-11.txt`，`tmux floe-queue`；`bash tools/system/run_queue.sh <queue> --check` 查看状态。
