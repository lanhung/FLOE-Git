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

## 1. 准备状态（2026-11-10）

- 服务器、环境、压缩副本、AutoICE 均已就绪，见 `docs/ENVIRONMENT.md`。417 景原始 NC 传输完成；压缩副本 `data_q10` 随 `tmux floe-compress` 生成（19 个较新的景另带 SIC/SOD 层，逐位复制）。
- 自检全部通过：`smoke_pgsg`、`smoke_degrade`、`seed_stream`（三种配置数据流逐位相同）、`paired_cv`/`radiometric`/`evidence_ablation --self-test`，GPU `smoke_pipeline`（PGSG、AutoICE、res160）。
- 代码清理前后，五个配置的模型初始化、训练 batch、首批 logits 与 loss、验证样本的哈希逐位相同。
- GPU 队列 `tools/system/queues/main_2026-11.txt`（`tmux floe-queue`）；AutoICE baseline / PGSG 已在 GPU 0/1 训练。
- [ ] fold 1–4 名单（原始 json，或由旧 CM 经 `tools.data.folds_from_cm` 恢复）。
- [ ] 数据构建代码 / 冰图属性（egg code）：组成监督的前提，见 §6。

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


## 6. 方法：冰图组成监督（2026-11-10 起）

文献综述（`reports/` 下的调研报告，未入库）的结论：FLOE 的瓶颈在标签语义与数据覆盖，不在网络结构。
AutoICE 冠军的 FLOE 只有 70.70%（最好 79.12%），top-5 对 cake/small/medium 的类准确率仅 14–21%；
组成感知的监督已在 SOD/SIC 上把少数类 F1 提高一倍以上，但从未用于 FLOE，也没有与尺寸序结合。

### 6.1 方法

- 每个冰图多边形由 egg code 得到 6 类组成 q（water = 1 − CT；FA/FB/FC × CA/CB/CC；无法映射的形态记为未知冰量 u）。
- 损失（`floe/composition.py`）：多边形均值预测与 q 的交叉熵（LLP）+ 冰类累积分布的序损失（cake < small < medium < big < vast）；
  纯多边形（某类 ≥ 0.9）保留逐像素 CE；未知冰量按当前预测分配到冰类。
- 实现：多边形 id 作为额外标签通道随翻转、旋转、cutmix 一起变换（`smoke_composition` [D] 验证对齐）；
  关闭时数据流与模型初始化逐位不变。
- 数据：每景一个 `<scene>.chart.h5`（`polygon_id` 栅格 + `composition`[P,6] + `unknown`[P]），由数据构建代码生成。

### 6.2 实验

| 对比 | 回答的问题 |
|---|---|
| baseline vs comp（同输入、同协议，五折配对） | 组成监督是否改善 medium↔small 与跨景泛化 |
| comp vs comp(w_ord=0) / comp(hard_ce=all) | 序项与纯多边形约束各自的作用 |
| res80 vs pooled（本档案 + AutoICE train） | 补上 DMI 区域的 big/vast 能否打破类别-地域混杂 |
| res80 / pooled / a4a_baseline 在两侧测试集上交叉评估 | 跨时期、跨冰图服务的迁移 |

### 6.3 评估（两条轴都要报）

- 像素级：沿用主导类标签，macro mIoU、逐类 IoU/F1、hard-ice；AutoICE 测试集另报官方 weighted F1。
- 多边形级（`tools.evaluate --composition-dir`）：预测组成与冰图组成的 TV 距离（像元加权与多边形平均）、冰类 CDF 的 EMD、
  纯/混合多边形的主导类一致率。基线用同一脚本重评，公平比较。
- 统计：五折配对（`paired_cv`），按区域与冰图服务分开报告。
