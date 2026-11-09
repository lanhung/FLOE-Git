"""SegFormer (Xie et al., 2021) on SMP's MiT encoder, with the PGSG extensions.

The baseline is a MiT encoder (default mit_b2, options['segformer_encoder_name']) plus a
pure-PyTorch all-MLP head. PGSG adds an evidence-conditioned per-pixel scale gate
(contrast-invariant variogram profile + polarimetric-ratio prior) and an optional
scale-axis transformer. Every other module is an optional ablation behind an options flag.
"""

import math
import types

import torch
import torch.nn as nn
import torch.nn.functional as F


class _SegformerHead(nn.Module):
    """All-MLP decoder: project each scale to embed_dim, upsample to
    finest scale, concatenate, fuse, dropout, classify."""

    def __init__(self, in_channels: list, embed_dim: int, n_classes: int,
                 scale_gate: bool = False, gate_mode: str = 'tanh',
                 gate_groups: int = 1, gate_temp: float = 1.0,
                 evidence_ch: int = 0):
        super().__init__()
        # 读数打印的计数器（普通 int，不进 state_dict）。
        self._pre_calls = 0
        self._gate_calls = 0
        self.n_scales = len(in_channels)
        self.embed_dim = embed_dim
        self.projections = nn.ModuleList([
            nn.Conv2d(c, embed_dim, 1, bias=False) for c in in_channels
        ])
        self.fuse = nn.Sequential(
            nn.Conv2d(embed_dim * len(in_channels), embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout2d(0.1)
        self.cls = nn.Conv2d(embed_dim, n_classes, 1)

        # 可选逐像素尺度门控：从 concat 预测每个 stage（及每组通道）的门 g，乘回该 stage 的通道。
        #   tanh:    g = 1 + tanh(a) ∈ (0, 2)，各尺度独立缩放；
        #   softmax: g = S·softmax(a / T)，Σ_s g_s = S，尺度之间竞争，T<1 竞争更锐。
        # 末层 weight/bias 零初始化 ⇒ a≡0 ⇒ 两种模式下 g 都精确为 1.0，初始与无门控基线逐位等价。
        # 门控必须在 projections/fuse/cls 之后构造：Conv2d 初始化消耗全局 RNG，建在前面会
        # 平移共有权重的初始化，相当于换了 seed。
        assert gate_mode in ('tanh', 'softmax'), f'未知 gate_mode: {gate_mode}'
        assert embed_dim % gate_groups == 0, \
            f'embed_dim {embed_dim} 不能被 gate_groups {gate_groups} 整除'
        assert gate_temp > 0, f'gate_temp 必须 >0，得到 {gate_temp}'
        assert gate_mode == 'softmax' or gate_temp == 1.0, \
            'gate_temp 仅对 softmax 门控有意义（tanh 模式请保持 1.0）'
        # evidence_ch>0：门控首层额外吃证据张量（vario 剖面 + 极化比先验）。证据只进门控的
        # 输入、不进主特征通路；末层零初始化不变，初始仍有 g≡1。
        assert evidence_ch == 0 or scale_gate, \
            'evidence_ch>0 需要 scale_gate=True（证据没有别的消费者）'
        self.evidence_ch = int(evidence_ch)
        self.gate_mode = gate_mode
        self.gate_groups = int(gate_groups)
        self.gate_temp = float(gate_temp)
        self.scale_gate = None
        if scale_gate:
            self.scale_gate = nn.Sequential(
                nn.Conv2d(embed_dim * self.n_scales + self.evidence_ch, 64, 1,
                          bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, self.n_scales * self.gate_groups, 1),
            )
            nn.init.zeros_(self.scale_gate[-1].weight)
            nn.init.zeros_(self.scale_gate[-1].bias)
        # 尺度轴 transformer 的挂载点。这里只占位：它在 SegFormer.__init__ 末尾构造后再赋值，
        # 以免其 RNG 消耗平移证据模块的初始化。state_dict 前缀仍为 heads.<chart>.scale_xattn.*。
        self.scale_xattn = None

    def forward(self, features: list, extra: torch.Tensor = None,
                extra_pre: torch.Tensor = None,
                gate_evidence: torch.Tensor = None) -> torch.Tensor:
        target = features[0].shape[-2:]
        projected = []
        for feat, proj in zip(features, self.projections):
            x = proj(feat)
            if x.shape[-2:] != target:
                x = F.interpolate(x, size=target, mode='bilinear', align_corners=False)
            projected.append(x)
        # 可选尺度轴 transformer：先跨尺度通信（零初始化残差），再 concat + 门控选择。
        if self.scale_xattn is not None:
            projected = self.scale_xattn(projected, gate_evidence)
        x = torch.cat(projected, dim=1)
        if self.scale_gate is not None:
            # 用 view 广播而不是 repeat_interleave，避免实体化一份 [B, S*E, H, W]。
            # concat 与门控输出都是 stage 主序、组次序，view 后按 dim=1/2 广播即对齐。
            B, _, Hh, Ww = x.shape
            G, S, E = self.gate_groups, self.n_scales, self.embed_dim
            # 证据缺失或多余都是接线错误，在这里直接报错。
            assert (self.evidence_ch > 0) == (gate_evidence is not None), \
                f'evidence_ch={self.evidence_ch} 与 gate_evidence ' \
                f'{"缺失" if gate_evidence is None else "多余"} 不匹配（接线 bug）'
            if self.evidence_ch > 0:
                if gate_evidence.shape[-2:] != x.shape[-2:]:
                    gate_evidence = F.interpolate(
                        gate_evidence, size=x.shape[-2:],
                        mode='bilinear', align_corners=False)
                gate_in = torch.cat([x, gate_evidence.to(x.dtype)], dim=1)
            else:
                gate_in = x
            a = self.scale_gate(gate_in).view(B, S, G, 1, Hh, Ww)
            if self.gate_mode == 'softmax':
                # Σ_s g_s ≡ S
                g = S * torch.softmax(a.float() / self.gate_temp, dim=1)
            else:
                g = 1.0 + torch.tanh(a)                   # 各尺度独立缩放
            # 只读监控（no_grad，不耗 RNG）：各 stage 平均门值；选择熵（≈ln S 表示没在选，
            # ≈0 表示硬选择）；组间 std（≈0 表示分组没被用）。前几次按几何间隔打印，之后每 2 万次一次。
            self._gate_calls += 1
            if (self._gate_calls in (1, 100, 1000, 10000)
                    or self._gate_calls % 20000 == 0):
                with torch.no_grad():
                    gf = g.float()
                    stage_mean = gf.mean(dim=(0, 2, 3, 4, 5)).tolist()
                    msg = (f'[SegformerHead] gate #{self._gate_calls} '
                           f'({"train" if self.training else "val"}): '
                           'stage-mean g = ['
                           + ', '.join(f'{v:.3f}' for v in stage_mean) + ']')
                    if self.gate_mode == 'softmax':
                        p = (gf / S).clamp_min(1e-12)
                        ent = -(p * p.log()).sum(dim=1).mean().item()
                        # 同时打印 |a|：熵下降可能来自温度 T，也可能只是门 logit 变大。
                        a_abs = a.float().abs().mean().item()
                        msg += (f' | 选择熵 {ent:.3f}/{math.log(S):.3f} nats'
                                '（≈上限=没在选尺度）'
                                f' | |a|={a_abs:.3f} T={self.gate_temp:g}')
                    if G > 1:
                        msg += (f' | 组间 std {gf.std(dim=2).mean().item():.3f}'
                                '（≈0=分组容量未被用）')
                    # 证据贡献比：用 conv1 的两个权重切片分别卷特征块与证据块，比较两路输出的 RMS。
                    # 权重 RMS 不可直接比较（两块列数不同、输入幅度不同），所以看激活级贡献。
                    # 统计在 fp32 里累加，避免 fp16 溢出。
                    if self.evidence_ch > 0:
                        w1 = self.scale_gate[0].weight
                        n_main = S * E
                        y_f = F.conv2d(x, w1[:, :n_main].to(x.dtype))
                        y_e = F.conv2d(gate_evidence.to(x.dtype),
                                       w1[:, n_main:].to(x.dtype))
                        c_f = y_f.pow(2).mean(dtype=torch.float32).sqrt()
                        c_e = y_e.pow(2).mean(dtype=torch.float32).sqrt()
                        contrib = (c_e / c_f.clamp_min(1e-12)).item()
                        rms_f = w1[:, :n_main].pow(2).mean(
                            dtype=torch.float32).sqrt().item()
                        rms_e = w1[:, n_main:].pow(2).mean(
                            dtype=torch.float32).sqrt().item()
                        msg += (f' | 证据/特征激活贡献比 {contrib:.3f}'
                                '（趋 0=证据被闲置；升=门控在听证据）'
                                f' | conv1 权重 RMS f {rms_f:.4f} / e {rms_e:.4f}')
                print(msg, flush=True)
            x = (x.view(B, S, G, E // G, Hh, Ww)
                 * g.to(x.dtype)).view(B, -1, Hh, Ww)
        # extra_pre 加在 fuse 之前的 concat 上，使其贡献经 fuse 的 conv+BN+ReLU 与 encoder
        # 特征非线性混合（加在 fuse 之后只剩 1×1 cls，等于一个线性偏置）。
        if extra_pre is not None:
            if extra_pre.shape[-2:] != x.shape[-2:]:
                extra_pre = F.interpolate(extra_pre, size=x.shape[-2:],
                                          mode='bilinear', align_corners=False)
            # pre-fuse 注入会改变 fuse 里 BN 的输入；若 extra_pre 长到与 concat 同量级，BN 会
            # 压低 encoder 通路的有效幅度。按几何间隔打印幅度比用于排查。
            self._pre_calls += 1
            if self._pre_calls in (1, 100, 1000, 10000, 100000):
                with torch.no_grad():
                    # 范数在 fp32 里累加：autocast 下 x 是 fp16，norm 的结果会被转回 fp16，大张量上
                    # 溢出成 inf，比值读成 0。pow/sum(dtype=) 也不实体化 fp32 副本。
                    num = extra_pre.pow(2).sum(dtype=torch.float32).sqrt()
                    den = x.pow(2).sum(dtype=torch.float32).sqrt().clamp_min(1e-12)
                    ratio = (num / den).item()
                print(f'[SegformerHead] forward #{self._pre_calls} '
                      f'({"train" if self.training else "val"}): '
                      f'‖extra_pre‖/‖concat‖ = {ratio:.4f}'
                      f'（远大于 ~1 = vario 在抢占 fuse 的带宽）', flush=True)
            x = x + extra_pre.to(x.dtype)
        x = self.fuse(x)
        # detail branch 在 fuse 之后残差相加，fuse 形状与基线相同。从不含 detail_branch.*
        # 的 checkpoint 加载需要 strict=False。
        if extra is not None:
            if extra.shape[-2:] != x.shape[-2:]:
                extra = F.interpolate(extra, size=x.shape[-2:],
                                      mode='bilinear', align_corners=False)
            # .to(x.dtype)：若旁路返回 fp32，避免把尾部提升成 fp32。
            x = x + extra.to(x.dtype)
        x = self.dropout(x)
        return self.cls(x)


class _DetailBranch(nn.Module):
    """全分辨率纹理旁路，在 stride-4 上与 head 的融合特征残差相加。

    以比 MiT stem（7×7/s4）更缓和的方式下采样（s1→s2→s2），为 1~6 px 尺度的纹理留出通道。
    只零初始化 proj：初始贡献为 0。不要再叠一个零初始化的标量门，双零级联在 AdamW 下
    是零梯度不动点，分支无法启动。
    """

    def __init__(self, in_ch: int, embed_dim: int):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 48, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.proj = nn.Conv2d(64, embed_dim, 1, bias=False)
        nn.init.zeros_(self.proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.convs(x))


class _VarioTexture(nn.Module):
    """对比度不变的二阶纹理模块（可微变异函数 / 归一化结构函数剖面）。

    在特征 f 的若干 probe 通道上，对每个 lag r 计算局部结构函数

        γ_r(x) = mean_win[(f(x) − f(x±r))²]        （±r 两轴共 4 个位移，固定窗口）

    并只输出跨 lag 归一化后的剖面

        v_r(x) = log(γ_r + ε) − log(mean_r γ_r + ε)

    v 对 f → a·f + b 不变（γ ≫ ε 时精确）：增益使每个 log γ_r 平移同一常数、被去均值消掉，
    偏置对差分无影响。幅度那一路（log mean_r γ）被刻意丢弃，剖面形状即相关长度的估计。

    stride-4 特征上（80 m 像元）1 格 = 320 m，默认 radii (1,2,4,8,16) 对应 320 m~5.12 km。
    估计器：4 方向对称位移使 γ_r 以 x 为中心、对 flip/rot90 等变；固定 11×11 窗口使各 lag
    的估计噪声相同（n≈484）。只有 profile() 走 fp32（平方后取 log，fp16 会下溢）。
    clamp(±6) 是 1-Lipschitz，不破坏不变性；首次前向打印 γ 分位数与 clamp 命中率。

    zero_proj=True（主通路残差）时只零初始化 proj；用作门控证据时 zero_proj=False，
    初始等价由门控末层的零初始化保证。nn.init.zeros_ 不耗 RNG，两种取值的 RNG 消耗相同。
    """

    def __init__(self, in_ch: int, out_ch: int,
                 radii=(1, 2, 4, 8, 16), n_probe: int = 16,
                 mid: int = 64, win: int = 11, eps: float = 1e-6,
                 clamp: float = 6.0, out_pool: int = 1,
                 zero_proj: bool = True):
        super().__init__()
        # out_pool>1：在输入分辨率上算剖面，再平均池化到 head 的 stride-4 网格（fine_vario 用）。
        self.out_pool = int(out_pool)
        self.radii = tuple(int(r) for r in radii)
        assert all(r >= 1 for r in self.radii), 'radii 必须为正整数格数'
        self.win = int(win)
        assert self.win >= 1 and self.win % 2 == 1, 'win 必须是正奇数'
        self.eps = float(eps)
        self.clamp = float(clamp)
        self._diag_done = False
        self._warned_lag = False

        # 探针：把 C 维特征线性压到 n_probe 个标量场。逐通道增益会被跨 lag 归一化消掉，
        # 所以这里不做标准化。
        self.probe = nn.Conv2d(in_ch, n_probe, 1, bias=False)
        self.mix = nn.Sequential(
            nn.Conv2d(n_probe * len(self.radii), mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
        )
        self.proj = nn.Conv2d(mid, out_ch, 1, bias=False)
        if zero_proj:
            nn.init.zeros_(self.proj.weight)

    def _gamma(self, f: torch.Tensor, r: int) -> torch.Tensor:
        """局部结构函数 γ_r：±r 两轴 4 个位移在固定窗口内平均，返回 [B,G,H,W]。

        边界用 reflect（近似保持二阶统计量；replicate 会让大 lag 的 γ 在边缘偏小）。
        位移截断到 min(r, dim−1)：reflect 要求 pad < 维长，且 lag 超出特征图时 γ 饱和到 sill。
        """
        h, w = f.shape[-2:]
        # reflect 的代价：越界位置的有效 lag 变成 |2(h−1)−2x−r|，在 x = h−1−r/2 处为 0。
        # 受影响比例约 r/h，且是与 tile 网格绑定的固定图案；r/h > 1/4 时告警一次。
        if (not self._warned_lag) and r * 4 > min(h, w):
            self._warned_lag = True
            print(f'[VarioTexture] ⚠️ 特征图 {h}×{w} 上 r={r}：约 {100 * r / min(h, w):.0f}% 的'
                  f'位置因 reflect 而有效 lag 失真（x=h−1−r/2 处退化为 0 lag）。该 lag 通道'
                  f'不可信 —— 给这个 stage 单独配更小的 radii（per-stage radii 已支持）。',
                  flush=True)
        rh, rw = min(r, h - 1), min(r, w - 1)
        pad = self.win // 2
        # 一次对称 pad，之后 4 个位移都只是切片；p 的中心块即 f。
        p = F.pad(f, (rw, rw, rh, rh), mode='reflect')
        # 4 个位移的 d² 先累加再池化一次：avg_pool2d 是线性的且 4 个 d² 形状相同，
        # 结果与分别池化后相加一致（仅求和顺序不同）。
        acc = None
        for dh, dw in ((rh, 0), (-rh, 0), (0, rw), (0, -rw)):
            fs = p[..., rh + dh:rh + dh + h, rw + dw:rw + dw + w]
            d2 = (f - fs).pow(2)
            acc = d2 if acc is None else acc + d2
        return F.avg_pool2d(acc / 4.0, self.win, stride=1, padding=pad,
                            count_include_pad=False)

    def _log_diag(self, g: torch.Tensor, v: torch.Tensor) -> None:
        """首次前向打印一次 γ 分位数、clamp 命中率与剖面的跨 lag 离散度。"""
        self._diag_done = True
        with torch.no_grad():
            # torch.quantile 对输入元素数有上限（2²⁴），先抽样。
            flat = g.reshape(-1).float()
            if flat.numel() > 1_000_000:
                flat = flat[::flat.numel() // 1_000_000]
            q = torch.quantile(flat, torch.tensor([0.01, 0.5, 0.99], device=g.device))
            # 命中率只在 ε 有效域内统计：掩膜区 γ≡0 ⇒ v≡0 永不命中，计入会稀释读数。
            live = (g.amax(dim=2, keepdim=True) > 100 * self.eps).expand_as(v)
            n_live = live.sum()
            frac_live = (n_live.float() / live.numel()).item()
            hit = (((v.abs() >= self.clamp - 1e-6) & live).sum().float()
                   / n_live.clamp_min(1)).item()
            # 剖面是否被压平：加性 nugget（斑点残余）对每个 lag 贡献相同的 2σ²，它大时 v→0，
            # 归一化挡不住。跨 lag std 很小说明模块几乎没有输出信号。
            per_lag = v.mean(dim=(0, 1, 3, 4))            # 每个 lag 的空间均值
            xstd = v.std(dim=2)                            # 跨 lag 标准差（逐像素）
            xf = xstd.reshape(-1).float()
            if xf.numel() > 1_000_000:
                xf = xf[::xf.numel() // 1_000_000]
            xq = torch.quantile(xf, torch.tensor([0.5, 0.9], device=g.device))
        print(f'[VarioTexture] γ 分位数 p1/p50/p99 = {q[0]:.3e}/{q[1]:.3e}/{q[2]:.3e} '
              f'(ε={self.eps:.0e}，p1 应远大于 ε) | ε 有效域占比 {100 * frac_live:.1f}% '
              f'| clamp(±{self.clamp:g}) 命中率(仅有效域) {100 * hit:.3f}% '
              f'(应 ≪1%；命中=该 lag 通道剖面被截断、失去对 γ 比值的分辨力。'
              f'clamp 是 1-Lipschitz、不破坏不变性)', flush=True)
        print(f'[VarioTexture] 剖面 radii={self.radii} 逐 lag 均值 = '
              f'{[round(float(x), 3) for x in per_lag]} | 跨 lag std p50/p90 = '
              f'{xq[0]:.3f}/{xq[1]:.3f} '
              f'(★ p50 ≲0.05 = 剖面被压平、模块没在说话，此时空结果【不能】读成'
              f'"该 lag 带无信号")', flush=True)

    def profile(self, f: torch.Tensor) -> torch.Tensor:
        """归一化结构函数剖面 v，[B, G*K, H, W]（probe 主序、lag 次序）。

        对 f → a·f + b（a≠0）不变；单独暴露以便直接测试这一性质。
        """
        with torch.cuda.amp.autocast(enabled=False):
            p = self.probe(f.float())                                  # [B,G,H,W]
            g = torch.stack([self._gamma(p, r) for r in self.radii],
                            dim=2)                                     # [B,G,K,H,W]
            v = (torch.log(g + self.eps)
                 - torch.log(g.mean(dim=2, keepdim=True) + self.eps))
            v = v.clamp(-self.clamp, self.clamp)
            if not self._diag_done:
                self._log_diag(g, v)
            return v.flatten(1, 2)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        v = self.profile(f)
        if self.out_pool > 1:
            v = F.avg_pool2d(v, self.out_pool)
        # mix/proj 留在外层 autocast（fp16）里：v∈[−6,6] 精度足够，且省掉 fp32 的
        # [B, 1024, H/4, W/4] 中间张量。
        return self.proj(self.mix(v))


class _ScaleXAttn(nn.Module):
    """尺度轴 transformer：每个像素的 S 个尺度特征（+1 个证据 token）组成一个短序列，
    过一层 pre-LN transformer 做跨尺度通信。

    token 先经 down（E→d）进瓶颈维；输出只取 S 个尺度 token，经零初始化的 out_proj（d→E）
    残差加回各尺度特征 ⇒ 初始与不开模块逐位等价。尺度门控仍在其后做选择。
    注意力对 token 置换等变，tok_embed（零初始化）给每个 token 一个可学习的尺度身份。
    按几何间隔打印 ‖resid‖/‖feat‖ 与尺度 token 对证据 token 的注意力份额。
    """

    def __init__(self, embed_dim: int, n_scales: int, evidence_ch: int = 0,
                 dim: int = 64, heads: int = 4, mlp_ratio: int = 2):
        super().__init__()
        self.n_scales = int(n_scales)
        self.evidence_ch = int(evidence_ch)
        self._calls = 0
        self.down = nn.Linear(embed_dim, dim)
        self.ev_proj = (nn.Conv2d(evidence_ch, dim, 1)
                        if evidence_ch > 0 else None)
        n_tok = self.n_scales + (1 if evidence_ch > 0 else 0)
        self.tok_embed = nn.Parameter(torch.zeros(1, n_tok, dim))
        self.ln1 = nn.LayerNorm(dim)
        # 手写 Q/K/V 而不用 nn.MultiheadAttention：这里的 batch 是 B·H·W 条长度 5 的序列，
        # SDPA 融合核在 batch×heads > 65535 时 GPU 上报 "invalid configuration argument"。
        assert dim % heads == 0
        self.heads = heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_out = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )
        self.out_proj = nn.Linear(dim, embed_dim, bias=False)
        nn.init.zeros_(self.out_proj.weight)      # 只零初始化最后的投影

    def forward(self, projected: list, evidence: torch.Tensor = None) -> list:
        B, E, H, W = projected[0].shape
        S = self.n_scales
        toks = torch.stack([p.flatten(2).transpose(1, 2) for p in projected],
                           dim=2)                              # [B, HW, S, E]
        toks = self.down(toks)                                 # [B, HW, S, d]
        if self.ev_proj is not None:
            assert evidence is not None, \
                '_ScaleXAttn 构造时带 evidence_ch>0，forward 却没收到 evidence（接线 bug）'
            if evidence.shape[-2:] != (H, W):
                evidence = F.interpolate(evidence, size=(H, W),
                                         mode='bilinear', align_corners=False)
            ev = self.ev_proj(evidence.to(projected[0].dtype))
            toks = torch.cat([toks, ev.flatten(2).transpose(1, 2).unsqueeze(2)],
                             dim=2)                            # [B, HW, S+1, d]
        L = toks.shape[2]
        t = toks.reshape(B * H * W, L, -1) + self.tok_embed.to(toks.dtype)
        h = self.ln1(t)
        Bn, d = t.shape[0], t.shape[2]
        dh = d // self.heads
        qkv = (self.qkv(h).reshape(Bn, L, 3, self.heads, dh)
               .permute(2, 0, 3, 1, 4))                        # [3, Bn, nH, L, dh]
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn_w = ((q @ k.transpose(-2, -1)) * dh ** -0.5).softmax(dim=-1)
        attn_out = self.attn_out((attn_w @ v).transpose(1, 2).reshape(Bn, L, d))
        t = t + attn_out
        t = t + self.mlp(self.ln2(t))
        resid = self.out_proj(t[:, :S])                        # [B*HW, S, E]
        resid = resid.reshape(B, H * W, S, E).permute(0, 2, 3, 1) \
                     .reshape(B, S, E, H, W)
        self._calls += 1
        if self._calls in (1, 100, 1000, 10000) or self._calls % 20000 == 0:
            with torch.no_grad():
                num = resid.pow(2).sum(dtype=torch.float32).sqrt()
                den = sum(p.pow(2).sum(dtype=torch.float32)
                          for p in projected).sqrt().clamp_min(1e-12)
                msg = (f'[ScaleXAttn] #{self._calls} '
                       f'({"train" if self.training else "val"}): '
                       f'‖resid‖/‖feat‖ = {(num / den).item():.4f}'
                       '（趋 0=通信结果被闲置）')
                if self.ev_proj is not None:
                    # attn_w 已实体化（[Bn,nH,L,L]），直接读取，不需要额外前向。
                    ev_share = attn_w[:, :, :S, S].mean(
                        dtype=torch.float32).item()
                    msg += (f' | 尺度→证据注意力份额 {ev_share:.3f}'
                            f'（均匀基线 {1.0 / L:.3f}）')
            print(msg, flush=True)
        return [projected[s] + resid[:, s].to(projected[s].dtype)
                for s in range(S)]


class _PolPrior(nn.Module):
    """极化比先验分支：从输入的 pol-ratio 通道生成门控证据。

    pol ratio（HH−HV，dB/z-score 域之差）是不依赖绝对定标的散射机制指纹；散射机制决定
    哪个空间尺度携带浮冰粒度信息，所以把它作为门控的条件输入。
    结构：4×4 平均池化到 stride-4（同时抑制斑点）→ 两层 3×3 conv+BN+ReLU。随机初始化：
    它只进门控，初始等价由门控末层的零初始化保证。
    """

    def __init__(self, out_ch: int = 32, pool: int = 4, mid: int = 16):
        super().__init__()
        self.pool = int(pool)
        self.net = nn.Sequential(
            nn.Conv2d(1, mid, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, pol: torch.Tensor) -> torch.Tensor:
        # pol: [B,1,H,W]，输入张量的最后一个通道
        p = F.avg_pool2d(pol, self.pool)
        return self.net(p)


def _pba_patch_attention(attn: nn.Module, proj: nn.Module, holder: dict,
                         tag: str = '', monitor: bool = False) -> None:
    """物理偏置注意力（PBA）：给一个 smp MiT `Attention` 实例的 forward 打补丁，在注意力
    logits 上加二次极化核偏置

        bias = pol_proj([p_i², p_j², p_i·p_j])，pol_proj: Linear(3, num_heads)，零初始化

    3 项的可学习组合同时包含点积核（w=[0,0,1]）与相似核 −(p_i−p_j)²（w=[−1,−1,2]）。

    - 只替换 forward、不替换模块：encoder 的 state_dict 键不变，pol_proj 注册在
      SegFormer.pba_projs。
    - forward 主体逐行照搬 smp 0.5.0 `mix_transformer.Attention.forward`，零偏置时与原实现
      逐位相同；升级 smp 前需与未打补丁的 encoder 对拍。
    - pol 经 `holder`（SegFormer 上的 dict）传入：SegFormer.forward 调 encoder 前放入、调完
      pop；holder 为空时退化为原始注意力。
    - pol_q 用 adaptive_avg_pool2d 池到该 stage 的 token 网格，pol_kv 再用
      avg_pool2d(sr_ratio) 池到 KV 网格，与 sr conv 的几何一致。
    """
    _calls = [0]          # 闭包计数器（仅 monitor=True 的实例使用，不进 state_dict）

    def forward(self, x, height, width):
        batch_size, N, C = x.shape
        q = (
            self.q(x)
            .reshape(batch_size, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(batch_size, C, height, width)
            x_ = self.sr(x_).reshape(batch_size, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = (
                self.kv(x_)
                .reshape(batch_size, -1, 2, self.num_heads, C // self.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
        else:
            kv = (
                self.kv(x)
                .reshape(batch_size, -1, 2, self.num_heads, C // self.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        pol = holder.get('pol')
        if pol is not None and pol.shape[0] != batch_size:
            # holder 与当前 batch 不一致：encoder 在 SegFormer.forward 之外被调用且 holder 有残留。
            if not holder.get('_warned_bs', False):
                holder['_warned_bs'] = True
                print(f'[PBA] ⚠️ holder 里的 pol batch={pol.shape[0]} 与当前 '
                      f'batch={batch_size} 不符，本次前向跳过极化偏置（只警告一次）',
                      flush=True)
            pol = None
        if pol is not None:
            pol_q = F.adaptive_avg_pool2d(pol, (height, width)) \
                     .flatten(2).transpose(1, 2)                    # [B, N, 1]
            if self.sr_ratio > 1:
                pol_kv = F.avg_pool2d(
                    pol_q.transpose(1, 2).reshape(batch_size, 1, height, width),
                    self.sr_ratio).flatten(2).transpose(1, 2)       # [B, N_kv, 1]
            else:
                pol_kv = pol_q
            # 二次核的 3 个特征：p_i² / p_j² 沿另一轴广播
            n_kv = pol_kv.shape[1]
            feats = torch.stack([
                (pol_q * pol_q).expand(-1, -1, n_kv),               # p_i²
                (pol_kv * pol_kv).transpose(-2, -1).expand(-1, N, -1),  # p_j²
                pol_q @ pol_kv.transpose(-2, -1),                   # p_i·p_j
            ], dim=-1)                                              # [B, N, N_kv, 3]
            bias = proj(feats).permute(0, 3, 1, 2)                  # [B, nH, N, N_kv]
            if monitor:
                _calls[0] += 1
                c = _calls[0]
                if c in (1, 1000, 10000) or c % 50000 == 0:
                    with torch.no_grad():
                        b_rms = bias.pow(2).mean(dtype=torch.float32).sqrt().item()
                        a_rms = attn.pow(2).mean(dtype=torch.float32).sqrt().item()
                    print(f'[PBA {tag}] #{c}: bias RMS {b_rms:.4f} vs attn-logit '
                          f'RMS {a_rms:.3f}（比值 {b_rms / max(a_rms, 1e-12):.4f}；'
                          '趋 0=物理偏置被闲置）', flush=True)
            attn = attn + bias.to(attn.dtype)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(batch_size, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

    attn.forward = types.MethodType(forward, attn)


class _ContextAggregator(nn.Module):
    """Decoder 侧上下文聚合：ASPP 空洞金字塔 + 轻量 non-local，输出经 extra_pre 加到 head 的
    concat 上。

    SegFormer 的 all-MLP head 只做逐点融合，没有跨位置聚合；本模块补上局部到中程（ASPP）与
    patch 内全局（non-local）的上下文。默认挂在 stride-8（patch 256 时 32×32）：在 stride-32
    的 8×8 上，大膨胀率的非中心抽头基本落在 padding 里，ASPP 退化成近似 1×1。
    stride-8 + dilations (1,2,4,8) 对应 640 m~5.12 km。non-local 只看得到当前 patch。
    只零初始化 proj。
    """

    def __init__(self, in_ch: int, out_ch: int, mid: int = 128,
                 dilations=(1, 2, 4, 8)):
        super().__init__()
        self.dilations = tuple(int(d) for d in dilations)
        self._warned = False
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, mid, 3, padding=d, dilation=d, bias=False),
                nn.BatchNorm2d(mid), nn.ReLU(inplace=True))
            for d in dilations
        ])
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, mid, 1, bias=False),
            nn.BatchNorm2d(mid), nn.ReLU(inplace=True))
        self.merge = nn.Sequential(
            nn.Conv2d(mid * (len(dilations) + 1), mid, 1, bias=False),
            nn.BatchNorm2d(mid), nn.ReLU(inplace=True))
        # 轻量 non-local（q/k 降维到 mid/8）
        self.q = nn.Conv2d(mid, mid // 8, 1, bias=False)
        self.k = nn.Conv2d(mid, mid // 8, 1, bias=False)
        self.v = nn.Conv2d(mid, mid, 1, bias=False)
        self.nl_out = nn.Sequential(nn.Conv2d(mid, mid, 1, bias=False),
                                    nn.BatchNorm2d(mid), nn.ReLU(inplace=True))
        self.proj = nn.Conv2d(mid, out_ch, 1, bias=False)
        nn.init.zeros_(self.proj.weight)          # 只零初始化最后的投影

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        B, _, H, W = f.shape
        # 3×3 空洞卷积要求 2d+1 ≤ 边长，否则非中心抽头恒落在 padding 上（该支退化成 1×1）。
        # 首次前向检查并打印一次。
        if not self._warned:
            dead = [d for d in self.dilations if 2 * d + 1 > min(H, W)]
            self._warned = True
            if dead:
                print(f'[ContextAggregator] ⚠️ 特征图 {H}×{W} 上 dilation {dead} 的非中心抽头'
                      f'恒落在 padding 内（梯度为 0，该支退化成 1×1）—— ASPP 的多尺度部分'
                      f'未生效，换 stage 或改小 dilations！', flush=True)
            else:
                print(f'[ContextAggregator] 特征图 {H}×{W}，dilations={self.dilations}，'
                      f'各支有效位置比例 '
                      f'{[round(max(0, min(H, W) - 2 * d) / min(H, W), 2) for d in self.dilations]}',
                      flush=True)
        feats = [b(f) for b in self.branches]
        g = self.gap(f)
        feats.append(g.expand(-1, -1, H, W))
        x = self.merge(torch.cat(feats, dim=1))
        # non-local 残差加回，保留 ASPP 的局部证据
        q = self.q(x).flatten(2).transpose(1, 2)              # [B, HW, c]
        k = self.k(x).flatten(2)                              # [B, c, HW]
        v = self.v(x).flatten(2).transpose(1, 2)              # [B, HW, mid]
        att = torch.softmax(q @ k / (q.shape[-1] ** 0.5), dim=-1)
        nl = (att @ v).transpose(1, 2).reshape(B, -1, H, W)
        x = x + self.nl_out(nl)
        return self.proj(x)


class _OrdinalHead(nn.Module):
    """CORAL 式单调有序输出头：替换 head 的 cls，返回 log 概率作为 logits。

    冰类 cake < small < medium < big < vast 是同一尺寸轴的量化。输出 2 个通道 z_w（水/冰）
    与 t（尺寸分数），外加 4 个保序阈值 b_1 < ... < b_4（b_1 + cumsum(softplus(d))）：

        P(water) = σ(z_w)，P(ice) = 1 − σ(z_w)，s_k = σ(t − b_k)，k = 1..4
        P(cake) = P(ice)(1 − s_1)，P(small) = P(ice)(s_1 − s_2)，...，P(vast) = P(ice)·s_4

    因 ΣP = 1，log_softmax(log P) = log P，下游 CrossEntropyLoss 与评估无需改动。
    与基线不是初始等价的，也不能从基线 checkpoint 暖启动 cls。
    """

    def __init__(self, embed_dim: int, n_classes: int):
        super().__init__()
        assert n_classes == 6, f'_OrdinalHead 假定 6 类（水+5 冰），收到 {n_classes}'
        self.n_classes = n_classes
        self.cls = nn.Conv2d(embed_dim, 2, 1)      # [z_w, t]
        # b_1 初始 -1.5，之后三个间隔 softplus(0.54)≈1.0 → 初始阈值 ≈ [-1.5,-0.5,0.5,1.5]
        self.b0 = nn.Parameter(torch.tensor(-1.5))
        self.dlt = nn.Parameter(torch.full((3,), 0.5413))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        zw, t = self.cls(x).float().split(1, dim=1)          # 概率组合在 fp32 里算，避免 fp16 下溢
        # softplus 加 1e-3 地板：dlt 极负时 softplus 下溢为 0，两个阈值相等，夹在中间的类
        # 概率恒为 0、梯度消失。
        b = torch.cat([self.b0.view(1),
                       self.b0 + torch.cumsum(F.softplus(self.dlt) + 1e-3, 0)])   # 恒保序
        # clamp(±12)：两个 sigmoid 同时饱和时 s_hi−s_lo 抵消为 0，而 clamp_min(1e-8) 处梯度为 0，
        # 最自信的错分像素反而学不动。e⁻¹²≈6e-6 对 6 类足够。
        s = torch.sigmoid((t - b.view(1, -1, 1, 1)).clamp(-12.0, 12.0))   # [B,4,H,W]，沿 k 递减
        p_ice = torch.sigmoid(-zw)                            # = 1 − σ(zw)
        ones = torch.ones_like(s[:, :1])
        zeros = torch.zeros_like(s[:, :1])
        s_hi = torch.cat([ones, s], dim=1)                    # s_0=1
        s_lo = torch.cat([s, zeros], dim=1)                   # s_5=0
        p_size = (s_hi - s_lo).clamp_min(0)                   # [B,5,H,W]，5 个冰类，和为 1
        probs = torch.cat([torch.sigmoid(zw), p_ice * p_size], dim=1)
        return torch.log(probs.clamp_min(1e-8))


class _SKAdapter(nn.Module):
    """Encoder 内的逐像素滤波器组选择适配器（selective-kernel 式）。

    每个 stage 输出上并联一组 depthwise 3×3 空洞卷积（不同 d），由 1×1 的 sel 头逐像素
    softmax 选择，经零初始化的 proj 残差加回；适配后的特征既进 head 也进下一个 stage。
    只零初始化 proj（sel 经 softmax 恒 > 0，无需零初始化）。
    读数只在训练前向中计数和累加：按几何间隔打印一行；epoch 级累加值（detached GPU
    张量，避免每个 batch 同步）由 SegFormer.pop_sk_stats() 取走。
    """

    def __init__(self, ch: int, dilations=(1, 2, 4), tag: str = ''):
        super().__init__()
        self.dilations = tuple(int(d) for d in dilations)
        assert len(self.dilations) >= 2, 'SK 适配器至少要有 2 个分支才谈得上"选择"'
        assert all(d >= 1 for d in self.dilations), f'dilations 必须为正整数：{dilations}'
        self.tag = str(tag)
        # 普通 int / None，不进 state_dict。
        self._calls = 0
        self._warned = False
        self._acc = None
        self.branches = nn.ModuleList([
            nn.Conv2d(ch, ch, 3, padding=d, dilation=d, groups=ch, bias=False)
            for d in self.dilations
        ])
        self.sel = nn.Conv2d(ch, len(self.dilations), 1)
        self.proj = nn.Conv2d(ch, ch, 1, bias=False)
        nn.init.zeros_(self.proj.weight)          # 只零初始化最后的投影

    def _observe(self, x: torch.Tensor, resid: torch.Tensor,
                 sel: torch.Tensor, do_print: bool) -> None:
        """只读观测：累加 epoch 统计，并按需打印一行。不耗 RNG、不进梯度。"""
        with torch.no_grad():
            w_mean = sel.mean(dim=(0, 2, 3))                       # [K]，fp32
            ent = -(sel.clamp_min(1e-12) * sel.clamp_min(1e-12).log()).sum(dim=1).mean()
            # 范数在 fp32 里累加（理由同 _SegformerHead 的 extra_pre 监控）。
            num = resid.pow(2).sum(dtype=torch.float32).sqrt()
            den = x.pow(2).sum(dtype=torch.float32).sqrt().clamp_min(1e-12)
            ratio = num / den
            if self.training:                 # 防御性：唯一调用点已是 training-only
                if self._acc is None:
                    self._acc = {'w': torch.zeros_like(w_mean),
                                 'ent': torch.zeros_like(ent),
                                 'ratio': torch.zeros_like(ratio),
                                 'n': 0}
                self._acc['w'] += w_mean
                self._acc['ent'] += ent
                self._acc['ratio'] += ratio
                self._acc['n'] += 1
            if do_print:
                # 只由训练前向调用，所以不打 train/val 标签。
                print(f'[SKAdapter {self.tag}] train #{self._calls}: '
                      'branch mean sel = ['
                      + ', '.join(f'{v:.3f}' for v in w_mean.tolist())
                      + f'] (d={self.dilations}) | 选择熵 {ent.item():.3f}/'
                      f'{math.log(len(self.dilations)):.3f} nats（≈上限=没在选）'
                      f' | ‖resid‖/‖x‖ = {ratio.item():.4f}', flush=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 首次前向检查各分支的有效位置比例（2d+1 > 边长时非中心抽头恒落在 padding 上）。
        # patch 256 时 stage2/3 只有 16×16 / 8×8，所以它们默认用更小的 dilations。
        if not self._warned:
            self._warned = True
            h, w = x.shape[-2:]
            frac = [max(0, min(h, w) - 2 * d) / min(h, w) for d in self.dilations]
            dead = [d for d, f in zip(self.dilations, frac) if f <= 0]
            # 有效位置比例 <0.3 的分支只发软告警。
            weak = [d for d, f in zip(self.dilations, frac) if 0 < f < 0.3]
            msg = (f'[SKAdapter {self.tag}] 特征图 {h}×{w}，dilations={self.dilations}，'
                   f'各支有效位置比例 {[round(f, 2) for f in frac]}')
            if dead:
                msg += (f' | ⚠️ dilation {dead} 的非中心抽头恒落在 padding 内（梯度为 0、'
                        '该支退化成 1×1）—— 改小该 stage 的 segformer_sk_dilations！')
            elif weak:
                msg += (f' | ⚠️ dilation {weak} 的有效位置比例 <0.3（未死但大部分位置靠 '
                        'padding 支撑，该支的选择权重读数需打折）')
            print(msg, flush=True)
        # 选择权重的 softmax 在 fp32 里做，再转回 x 的 dtype。
        sel = torch.softmax(self.sel(x).float(), dim=1)
        y = None
        for i, branch in enumerate(self.branches):
            t = sel[:, i:i + 1].to(x.dtype) * branch(x)
            y = t if y is None else y + t
        resid = self.proj(y)
        # 只在训练时计数：验证要前向几万个 tile，会占掉几何间隔的打印点并混进 epoch 累加。
        if self.training:
            self._calls += 1
            do_print = (self._calls in (1, 100, 1000, 10000)
                        or self._calls % 20000 == 0)
            self._observe(x, resid, sel, do_print)
        return x + resid


class _ScaleRegHead(nn.Module):
    """逐像素 log10-尺寸回归辅助头，接在 encoder stride-4 特征上（1×1 → 64 → ReLU → 1×1）。

    目标与损失见 tools/train.py 的 scale_reg_aux_loss。只在训练时调用；不在主通路上，
    所以不做零初始化。
    """

    def __init__(self, in_ch: int, mid: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mid, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _GradReverse(torch.autograd.Function):
    """梯度反转层（Ganin & Lempitsky, 2015）：前向恒等，反向把梯度乘以 −λ。

    判别器损失按权重 1.0 加进总损失，对抗强度完全由 λ（SegFormer.dann_lambda）控制。
    λ 存在 ctx 上，不进 state_dict。
    """

    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = float(lambd)
        return x.view_as(x)          # 恒等，但返回新 view 以免 autograd 把它当 inplace

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None      # 第二个输入 lambd 无梯度


class _DANNHead(nn.Module):
    """场景判别器：GAP → Linear(ch, 256) → ReLU → Linear(256, n_domains)。

    吃 encoder stride-4 特征，patch 级地分类"来自哪一景"。用 GAP 是因为要去掉的是景级的
    全局性质（定标、亮度），而不是逐像素结构。只在训练时调用。
    """

    def __init__(self, in_ch: int, n_domains: int, mid: int = 256):
        super().__init__()
        assert n_domains >= 2, f'n_domains 必须 ≥2，得到 {n_domains}'
        self.n_domains = int(n_domains)
        self.net = nn.Sequential(
            nn.Linear(in_ch, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, n_domains),
        )

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        # GAP 在 fp32 里累加（dtype= 不实体化 fp32 副本）。
        return self.net(f.mean(dim=(2, 3), dtype=torch.float32))   # [B,C,h,w] → [B,n_domains]


class SegFormer(nn.Module):
    """SegFormer: MiT encoder (via SMP, default mit_b2) + SegformerHead.

    Optional modules are built after the encoder and heads, each after every module that
    existed before it, so enabling one leaves the initialisation of all earlier-built weights
    unchanged under a fixed seed. Add new modules at the end of __init__.

    Requires: pip install segmentation-models-pytorch
    """

    def __init__(self, options: dict):
        super().__init__()
        try:
            import segmentation_models_pytorch as smp
        except ImportError as e:
            raise ImportError(
                "segmentation_models_pytorch not found. "
                "Run: pip install segmentation-models-pytorch"
            ) from e

        in_ch = len(options['train_variables'])
        if options.get('month_encoding', False):
            in_ch += 2
        if options.get('pol_ratio_channel', False):
            in_ch += 1
        if options.get('texture_channels', False):
            from floe.loaders import N_TEXTURE_CHANNELS  # 懒导入避免顶层循环依赖
            in_ch += N_TEXTURE_CHANNELS  # 3 个多尺度局部 std + 1 个边界密度

        self.charts = options['charts']
        self.n_classes = options['n_classes']

        # 借一个临时 SMP FPN 取出 MiT encoder，decoder 换成自己的 SegformerHead。
        # segformer_encoder_name: mit_b0~mit_b5（默认 mit_b2），head 经 encoder.out_channels 自动适配。
        # segformer_encoder_weights: None（从零）或 'imagenet'（SMP 把 3 通道 patch embedding
        # 循环平铺到 in_ch 通道）。
        _tmp = smp.FPN(
            encoder_name=options.get('segformer_encoder_name', 'mit_b2'),
            encoder_weights=options.get('segformer_encoder_weights', None),
            in_channels=in_ch,
            classes=1,
        )
        self.encoder = _tmp.encoder
        del _tmp

        # encoder.out_channels 的最后 4 项是 4 个 transformer stage。
        feat_channels = list(self.encoder.out_channels[-4:])

        embed_dim = 256
        # 证据门控：这里只读 options（不耗 RNG），证据模块在 __init__ 末尾构造。
        self._gate_ev_on = bool(options.get('segformer_gate_evidence', False))
        _EV_VARIO_CH, _EV_POL_CH = 64, 32
        ev_ch = (_EV_VARIO_CH + _EV_POL_CH) if self._gate_ev_on else 0
        if self._gate_ev_on:
            assert options.get('segformer_scale_gate', False), \
                'segformer_gate_evidence=True 需要 segformer_scale_gate=True（证据的' \
                '唯一消费者是门控）'
            assert options.get('pol_ratio_channel', False), \
                'segformer_gate_evidence=True 需要 pol_ratio_channel=True（极化比' \
                '先验分支从输入最后一个通道取 pol ratio）'
            assert not options.get('texture_channels', False), \
                'segformer_gate_evidence 与 texture_channels 不兼容：后者会追加在 ' \
                'pol_ratio 之后，"pol 是最后一个通道"的取法失效'
            # 记下输入通道数，forward 里断言，防止新加派生通道后"pol=最后一个通道"静默错位。
            self._gate_ev_in_ch = in_ch
        self.heads = nn.ModuleDict({
            chart: _SegformerHead(
                feat_channels, embed_dim, self.n_classes[chart],
                scale_gate=options.get('segformer_scale_gate', False),
                gate_mode=options.get('segformer_gate_mode', 'tanh'),
                gate_groups=options.get('segformer_gate_groups', 1),
                gate_temp=options.get('segformer_gate_temp', 1.0),
                evidence_ch=ev_ch)
            for chart in self.charts
        })
        # 有序头替换 head 的 cls（见 _OrdinalHead）。
        if options.get('segformer_ordinal_head', False):
            for chart in self.charts:
                self.heads[chart].cls = _OrdinalHead(embed_dim, self.n_classes[chart])

        # 以下可选模块按固定顺序构造在 encoder/heads 之后（见类 docstring）。
        if options.get('segformer_detail_branch', False):
            self.detail_branch = _DetailBranch(in_ch, embed_dim)
        else:
            self.detail_branch = None

        # 对比度不变纹理残差（经 extra_pre 注入）。vario_stage 可为 int 或 list，每个 stage 一个
        # 独立模块；stride 4 / 8 的 1 格分别为 320 m / 640 m。
        vs = options.get('segformer_vario_stage', 0)
        assert isinstance(vs, (int, list, tuple)) and not isinstance(vs, bool), \
            f'segformer_vario_stage 必须是 int 或 list/tuple，收到 {type(vs).__name__}: {vs!r}'
        self.vario_stages = ([int(vs)] if isinstance(vs, int)
                             else [int(v) for v in vs])
        # radii 可按 stage 分别给（嵌套 list）：不同 stage 的格距与特征图边长不同，
        # 可安全使用的最大 lag 也不同（见 _gamma 的 reflect 说明）。
        radii_cfg = options.get('segformer_vario_radii', (1, 2, 4, 8, 16))
        per_stage = (isinstance(radii_cfg, (list, tuple)) and radii_cfg
                     and isinstance(radii_cfg[0], (list, tuple)))
        if per_stage:
            assert len(radii_cfg) == len(self.vario_stages), \
                f'per-stage radii 有 {len(radii_cfg)} 组，但 vario_stage 有 ' \
                f'{len(self.vario_stages)} 个'
            radii_list = [tuple(r) for r in radii_cfg]
        else:
            radii_list = [tuple(radii_cfg)] * len(self.vario_stages)
        if options.get('segformer_vario_texture', False):
            for s in self.vario_stages:
                assert 0 <= s < len(feat_channels), \
                    f'segformer_vario_stage 越界：{s} 不在 [0,{len(feat_channels)})'
            assert len(set(self.vario_stages)) == len(self.vario_stages), \
                'segformer_vario_stage 有重复项'
            # out_ch = embed_dim × stage 数：残差加在 head 的 concat 上（fuse 之前）。
            self.vario_texture = nn.ModuleList([
                _VarioTexture(
                    feat_channels[s], embed_dim * len(feat_channels),
                    radii=r,
                    n_probe=options.get('segformer_vario_probes', 16),
                    win=options.get('segformer_vario_win', 11),
                ) for s, r in zip(self.vario_stages, radii_list)
            ])
        else:
            self.vario_texture = None

        # 上下文聚合模块，与 vario 共用 extra_pre 注入口。默认挂 stride-8（stage 1），
        # 原因见 _ContextAggregator。
        if options.get('segformer_context_module', False):
            self.context_stage = int(options.get('segformer_context_stage', 1))
            assert 0 <= self.context_stage < len(feat_channels), \
                f'segformer_context_stage 越界：{self.context_stage}'
            self.context_module = _ContextAggregator(
                feat_channels[self.context_stage], embed_dim * len(feat_channels),
                dilations=options.get('segformer_context_dilations', (1, 2, 4, 8)))
        else:
            self.context_module = None
            self.context_stage = -1

        # 细尺度归一化纹理：在原始输入分辨率上算变异函数剖面，lag 2~8 px = 160~640 m，
        # 这一带在 stem（7×7/s4）之后的特征上已经测不到。最小 lag 取 2 px 以避开斑点 nugget；
        # 剖面池化到 stride-4 后再进 mix/proj，控制开销。
        if options.get('segformer_fine_vario', False):
            self.fine_vario = _VarioTexture(
                in_ch, embed_dim * len(feat_channels),
                radii=options.get('segformer_fine_vario_radii', (2, 3, 4, 6, 8)),
                n_probe=options.get('segformer_fine_vario_probes', 8),
                win=options.get('segformer_fine_vario_win', 7),
                out_pool=4)
        else:
            self.fine_vario = None

        # SupCon 投影头（训练时输出 '_supcon'，stride-4）。不在主通路上，不做零初始化。
        if options.get('segformer_supcon', False):
            _c0 = feat_channels[0]
            self.supcon_proj = nn.Sequential(
                nn.Conv2d(_c0, 128, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 64, 1),
            )
        else:
            self.supcon_proj = None

        # Encoder 内 SK 适配器。注册在顶层 self.sk_adapters 而不是 'encoder.' 命名空间下：
        # functions._optimizer_param_groups 按名字前缀分组，否则会拿到 encoder 的小 LR。
        if options.get('segformer_sk_adapter', False):
            sk_stages = [int(s) for s in options.get('segformer_sk_stages', [0, 1, 2, 3])]
            for s in sk_stages:
                assert 0 <= s < len(feat_channels), \
                    f'segformer_sk_stages 越界：{s} 不在 [0,{len(feat_channels)})'
            assert len(set(sk_stages)) == len(sk_stages), 'segformer_sk_stages 有重复项'
            # 默认 per-stage dilations：patch 256 时 stage2/3 只有 16×16 / 8×8，粗 stage 用 (1,2,3)。
            _dil_default = {0: (1, 2, 4), 1: (1, 2, 4), 2: (1, 2, 3), 3: (1, 2, 3)}
            dil_cfg = options.get('segformer_sk_dilations', None)
            assert dil_cfg is None or len(dil_cfg) > 0, 'segformer_sk_dilations 不能为空'
            if dil_cfg is None:
                dil_list = [_dil_default[s] for s in sk_stages]
            elif isinstance(dil_cfg[0], (list, tuple)):   # 嵌套 = 逐 stage
                assert len(dil_cfg) == len(sk_stages), \
                    f'per-stage dilations 有 {len(dil_cfg)} 组，但 sk_stages 有 {len(sk_stages)} 个'
                dil_list = [tuple(d) for d in dil_cfg]
            else:                                          # 单组 = 全 stage 同用
                dil_list = [tuple(dil_cfg)] * len(sk_stages)
            # _encoder_forward_sk 复刻 smp 0.4/0.5 的分段前向。这里只检查属性存在：smp ≤0.3 的
            # 属性名相同但调用约定不同，版本变化要靠与 self.encoder(x) 逐位对拍来确认。
            for i in range(1, 5):
                for attr in (f'patch_embed{i}', f'block{i}', f'norm{i}'):
                    if not hasattr(self.encoder, attr):
                        raise RuntimeError(
                            f'segformer_sk_adapter=True 需要显式分段前向，但 encoder 缺少 '
                            f'属性 {attr!r}（当前 encoder 类型 '
                            f'{type(self.encoder).__name__}）—— 不是 smp 的 MiT encoder，'
                            '或 smp 版本结构已变，需更新 _encoder_forward_sk。')
            _depth = getattr(self.encoder, '_depth', 5)
            if _depth != 5:
                raise RuntimeError(
                    f'segformer_sk_adapter=True 假定 encoder._depth==5（返回 '
                    f'[input, dummy] + 4 个 stage），实际 {_depth}。')
            self.sk_adapters = nn.ModuleDict({
                str(s): _SKAdapter(feat_channels[s], d, tag=f's{s}')
                for s, d in zip(sk_stages, dil_list)
            })
        else:
            self.sk_adapters = None

        # 尺寸回归辅助头（训练时输出 '_scale_reg'）。
        if options.get('segformer_scale_reg', False):
            self.scale_reg_head = _ScaleRegHead(feat_channels[0])
        else:
            self.scale_reg_head = None

        # DANN 场景判别器（训练时输出 '_dann'）。segformer_dann_domains（= train 景数）必须显式
        # 给出，不设默认值，以免判别器类数与实际域数静默不一致。
        if options.get('segformer_dann', False):
            if 'segformer_dann_domains' not in options:
                raise ValueError(
                    'segformer_dann=True 必须显式给出 segformer_dann_domains（= train '
                    '景数，417 数据集是 375）—— 它决定判别器输出维度与域 acc 的基线 1/N，'
                    '不设默认值。')
            self.dann_head = _DANNHead(feat_channels[0],
                                       int(options['segformer_dann_domains']))
            # 普通 float 属性（不进 state_dict），tools/train.py 每 epoch 按 warmup 覆写；
            # 初值 0 即没有对抗梯度。
            self.dann_lambda = 0.0
        else:
            self.dann_head = None

        # PGSG 证据模块，forward 里 concat 成 [B, 96, H/4, W/4] 传给 head 的 gate_evidence。
        #   gate_ev_vario：vario 剖面证据，radii 1~16 @stride-4 = 320 m~5.12 km，zero_proj=False；
        #   gate_ev_pol：极化比先验，取输入的最后一个通道。
        # 名字不含 'encoder.'，归入满 LR 的参数组。
        if self._gate_ev_on:
            self.gate_ev_vario = _VarioTexture(
                feat_channels[0], _EV_VARIO_CH,
                radii=options.get('segformer_gate_ev_radii', (1, 2, 4, 8, 16)),
                n_probe=options.get('segformer_gate_ev_probes', 16),
                win=options.get('segformer_gate_ev_win', 11),
                zero_proj=False)
            self.gate_ev_pol = _PolPrior(out_ch=_EV_POL_CH, pool=4)
        else:
            self.gate_ev_vario = None
            self.gate_ev_pol = None

        # 尺度轴 transformer：在 gate_ev 之后构造，再挂到 head 的占位符上（见 _SegformerHead）。
        if options.get('segformer_scale_xattn', False):
            for chart in self.charts:
                self.heads[chart].scale_xattn = _ScaleXAttn(
                    embed_dim, len(feat_channels), evidence_ch=ev_ch)

        # 物理偏置注意力 PBA，最后构造。pol_proj 注册在顶层 self.pba_projs：归入满 LR 参数组，
        # 且 monkey-patch 不改变 encoder 的 state_dict 键。
        if options.get('segformer_pba', False):
            assert options.get('pol_ratio_channel', False), \
                'segformer_pba=True 需要 pol_ratio_channel=True（偏置从输入最后一个' \
                '通道取 pol ratio）'
            assert not options.get('texture_channels', False), \
                'segformer_pba 与 texture_channels 不兼容（pol 必须是最后一个通道）'
            self._pba_in_ch = in_ch
            pba_stages = [int(s) for s in
                          options.get('segformer_pba_stages', [0, 1, 2, 3])]
            assert len(set(pba_stages)) == len(pba_stages), 'pba_stages 有重复项'
            for s in pba_stages:
                assert 0 <= s <= 3, f'segformer_pba_stages 越界：{s}'
            # 只检查结构存在；调用约定需与未打补丁的 encoder 零偏置对拍确认。
            for s in pba_stages:
                if not hasattr(self.encoder, f'block{s + 1}'):
                    raise RuntimeError(
                        f'segformer_pba 需要 encoder.block{s + 1}（smp MiT 结构），'
                        f'当前 encoder 类型 {type(self.encoder).__name__} 没有——'
                        'smp 版本结构已变，需更新 _pba_patch_attention。')
            self._pba_holder = {}
            _projs = []
            # 与 EMA 不兼容：deepcopy 只复制绑定方法的 __self__，闭包里的 proj/holder 仍指向
            # 原模型，EMA 副本的 PBA 会静默失效。
            assert not options.get('ema', False), \
                'segformer_pba 与 ema 同开未适配：deepcopy 不会跟走 monkey-patch 的闭包'
            for s in pba_stages:
                blocks = getattr(self.encoder, f'block{s + 1}')
                for bi, blk in enumerate(blocks):
                    proj = nn.Linear(3, blk.attn.num_heads)
                    nn.init.zeros_(proj.weight)
                    nn.init.zeros_(proj.bias)
                    _projs.append(proj)
                    # 每个 stage 只让第一个 block 打印监控。
                    _pba_patch_attention(blk.attn, proj, self._pba_holder,
                                         tag=f's{s}b{bi}', monitor=(bi == 0))
            self.pba_projs = nn.ModuleList(_projs)
        else:
            self.pba_projs = None
            self._pba_holder = None

    def _encoder_forward_sk(self, x: torch.Tensor) -> list:
        """复刻 smp 0.4/0.5 的 `MixVisionTransformerEncoder.forward`，在每个 stage 的 NCHW 输出上
        插入 `_SKAdapter`；适配后的特征既进 features 也进下一个 stage。

        对应 smp v0.5.0 `encoders/mix_transformer.py` 的每个 stage：
            x = patch_embed{i}(x); x = block{i}(x); x = norm{i}(x).contiguous()
        返回值与 `self.encoder(x)` 同构：[x, dummy] + 4 个 stage。
        smp ≤0.3 的调用约定不同（patch_embed 返回 (x, H, W)，block 要传 H/W）。
        """
        enc = self.encoder
        feats = []
        h = x
        for i in range(1, 5):
            h = getattr(enc, f'patch_embed{i}')(h)
            h = getattr(enc, f'block{i}')(h)
            h = getattr(enc, f'norm{i}')(h).contiguous()
            key = str(i - 1)
            if key in self.sk_adapters:
                h = self.sk_adapters[key](h)
            feats.append(h)
        dummy = torch.empty([x.shape[0], 0, x.shape[2] // 2, x.shape[3] // 2],
                            dtype=x.dtype, device=x.device)
        return [x, dummy] + feats

    def pop_sk_stats(self):
        """取走并清零 SK 适配器的 epoch 级读数（tools/train.py 每 epoch 调一次）。
        关闭 SK 时返回 None；本 epoch 没有训练前向时返回空 dict。"""
        if self.sk_adapters is None:
            return None
        stats = {}
        for key, adapter in self.sk_adapters.items():
            acc = adapter._acc
            adapter._acc = None
            if acc is None or acc['n'] == 0:
                continue
            n = float(acc['n'])
            # 每 epoch 只在这里同步一次
            for i, v in enumerate((acc['w'] / n).tolist()):
                stats[f'SK s{key} branch{i} mean'] = float(v)
            stats[f'SK s{key} entropy'] = (acc['ent'] / n).item()
            stats[f'SK s{key} resid ratio'] = (acc['ratio'] / n).item()
        return stats

    def forward(self, x: torch.Tensor) -> dict:
        detail = self.detail_branch(x) if self.detail_branch is not None else None
        # PBA：调 encoder 前把全分辨率 pol 图放进 holder，调完立刻 pop，不跨 batch 残留。
        if self.pba_projs is not None:
            assert x.shape[1] == self._pba_in_ch, \
                f'输入通道数 {x.shape[1]} ≠ 构造时的 {self._pba_in_ch}——' \
                '"pol=最后一个通道"的前提失效（新加了派生通道？）'
            self._pba_holder['pol'] = x[:, -1:, :, :]
        try:
            # 开 SK 时走显式分段前向，否则直接调用 self.encoder。
            all_feats = (self._encoder_forward_sk(x) if self.sk_adapters is not None
                         else self.encoder(x))   # tuple/list of feature maps
        finally:
            if self.pba_projs is not None:
                self._pba_holder.pop('pol', None)
        # The last 4 entries are the 4 MiT stages.
        features = list(all_feats[-4:])
        # extra_pre 的各路来源（vario / fine_vario / context）都是零初始化投影，上采样到
        # features[0] 的分辨率后相加。
        vario = None
        if self.vario_texture is not None:
            tgt = features[0].shape[-2:]
            for mod, s in zip(self.vario_texture, self.vario_stages):
                v = mod(features[s])
                if v.shape[-2:] != tgt:
                    v = F.interpolate(v, size=tgt, mode='bilinear', align_corners=False)
                vario = v if vario is None else vario + v
        # fine_vario 吃原始输入，走同一个 extra_pre 注入口。
        if self.fine_vario is not None:
            fv = self.fine_vario(x)
            tgt = features[0].shape[-2:]
            if fv.shape[-2:] != tgt:
                fv = F.interpolate(fv, size=tgt, mode='bilinear', align_corners=False)
            vario = fv if vario is None else vario + fv.to(vario.dtype)
        # 上下文模块，同一个 extra_pre 注入口。
        if self.context_module is not None:
            c = self.context_module(features[self.context_stage])
            tgt = features[0].shape[-2:]
            if c.shape[-2:] != tgt:
                c = F.interpolate(c, size=tgt, mode='bilinear', align_corners=False)
            vario = c if vario is None else vario + c.to(vario.dtype)
        # 门控证据：vario 剖面在 stride-4 特征上算；pol 先验取输入的最后一个通道（__init__ 已
        # 断言 pol_ratio_channel=True 且 texture_channels=False）。
        gate_ev = None
        if self.gate_ev_vario is not None:
            assert x.shape[1] == self._gate_ev_in_ch, \
                f'输入通道数 {x.shape[1]} ≠ 构造时的 {self._gate_ev_in_ch}——' \
                '"pol=最后一个通道"的前提失效（新加了派生通道？）'
            ev_v = self.gate_ev_vario(features[0])
            ev_p = self.gate_ev_pol(x[:, -1:, :, :])
            if ev_p.shape[-2:] != ev_v.shape[-2:]:
                ev_p = F.interpolate(ev_p, size=ev_v.shape[-2:],
                                     mode='bilinear', align_corners=False)
            gate_ev = torch.cat([ev_v, ev_p.to(ev_v.dtype)], dim=1)
        out = {}
        for chart in self.charts:
            logits = self.heads[chart](features, extra=detail, extra_pre=vario,
                                       gate_evidence=gate_ev)
            if logits.shape[-2:] != x.shape[-2:]:
                logits = F.interpolate(logits, size=x.shape[-2:],
                                       mode='bilinear', align_corners=False)
            out[chart] = logits
        # 以下辅助输出只在训练时加入，eval 输出 dict 与基线一致。
        if self.supcon_proj is not None and self.training:
            out['_supcon'] = self.supcon_proj(features[0])
        if self.scale_reg_head is not None and self.training:
            out['_scale_reg'] = self.scale_reg_head(features[0])
        # GRL 夹在 encoder 特征与判别器之间；λ 由 tools/train.py 每 epoch 写入 self.dann_lambda。
        if self.dann_head is not None and self.training:
            out['_dann'] = self.dann_head(
                _GradReverse.apply(features[0], self.dann_lambda))
        return out
