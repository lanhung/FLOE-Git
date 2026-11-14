"""Ice-chart composition supervision for floe size.

An ice-chart polygon does not say which pixel is which floe size; its egg code gives
the total concentration CT and up to three partial concentrations (CA, CB, CC), each
with a form of ice (FA, FB, FC). The usual label keeps only the dominant form. Here
each polygon becomes a composition over the 6 FLOE classes (water = 1 - CT), and the
model is trained so that the mean prediction over the pixels of a polygon matches it:

  L_prop  cross-entropy between the chart composition and the polygon-mean prediction
          (learning from label proportions, Ardehaly & Culotta 2017);
  L_ord   squared distance between the cumulative floe-size distributions of the two,
          over the ordered ice classes cake < small < medium < big < vast;
  pure polygons (one class >= pure_threshold) keep the ordinary pixel cross-entropy.

Concentration that cannot be mapped to a class (undetermined form, pancake, fast ice,
bergs) is "unknown" mass: it is assigned to the ice classes in proportion to the
current prediction, so it neither rewards nor penalises any ice class.

Polygon ids reach the loss as a label channel ('_bag', global id per training polygon,
-1 = none); see AI4ArcticChallengeDataset and tools/build/make_scene.py.
"""
import numpy as np
import torch
import torch.nn as nn

N_CLASSES = 6          # water, cake, small, medium, big, vast
ICE = slice(1, 6)      # ordered ice classes
BAG_OFFSET = 256       # bag ids travel as label values >= 256 (255 = affine fill / ignore)

# SIGRID-3 form of ice (FA/FB/FC) -> FLOE class; None = not one of the 6 classes.
# Same mapping as FLOE_LOOKUP in configs/_base_/base.py.
FORM_TO_CLASS = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 5}


def sigrid_concentration(code):
    """SIGRID-3 concentration code -> fraction in [0, 1], or NaN if missing/invalid.

    00 ice free, 01 open water (<1/10), 02 bergy water, 10..90 tenths, 91 = 9+/10,
    92 = 10/10, two-digit ranges such as 13 (1-3/10) or 46 -> midpoint.
    """
    if code is None:
        return np.nan
    c = int(code)
    if c in (0, 2):           # ice free, bergy water
        return 0.0
    if c == 1:                # open water, < 1/10 ice
        return 0.05
    if c == 91:
        return 0.95
    if c == 92:
        return 1.0
    if c % 10 == 0 and 10 <= c <= 90:
        return c / 100.0
    lo, hi = divmod(c, 10)
    if 1 <= lo < hi <= 9:
        return (lo + hi) / 20.0
    if lo == 9 and hi == 1:
        return 0.95
    return np.nan


def egg_to_composition(ct, partials):
    """Composition over the 6 FLOE classes from decoded concentrations.

    ct        total concentration in [0, 1].
    partials  list of (concentration, form_code); concentration may be NaN when the
              chart has a single ice type (then it is CT).
    Returns (q, unknown): q[6] sums to 1 - unknown; unknown is ice of unmapped form.
    """
    q = np.zeros(N_CLASSES, dtype=np.float64)
    if not np.isfinite(ct):
        return q, 1.0
    ct = float(np.clip(ct, 0.0, 1.0))
    q[0] = 1.0 - ct
    parts = [(c, f) for c, f in partials if f is not None and int(f) not in (-9, 99)]
    if len(parts) == 1 and not np.isfinite(parts[0][0]):
        parts = [(ct, parts[0][1])]
    known = 0.0
    for c, f in parts:
        if not np.isfinite(c) or c <= 0:
            continue
        k = FORM_TO_CLASS.get(int(f))
        if k is not None:
            q[k] += c
            known += c
    if known > ct and known > 0:          # partials that overshoot CT are rescaled
        q[ICE] *= ct / known
        known = ct
    unknown = ct - known
    return q, unknown if unknown > 1e-6 else 0.0


class CompositionLoss(nn.Module):
    """Polygon-level proportion and ordinal losses on softmax predictions.

    Q: [P, 6] chart compositions, U: [P] unknown ice fraction, for all training
    polygons (indexed by the global bag id).
    """

    def __init__(self, Q, U, pure_threshold=0.9, w_prop=1.0, w_ord=1.0,
                 min_pixels=64, full_weight_pixels=4096, max_unknown=0.5, ice_only=False):
        super().__init__()
        Q = torch.as_tensor(Q, dtype=torch.float32)
        U = torch.as_tensor(U, dtype=torch.float32)
        self.register_buffer('Q', Q)
        self.register_buffer('U', U)
        top, arg = Q.max(dim=1)
        pure = (top >= pure_threshold) & (U <= 1 - pure_threshold)
        self.register_buffer('pure_class', torch.where(pure, arg, torch.full_like(arg, -1)))
        self.w_prop, self.w_ord = w_prop, w_ord
        # ice_only: match only the floe-size mix of the ice, not the open-water share
        # (1 - CT). The water share of a 9/10 polygon says nothing about where the leads
        # are, and fitting it teaches the model to put water inside pack ice; water vs ice
        # is then left to the pixel CE (use with hard_ce='all').
        self.ice_only = ice_only
        self.min_pixels, self.full_weight_pixels = min_pixels, full_weight_pixels
        # polygons that are mostly unknown ice (fast ice, bergs, undetermined form) would
        # only feed the model its own guess back, so they are left out
        self.register_buffer('usable', U <= max_unknown)

    def pure_only(self, y, bag):
        """Labels with every pixel outside a pure polygon set to 255."""
        pc = torch.full_like(y, -1)
        has = bag >= 0
        pc[has] = self.pure_class[bag[has]]
        return torch.where(pc >= 0, y, torch.full_like(y, 255))

    def forward(self, logits, bag):
        """logits [B, 6, H, W], bag [B, H, W] (long, -1 = no polygon). Returns (loss, stats)."""
        has = bag >= 0
        if not bool(has.any()):
            zero = logits.sum() * 0.0
            return zero, {'bags': 0}
        p = torch.softmax(logits.float(), dim=1).permute(0, 2, 3, 1)[has]   # [N, 6]
        ids, inv, cnt = torch.unique(bag[has], return_inverse=True, return_counts=True)
        S = torch.zeros(len(ids), p.shape[1], device=p.device).index_add_(0, inv, p)
        keep = (cnt >= self.min_pixels) & self.usable[ids]
        if not bool(keep.any()):
            zero = logits.sum() * 0.0
            return zero, {'bags': 0}
        ids, cnt, pbar = ids[keep], cnt[keep].float(), S[keep] / cnt[keep, None].float()
        q, u = self.Q[ids], self.U[ids]

        # unknown ice mass follows the current prediction over the ice classes
        with torch.no_grad():
            ice_share = pbar[:, ICE] / pbar[:, ICE].sum(1, keepdim=True).clamp_min(1e-6)
        target = q.clone()
        target[:, ICE] += u[:, None] * ice_share
        target = target / target.sum(1, keepdim=True).clamp_min(1e-6)

        w = (cnt / self.full_weight_pixels).clamp(max=1.0)
        w = w / w.sum()

        # ice-conditional distributions (floe-size mix of the ice alone)
        ice_t = target[:, ICE]
        ice_frac = ice_t.sum(1)
        cond_t = ice_t / ice_frac[:, None].clamp_min(1e-6)
        ice_p = pbar[:, ICE]
        cond_p = ice_p / ice_p.sum(1, keepdim=True).clamp_min(1e-6)
        if self.ice_only:
            l_prop = -(cond_t * torch.log(cond_p.clamp_min(1e-6))).sum(1) * ice_frac
        else:
            l_prop = -(target * torch.log(pbar.clamp_min(1e-6))).sum(1)

        # ordinal term on the ice-conditional distributions, weighted by ice fraction
        cdf_t = torch.cumsum(cond_t, 1)[:, :-1]
        cdf_p = torch.cumsum(cond_p, 1)[:, :-1]
        l_ord = ((cdf_p - cdf_t) ** 2).sum(1) * ice_frac

        loss = (w * (self.w_prop * l_prop + self.w_ord * l_ord)).sum()
        stats = {'bags': int(len(ids)), 'l_prop': float((w * l_prop).sum()),
                 'l_ord': float((w * l_ord).sum()),
                 'tv': float((w * 0.5 * (pbar - target).abs().sum(1)).sum())}
        return loss, stats


# ---------------------------------------------------------------- evaluation

def polygon_histograms(pred, poly, n_classes=N_CLASSES, valid=None):
    """Per-polygon histogram of predicted classes. pred, poly: int arrays of one scene
    (poly = -1 outside polygons). Returns (ids, hist[n, n_classes])."""
    m = poly >= 0
    if valid is not None:
        m &= valid
    m &= (pred >= 0) & (pred < n_classes)
    if not m.any():
        return np.zeros(0, np.int64), np.zeros((0, n_classes), np.int64)
    ids, inv = np.unique(poly[m], return_inverse=True)
    hist = np.bincount(inv * n_classes + pred[m].astype(np.int64),
                       minlength=len(ids) * n_classes).reshape(len(ids), n_classes)
    return ids, hist


def composition_metrics(hist, q, u, pure_threshold=0.9, min_pixels=64):
    """Polygon-level scores of predicted class histograms against chart compositions.

    Returns a dict with pixel-weighted and polygon-mean total variation (TV, half L1
    between predicted and chart composition, unknown mass excluded by renormalising
    both over the known classes), ice-CDF EMD, and dominant-class agreement on pure and
    mixed polygons.
    """
    n = hist.sum(1)
    sel = (n >= min_pixels) & (u < 0.5)
    hist, q, u, n = hist[sel], q[sel], u[sel], n[sel]
    if len(n) == 0:
        return {'polygons': 0}
    p = hist / n[:, None]
    qk = q / q.sum(1, keepdims=True).clip(1e-9)
    tv = 0.5 * np.abs(p - qk).sum(1)
    pi, qi = p[:, ICE], qk[:, ICE]
    has_ice = qi.sum(1) > 0.05
    cp = np.cumsum(pi / pi.sum(1, keepdims=True).clip(1e-9), 1)[:, :-1]
    cq = np.cumsum(qi / qi.sum(1, keepdims=True).clip(1e-9), 1)[:, :-1]
    emd = np.abs(cp - cq).sum(1)
    pure = qk.max(1) >= pure_threshold
    agree = p.argmax(1) == qk.argmax(1)
    out = {
        'polygons': int(len(n)), 'pixels': int(n.sum()),
        'tv_pix': float((tv * n).sum() / n.sum()), 'tv_poly': float(tv.mean()),
        'emd_ice_poly': float(emd[has_ice].mean()) if has_ice.any() else float('nan'),
        'dominant_agree_pure': float(agree[pure].mean()) if pure.any() else float('nan'),
        'dominant_agree_mixed': float(agree[~pure].mean()) if (~pure).any() else float('nan'),
        'n_pure': int(pure.sum()), 'n_mixed': int((~pure).sum()),
    }
    return out


def _coupling(t, h):
    """Most favourable confusion matrix between two class histograms per polygon.

    t, h: [P, K] pixel counts by class (chart, prediction) with equal row sums. The
    chart does not place the classes inside a polygon, so min(t, h) pixels of each class
    are taken as agreeing and the surplus on either side is paired in proportion.
    Returns the [K, K] matrix summed over polygons (rows chart, columns prediction).
    """
    same = np.minimum(t, h)
    rt, rh = t - same, h - same
    r = rt.sum(1).clip(1e-12)
    cm = np.einsum('pi,pj->ij', rt / r[:, None], rh)
    cm[np.diag_indices_from(cm)] += same.sum(0)
    return cm


def composition_confusion(hist, q, u, pure_threshold=0.9, min_pixels=64, max_unknown=0.5):
    """Confusion matrices of predicted polygon histograms against chart compositions.

    'full'  all 6 classes, water share of ice polygons included (most favourable
            arrangement inside each polygon, see _coupling); on a pure polygon this is
            the ordinary pixel confusion matrix.
    'ice'   floe-size mix of the ice only: prediction and chart are both renormalised
            over the 5 ice classes and weighted by the chart's ice pixels; [5, 6], the
            last column holds chart ice in polygons where no ice was predicted.
    'pure'  ordinary pixel confusion matrix on polygons with one class >= pure_threshold.
    Polygons smaller than min_pixels or with more than max_unknown unknown mass are left out.
    """
    n = hist.sum(1).astype(np.float64)
    sel = (n >= min_pixels) & (u <= max_unknown) & (q.sum(1) > 0)
    hist, q, n = hist[sel].astype(np.float64), q[sel].astype(np.float64), n[sel]
    q = q / q.sum(1, keepdims=True)
    t = q * n[:, None]
    full = _coupling(t, hist)

    n_ice = t[:, ICE].sum(1)
    pred_ice = hist[:, ICE].sum(1)
    both = (n_ice > 0) & (pred_ice > 0)
    ti = t[both][:, ICE]
    hi = hist[both][:, ICE] / pred_ice[both, None] * n_ice[both, None]
    ice = np.zeros((5, 6))
    ice[:, :5] = _coupling(ti, hi)
    ice[:, 5] = t[(n_ice > 0) & (pred_ice == 0)][:, ICE].sum(0)

    is_pure = q.max(1) >= pure_threshold
    pure = np.zeros((N_CLASSES, N_CLASSES))
    np.add.at(pure, q[is_pure].argmax(1), hist[is_pure])
    return {'full': full, 'ice': ice, 'pure': pure, 'polygons': int(sel.sum()), 'pure_polygons': int(is_pure.sum())}


def confusion_scores(cm):
    """Per-class IoU and F1, their means over classes present in the reference (rows),
    and the reference-weighted F1 (the AutoICE floe score). Extra columns (e.g. the
    'no ice predicted' column of the ice-only matrix) count as misses."""
    k = cm.shape[0]
    tp = np.diag(cm[:, :k])
    ref, pred = cm.sum(1), cm[:, :k].sum(0)
    iou = tp / np.maximum(ref + pred - tp, 1e-12)
    f1 = 2 * tp / np.maximum(ref + pred, 1e-12)
    present = ref > 0
    return {'iou': iou, 'f1': f1, 'miou': float(iou[present].mean()),
            'f1_weighted': float((ref * f1).sum() / max(ref.sum(), 1e-12)),
            'oa': float(tp.sum() / max(cm.sum(), 1e-12))}
