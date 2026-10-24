"""Conformal calibration of quantile intervals for streaming multi-horizon forecasts.

All arrays are [N, O, K]: N series, O consecutive forecast origins, K horizons (EVAL_H).
A forecast at origin index j for horizon h (steps ahead, 1-based) only becomes scoreable at
origin index j + h, so every online update below respects that delay; no future information
leaks into an interval.
"""
import numpy as np

from data import EVAL_H


def scores(lo, hi, y):
    """CQR conformity score: how far y falls outside [lo, hi] (negative when inside)."""
    return np.maximum(lo - y, y - hi)


def _qlevel(alpha, n):
    return np.clip(np.ceil((n + 1) * (1 - alpha)) / n, 0.0, 1.0)


def _quantile_lastaxis(s, level):
    """Per-row empirical quantile of s [..., W] at per-row level [...] (higher interpolation)."""
    w = s.shape[-1]
    srt = np.sort(s, axis=-1)
    level = np.broadcast_to(level, s.shape[:-1])
    idx = np.clip(np.ceil(level * w).astype(int) - 1, 0, w - 1)
    return np.take_along_axis(srt, idx[..., None], axis=-1)[..., 0]


def _replace_sorted(srt, old, new):
    """Row-wise: remove one occurrence of `old` from sorted rows and insert `new`, keeping order.

    srt [..., W] sorted on the last axis; old, new [...]. Vectorised O(W) per row.
    """
    w = srt.shape[-1]
    j = np.arange(w - 1)
    p_old = (srt < old[..., None]).sum(-1)                 # first index equal to old
    deleted = np.take_along_axis(srt, j + (j >= p_old[..., None]), axis=-1)  # [..., W-1]
    p_new = (deleted < new[..., None]).sum(-1)
    jw = np.arange(w)
    src = np.where(jw < p_new[..., None], jw, jw - 1)
    out = np.take_along_axis(deleted, np.clip(src, 0, w - 2), axis=-1)
    return np.where(jw == p_new[..., None], new[..., None], out)


def calibrate(lo, hi, y, lo_c, hi_c, y_c, alpha=0.1, method="scqr", gamma=0.005, window=None):
    """Return calibrated (lo, hi) for the test arrays.

    method:
      raw      - model quantiles unchanged
      scqr     - split CQR (Romano et al. 2019): fixed offset from the calibration scores
      aci      - adaptive conformal inference (Gibbs & Candes 2021) on the calibration scores
      rcqr     - rolling-window CQR: calibration scores are replaced online by recent scores
      rw-aci   - proposed: rolling-window scores + ACI level update, horizon-delayed feedback
    Missing targets (NaN) are skipped in scores and updates.
    """
    if method == "raw":
        return lo.copy(), hi.copy(), np.full(lo.shape[::2], alpha)
    n, o, k = lo.shape
    s_cal = scores(lo_c, hi_c, y_c)                     # [N, Oc, K]
    with np.errstate(all="ignore"):
        # missing calibration targets: use the series' median score, or the pooled median
        # over all series when a series has no calibration data at all
        fill = np.nanmedian(s_cal, axis=1, keepdims=True)
        fill = np.where(np.isnan(fill), np.nanmedian(s_cal, axis=(0, 1), keepdims=True), fill)
    s_cal = np.where(np.isnan(s_cal), fill, s_cal)
    if method == "scqr":
        q = _quantile_lastaxis(s_cal.transpose(0, 2, 1), _qlevel(alpha, s_cal.shape[1]))
        return lo - q[:, None, :], hi + q[:, None, :], np.full((n, k), alpha)

    adaptive = method in ("aci", "rw-aci")
    rolling = method in ("rcqr", "rw-aci")
    w = window or s_cal.shape[1]
    buf = s_cal[:, -w:, :].transpose(0, 2, 1).copy()    # [N, K, W] ring buffer
    pos = np.zeros((n, k), dtype=int)
    a_t = np.full((n, k), alpha, dtype=float)
    s_test = scores(lo, hi, y)                           # [N, O, K]
    q_out = np.zeros((n, o, k))
    ar_n, ar_k = np.arange(n)[:, None], np.arange(k)[None, :]
    srt = np.sort(buf, axis=-1)
    feed_ok = np.array(EVAL_H)
    for j in range(o):
        # 1) feedback that has matured by origin j: for horizon h, the forecast made at j - h
        jj = j - feed_ok                                   # [K]
        live = jj >= 0
        if live.any():
            kk = np.flatnonzero(live)
            s_new = s_test[:, jj[kk], kk]                  # [N, K_live]
            ok = ~np.isnan(s_new)
            if adaptive:
                err = (s_new > q_out[:, jj[kk], kk]).astype(float)
                a_t[:, kk] += np.where(ok, gamma * (alpha - err), 0.0)
            if rolling:
                p = pos[:, kk]
                old = buf[ar_n, kk[None, :], p]
                new = np.where(ok, s_new, old)
                buf[ar_n, kk[None, :], p] = new
                pos[:, kk] = np.where(ok, (p + 1) % w, p)
                srt[:, kk, :] = _replace_sorted(srt[:, kk, :], old, new)
        # 2) interval for origin j
        lev = _qlevel(a_t, w)
        idx = np.clip(np.ceil(lev * w).astype(int) - 1, 0, w - 1)
        q_out[:, j, :] = srt[ar_n, ar_k, idx]
    return lo - q_out, hi + q_out, a_t
