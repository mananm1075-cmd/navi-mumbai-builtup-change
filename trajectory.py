"""Temporal trajectory analysis of built-up probability (reduces year-to-year noise).

For every pixel we have p_built(t) for t = 2017..2026 from the Random Forest.
A pixel is:
  * stable built-up     : mean p over all years >= 0.6 and no confident change
  * new built-up (year k): best step change at year k has mean-before <= 0.4,
                           mean-after >= 0.6 and a jump >= 0.35
                           (k = last year needs mean-before <= 0.3 and p >= 0.8;
                           k = second year needs p(2017) <= 0.25, mean-after >= 0.65)
  * otherwise           : not built-up (stable non-built or too uncertain)
Built-up area in year t = stable built-up + new built-up with change year <= t.
"""
import numpy as np

def analyse(P, years, lo=0.4, hi=0.6, jump=0.35):
    n = len(years)
    best_k = np.full(P.shape[1:], -1, np.int16)
    best_gain = np.zeros(P.shape[1:], np.float32)
    best_before = np.zeros(P.shape[1:], np.float32)
    best_after = np.zeros(P.shape[1:], np.float32)
    cs = np.cumsum(P, axis=0)
    tot = cs[-1]
    for k in range(1, n):
        before = cs[k - 1] / k
        after = (tot - cs[k - 1]) / (n - k)
        if k == n - 1:                      # only one year after the change
            ok = (before <= 0.3) & (after >= 0.8)
        elif k == 1:                        # only one year before the change
            ok = (before <= 0.25) & (after >= 0.65) & ((after - before) >= 0.45)
        else:
            ok = (before <= lo) & (after >= hi) & ((after - before) >= jump)
        gain = np.where(ok, after - before, 0)
        upd = gain > best_gain
        best_gain[upd], best_k[upd] = gain[upd], k
        best_before[upd], best_after[upd] = before[upd], after[upd]
    new = best_k >= 0
    stable = (~new) & (P.mean(0) >= hi)
    change_year = np.where(new, np.array(years)[np.clip(best_k, 0, n - 1)], 0)
    area = [float((stable | (new & (change_year <= y))).sum() * 1e-4) for y in years]
    return {"stable": stable, "new": new, "change_year": change_year, "area_km2": area}
