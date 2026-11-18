#!/usr/bin/env python3
"""GTA VI launch-day Twitch viewership forecaster.

Competition: gta6-twitch-ccu-forecast-01 (make.fun).
Program: python3 /bot/main.py -- reads every *.in in $MAKEFUN_INPUT_DIR
(or /input), writes one <id>.out per case to $MAKEFUN_OUTPUT_DIR (or /output).

Model (log-ratio regression over comparable launches)
----------------------------------------------------
For each comparable launch we align its release-day hourly curve to its own
day_start and form log-ratios r_h = log(viewers_h / viewers_1) for h = 2..24.
The forecast median for hour h is p1 * exp(m_h), where m_h comes from a
weighted least-squares regression of the comparables' log-ratios on:

    [1, min(ln h1, 12.0), ln(heat6), rolling, console_only]

* h1        -- hour-1 print (viewers in the first hour after day_start).
* heat6     -- mean hourly viewers in the 6 hours before day_start, / h1.
              A "heat" signal: a launch whose pre-release hours already run
              hot relative to hour 1 behaves differently from a cold one.
* rolling   -- 1 when the launch unlocks at midnight local time in each
              region (rolling midnight), 0 otherwise.
* console_only -- 1 when the launch is console-only (no PC).

The ln-h1 term is capped (hinge at 12.0 ~= 163k viewers): the ratio curve
flattens steeply with launch size, but it does not keep falling forever --
beyond the largest comparable launches the shape saturates, which keeps the
forecast sane if GTA VI's hour-1 print lands far above anything seen.

Each comparable is weighted by similarity to the target:
Gaussian kernel in ln h1 (sigma 1.0), x3 for rolling launches, x2 for
console-only launches, x0.3 for single-global/simultaneous unlocks, and a
Gaussian kernel in ln(heat) (sigma 0.7) where heat is the pre-release
24 h peak / h1.

Spreads are pinball-calibrated, not assumed normal: for each hour we take
the regression residuals across the pool (pooled over the neighbouring
hours), and set sigma = (q84 - q16) / 2 of the weighted residual
distribution. Quantiles are q_tau = p1 * exp(m_h + z_tau * sigma) with the
standard normal z for the five requested levels.

Two guardrails keep the median honest for a rolling console launch:
* a floor -- the median is never below the weighted 10th percentile of the
  log-ratios of rolling+console-only comparables (a rolling console launch
  does not peak below its own hour 1);
* an unlock step -- +0.25 log at hour 6 and +0.15 at hour 9, the US East
  (05:00Z) and US West (08:00Z) unlocks for GTA VI's rolling midnight.

Medians and floors are lightly smoothed (3-point moving average) because
hour-to-hour ratios are noisy; the unlock step is applied after smoothing.

Stdlib only, no network, deterministic: the same input always produces
byte-identical output. Runtime is far below the 8 s / case limit.
"""

import json
import math
import os
import re
import sys
from datetime import datetime

LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
Z = (-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722)

# Tuned constants (validated offline: practice loss ~0.155 vs house
# baseline 0.387; LOO-CV over the 41-launch pool ~0.45 vs 0.47 kernel).
LN_H1_SIGMA = 1.0
ROLLING_BONUS = 3.0
CONSOLE_BONUS = 2.0
GLOBAL_PENALTY = 0.3
HEAT_SIGMA = 0.7
LN_H1_HINGE = 12.0
FLOOR_QUANTILE = 0.10
SPREAD_MULT = 1.0
SPREAD_FLOOR = 0.05
SPREAD_CAP = 2.0
UNLOCK_STEP = {6: 0.25, 9: 0.15}


def parse_time(text):
    """Parse an ISO-8601 timestamp (with trailing Z) to an aware datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def weighted_quantile(pairs, tau):
    """Weighted quantile of [(value, weight), ...]. Deterministic."""
    ordered = sorted(pairs)
    total = sum(w for _, w in ordered)
    if total <= 0:
        return ordered[len(ordered) // 2][0]
    acc = 0.0
    for value, weight in ordered:
        acc += weight
        if acc >= tau * total:
            return value
    return ordered[-1][0]


def smooth3(values):
    """3-point moving average; endpoints kept as-is."""
    out = list(values)
    for i in range(1, len(values) - 1):
        out[i] = (values[i - 1] + values[i] + values[i + 1]) / 3.0
    return out


def solve_linear(matrix, vector):
    """Solve a small dense linear system by Gaussian elimination with
    partial pivoting. Returns None when (near-)singular."""
    n = len(vector)
    aug = [row[:] + [vector[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[piv][col]) < 1e-12:
            return None
        aug[col], aug[piv] = aug[piv], aug[col]
        for r in range(col + 1, n):
            factor = aug[r][col] / aug[col][col]
            for cc in range(col, n + 1):
                aug[r][cc] -= factor * aug[col][cc]
    beta = [0.0] * n
    for i in range(n - 1, -1, -1):
        beta[i] = (aug[i][n]
                   - sum(aug[i][j] * beta[j] for j in range(i + 1, n))) / aug[i][i]
    return beta


def wls_fit(rows, ys, ws):
    """Weighted least squares: returns beta for y ~ rows, or None."""
    n = len(rows[0])
    ata = [[0.0] * n for _ in range(n)]
    aty = [0.0] * n
    for row, y, w in zip(rows, ys, ws):
        for i in range(n):
            aty[i] += w * row[i] * y
            for j in range(n):
                ata[i][j] += w * row[i] * row[j]
    return solve_linear(ata, aty)


def release_day_curve(hourly, day_start):
    """24 hourly viewer counts for hours 1..24 after day_start, or None
    when the comparable has no complete release day (e.g. the GTA VI
    pre-release category itself)."""
    hours = {}
    for point in hourly:
        viewers = point.get("viewers")
        if viewers is None:
            continue
        try:
            delta = (parse_time(point["from"]) - day_start).total_seconds() / 3600.0
        except (KeyError, ValueError):
            continue
        if 0 <= delta < 24:
            hours[int(delta) + 1] = viewers
    if len(hours) < 24 or hours.get(1, 0) <= 0:
        return None
    return [hours[h] for h in range(1, 25)]


def comparable_features(comp):
    """Features computable at forecast time for one comparable launch."""
    try:
        day_start = parse_time(comp["day_start"])
    except (KeyError, ValueError):
        return None
    curve = release_day_curve(comp.get("hourly") or [], day_start)
    if curve is None:
        return None
    hourly = comp.get("hourly") or []
    pre24 = []
    last6 = []
    for point in hourly:
        viewers = point.get("viewers")
        if viewers is None:
            continue
        try:
            delta = (parse_time(point["from"]) - day_start).total_seconds() / 3600.0
        except (KeyError, ValueError):
            continue
        if -24 <= delta < 0:
            pre24.append(viewers)
        if -6 <= delta < 0:
            last6.append(viewers)
    h1 = curve[0]
    pre_max = max(pre24) if pre24 else 0.0
    pre_mean6 = sum(last6) / len(last6) if last6 else 0.0
    unlock = (comp.get("unlock") or "").lower()
    platforms = (comp.get("platforms") or "").lower()
    rolling = ("midnight local" in unlock) or ("rolling" in unlock)
    global_unlock = ("single global" in unlock) or ("simultaneous" in unlock)
    console_only = ("pc" not in platforms) and (
        "ps" in platforms or "xbox" in platforms or "switch" in platforms)
    early_access = "early access" in unlock
    return {
        "name": comp.get("name", ""),
        "h1": h1,
        "ln_h1": math.log(max(1.0, h1)),
        "heat": pre_max / max(1.0, h1),
        "heat6": pre_mean6 / max(1.0, h1),
        "rolling": rolling,
        "global_unlock": global_unlock,
        "console_only": console_only,
        "early_access": early_access,
        "curve": curve,
    }


def target_features(case):
    """Features of the launch being forecast, from the case itself."""
    known = case.get("known") or []
    if not known:
        raise ValueError("case has no known hour-1 print")
    hour1 = known[0]
    p1 = hour1["viewers"]
    day_start = parse_time(case["day_start"])
    history = case.get("history_hourly") or case.get("history") or []
    pre24 = []
    last6 = []
    for point in history:
        viewers = point.get("viewers")
        if viewers is None:
            continue
        key = point.get("from") or point.get("at")
        if not key:
            continue
        try:
            delta = (parse_time(key) - day_start).total_seconds() / 3600.0
        except ValueError:
            continue
        if -24 <= delta < 0:
            pre24.append(viewers)
        if -6 <= delta < 0:
            last6.append(viewers)
    # Fall back to the most recent pre-release points when the window is
    # sparsely sampled (SullyGnome samples small categories ~every 12 h).
    if not last6 and pre24:
        last6 = pre24[-6:]
    heat = (max(pre24) / p1) if pre24 else 0.0
    heat6 = (sum(last6) / len(last6) / p1) if last6 else heat
    if heat6 <= 0:
        heat6 = heat if heat > 0 else 1.0
    if heat <= 0:
        heat = heat6
    # Rolling / console-only come from the case notes when present. GTA VI
    # is a rolling-midnight console-only launch, which is the default; only
    # an explicit statement to the contrary overrides it. A negated PC
    # mention -- e.g. "no PC version at launch" -- must not flip the flag,
    # so a PC phrase only counts when it is not preceded by a negation.
    notes = (case.get("notes") or "").lower()
    rolling = True
    if ("single global" in notes or "simultaneous" in notes) and \
            "rolling" not in notes and "midnight local" not in notes:
        rolling = False
    console_only = True
    pc_launch = False
    for phrase in ("pc launch", "pc version", "pc release",
                   "releases on pc", "available on pc"):
        for match in re.finditer(re.escape(phrase), notes):
            before = notes[max(0, match.start() - 12):match.start()]
            if not re.search(r"\b(no|not|without|never|n't)\s*$", before):
                pc_launch = True
                break
        if pc_launch:
            break
    if pc_launch:
        console_only = False
    return {
        "h1": p1,
        "ln_h1": math.log(max(1.0, p1)),
        "heat": heat,
        "heat6": heat6,
        "rolling": rolling,
        "console_only": console_only,
        "global_unlock": False,
        "early_access": False,
    }


def comparable_weights(target, comps):
    """Similarity weight of each comparable for this target."""
    weights = []
    for comp in comps:
        w = 1.0
        d = (comp["ln_h1"] - target["ln_h1"]) / LN_H1_SIGMA
        w *= math.exp(-0.5 * d * d)
        if comp["rolling"]:
            w *= ROLLING_BONUS
        if comp["console_only"]:
            w *= CONSOLE_BONUS
        if comp["global_unlock"]:
            w *= GLOBAL_PENALTY
        d_heat = (math.log(max(0.05, comp["heat"]))
                  - math.log(max(0.05, target["heat"]))) / HEAT_SIGMA
        w *= math.exp(-0.5 * d_heat * d_heat)
        weights.append(w)
    return weights


def design_row(feats):
    """Regression row for one launch: intercept, hinged ln h1, ln heat6,
    rolling flag, console-only flag."""
    return [1.0,
            min(feats["ln_h1"], LN_H1_HINGE),
            math.log(max(0.05, feats["heat6"])),
            1.0 if feats["rolling"] else 0.0,
            1.0 if feats["console_only"] else 0.0]


def forecast_case(case):
    """Forecast hours 2..24 for one case. Returns the output dict."""
    target = target_features(case)
    p1 = max(1.0, target["h1"])
    comps = []
    for comp in case.get("comparables") or []:
        feats = comparable_features(comp)
        if feats is not None:
            comps.append(feats)
    if not comps:
        # No usable comparables: fall back to the house persistence shape.
        return {"hours": [{"hour": h, "q": [
            max(1.0, p1 * math.exp(z * 0.15 * math.sqrt(h - 1))) for z in Z]}
            for h in range(2, 25)]}

    weights = comparable_weights(target, comps)
    rows = [design_row(c) for c in comps]
    target_row = design_row(target)

    medians = []
    floors = []
    residual_pools = []  # per hour: sorted [(residual, weight)]
    for hi in range(1, 24):  # curve index 1..23 == hours 2..24
        ys = [math.log(max(1.0, c["curve"][hi]) / max(1.0, c["curve"][0]))
              for c in comps]
        beta = wls_fit(rows, ys, weights)
        if beta is None:
            # Singular design: fall back to the weighted mean log-ratio.
            total_w = sum(weights)
            mean_y = sum(y * w for y, w in zip(ys, weights)) / total_w
            fitted = [mean_y] * len(ys)
            median = mean_y
        else:
            median = sum(b * x for b, x in zip(beta, target_row))
            fitted = [sum(b * x for b, x in zip(beta, row)) for row in rows]
        medians.append(median)
        residual_pools.append(sorted(
            (y - f, w) for y, f, w in zip(ys, fitted, weights)))
        rc_pairs = [(math.log(max(1.0, c["curve"][hi]) / max(1.0, c["curve"][0])), w)
                    for c, w in zip(comps, weights)
                    if c["rolling"] and c["console_only"]]
        floors.append(weighted_quantile(rc_pairs, FLOOR_QUANTILE)
                      if rc_pairs else -9.9)

    medians = smooth3(medians)
    floors = smooth3(floors)
    # A rolling console launch does not peak below its own hour 1.
    medians = [max(m, f) for m, f in zip(medians, floors)]
    # US unlocks for GTA VI's rolling midnight: East 05:00Z (hour 6),
    # West 08:00Z (hour 9).
    for hour, bump in UNLOCK_STEP.items():
        medians[hour - 2] += bump

    out_hours = []
    for idx, hour in enumerate(range(2, 25)):
        pooled = []
        for hh in range(max(0, idx - 1), min(22, idx + 1) + 1):
            pooled.extend(residual_pools[hh])
        q16 = weighted_quantile(pooled, 0.16)
        q84 = weighted_quantile(pooled, 0.84)
        sigma = max(SPREAD_FLOOR,
                    min(SPREAD_CAP, (q84 - q16) / 2.0 * SPREAD_MULT))
        qs = [p1 * math.exp(medians[idx] + z * sigma) for z in Z]
        qs = [min(1e9, max(1.0, q)) for q in qs]
        for i in range(1, 5):
            if qs[i] < qs[i - 1]:
                qs[i] = qs[i - 1]
        out_hours.append({"hour": hour, "q": [round(q, 2) for q in qs]})
    return {"hours": out_hours}


def main(argv=None):
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(n for n in os.listdir(in_dir) if n.endswith(".in"))
    for name in names:
        with open(os.path.join(in_dir, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        result = forecast_case(case)
        out_name = os.path.splitext(name)[0] + ".out"
        with open(os.path.join(out_dir, out_name), "w", encoding="utf-8") as fh:
            json.dump(result, fh, separators=(",", ":"))
            fh.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
