# GTA VI launch-day Twitch viewership forecaster

Entry for make.fun competition `gta6-twitch-ccu-forecast-01` — forecast GTA VI's
launch-day Twitch viewers hour by hour (hours 2–24), as five quantiles
(0.05 / 0.25 / 0.5 / 0.75 / 0.95) per hour. Scored by mean pinball loss on the
log scale against the true hourly prints.

## How it works

The program reads every `*.in` case in `$MAKEFUN_INPUT_DIR` (default `/input`)
and writes one `<id>.out` per case to `$MAKEFUN_OUTPUT_DIR` (default
`/output`). Stdlib only, no network, deterministic.

For each comparable past launch it aligns the release-day hourly curve to
that launch's own `day_start` and forms log-ratios
`r_h = log(viewers_h / viewers_1)` for hours 2–24. The forecast median for
hour `h` is `p1 * exp(m_h)`, where `p1` is the case's hour-1 print and `m_h`
comes from a **weighted least-squares regression** of the comparables'
log-ratios on:

```
[1, min(ln h1, 12.0), ln(heat6), rolling, console_only]
```

- `h1` — hour-1 print. Bigger launches have flatter curves (the peak
  log-ratio falls from ~+7 at ln h1 ≈ 5 to ~0 at ln h1 ≈ 13.7), so launch
  size is the strongest single predictor.
- `heat6` — mean hourly viewers in the 6 hours before `day_start`, divided
  by `h1`. A launch whose pre-release hours already run hot relative to
  hour 1 behaves differently from a cold one.
- `rolling` — 1 when the launch unlocks at midnight local time in each
  region; `console_only` — 1 when it is console-only (no PC).

The `ln h1` term is **hinged at 12.0** (≈163k viewers): the curve flattens
with size, but it does not keep falling forever — beyond the largest
comparable launches the shape saturates, which keeps the forecast sane if
GTA VI's hour-1 print lands far above anything seen before.

Each comparable is weighted by similarity to the target: a Gaussian kernel
in `ln h1` (σ = 1.0), ×3 for rolling launches, ×2 for console-only
launches, ×0.3 for single-global/simultaneous unlocks, and a Gaussian kernel
in `ln(heat)` (σ = 0.7), where `heat` is the pre-release 24 h peak / `h1`.

**Spreads are pinball-calibrated, not assumed normal.** For each hour the
spread is `(q84 − q16) / 2` of the weighted distribution of the regression
residuals across the pool (pooled over the neighbouring hours). Quantiles
are `q_τ = p1 * exp(m_h + z_τ · σ)` with the standard normal `z` for the
five requested levels.

**Guardrails for a rolling console launch** (GTA VI unlocks at rolling
midnight, console-only):

- a **floor** — the median is never below the weighted 10th percentile of
  the log-ratios of rolling + console-only comparables (a rolling console
  launch does not peak below its own hour 1);
- an **unlock step** — +0.25 log at hour 6 and +0.15 at hour 9, the US East
  (05:00Z) and US West (08:00Z) unlocks.

Medians and floors are lightly smoothed (3-point moving average); the
unlock step is applied after smoothing. Quantiles are clamped to
`[1, 1e9]` and forced non-decreasing. If a case has no usable comparable
(a release-day curve needs all 24 hours), the program falls back to the
house persistence shape.

## Offline validation

Practice answers for the four practice launches (Wolverine, Red Dead
Redemption 2, God of War Ragnarök, Spider-Man 2) were read once from the
SullyGnome API named in the kit's NOTICE.txt (targeted reads only), and the
entry was scored with the same pinball loss the grader uses.

| Model | Practice loss (4 cases) | LOO-CV (41 launches) |
|---|---|---|
| House persistence baseline | 0.387 | 0.877 |
| Weighted-median log-ratios | 0.310 | 0.81 |
| This entry (regression + residual spread + floor + unlock step) | **0.158** | **0.443** |

Practice loss per case with the final pool (37 comparables + the other
3 practice launches, mimicking the final's 42-comparable pool):
Wolverine 0.192, RDR2 0.155, GoW 0.107, Spider-Man 2 0.179. Leave-one-out
over the 41-launch pool: 0.443 overall, 0.210 for big launches
(h1 ≥ 50k). Runtime ≈ 0.16 s per case; two runs are byte-identical.

## Forecast shape (practice cases, median as a multiple of hour 1)

- Wolverine (h1 23,357): hour 2 ≈ 0.73×, hour 6 ≈ 1.36×, hour 19 ≈ 4.61×,
  hour 24 ≈ 4.11×.
- RDR2 (h1 116,550): 0.77× / 1.30× / 2.24× / 1.95×.
- GoW Ragnarök (h1 269,517): 0.76× / 1.05× / 1.77× / 1.60×.
- Spider-Man 2 (h1 119,087): 0.77× / 1.26× / 2.12× / 1.84×.

Common shape: a dip in hours 2–4, a step up at the US unlocks, a midday
trough, then an evening ramp peaking around hours 19–20, with hour 24
still elevated.

## Known limitations

- The median is a regression over 37–42 past launches; a launch whose
  shape is unlike anything in the pool (e.g. a much deeper midday trough,
  as Spider-Man 2 had) is only covered by the spread, not predicted.
- The hour-1 print is the dominant signal. If GTA VI's hour-1 print is far
  outside the comparable range, the hinge + floor keep the shape bounded,
  but the exact evening multiple is an extrapolation.
- The model uses only the case file: hour-1 print, pre-release history,
  and the comparable pool. It does not use `streams` / `categories_hourly`
  (null in the practice cases) or the GTA VI pre-release category's own
  long hourly history beyond the heat signals.
