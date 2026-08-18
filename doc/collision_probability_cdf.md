# Collision Probability under Random Obstacle Inflation

Replaces the ad-hoc Gaussian halo in [risk_map.py](../map/risk_map.py#L109-L120) with an
actual collision probability, derived from an explicit model of obstacle-boundary error.

**The model.** A single obstacle's true extent is its nominal (ESDF-reported) shape
inflated outward by a random amount ε ≥ 0. The sensor may miss material; it never invents
material. For a query point at nominal signed distance `d`:

```
P(collision | d) = P(ε ≥ d) = erfc( d / (σ√2) )        d > 0
                            = 1                         d ≤ 0
```

with ε = σ|Z|, Z ~ N(0,1) — a half-normal of **scale** σ. Read §4 before setting σ: the
scale is *not* the standard deviation of the inflation.

---

## 1. Why this is legitimate

Three things have to hold. The first is the load-bearing one, and it is exact.

### 1.1 The geometric event reduces exactly to a scalar comparison

The true obstacle is `O_ε` = the nominal obstacle `O` dilated by ε. A point `x` collides
iff `x ∈ O_ε`. That is a 2-D geometric event about an arbitrarily-shaped region — but it
collapses, without approximation, to a comparison of two numbers.

For any closed `O` and any `x` outside `O_ε`:

```
dist(x, O_ε) = dist(x, O) − ε
```

(`≥` by the triangle inequality; `≤` by walking from `x` toward its nearest point in `O`
and stopping ε short, which lands on `∂O_ε`.) Therefore

```
x ∈ O_ε   ⟺   d − ε ≤ 0   ⟺   ε ≥ d
```

**This is the whole justification.** The messy random-set question "is my point inside this
randomly-grown blob?" is *identical* to the one-dimensional question "did the blob grow by
at least d?" — no linearisation, no small-ε assumption, no dependence on obstacle shape or
curvature.

*Verified:* on a union of 7 random discs with analytic distances and ε = 0.37 m, the
exterior residual of the identity is **4.4e-16 m** (machine precision). On a raster the
residual is exactly **1 pixel** at every resolution from 0.10 down to 0.0125 m — that is
thresholding a discrete field, and it shrinks with the grid.

### 1.2 P is therefore a survival function, hence a genuine probability

`P(d) = P(ε ≥ d) = S_ε(d)`. For any nonnegative random variable, its survival function is
automatically a valid probability: it is monotone non-increasing, `S(d) = 1` for `d ≤ 0`,
and `S(d) → 0` as `d → ∞`. Nothing is normalised by hand, nothing is clipped, and the value
at every d is the fraction of possible obstacles that swallow the point.

#### This is the survival function, NOT the CDF — they run in opposite directions

Easy to trip on. The half-normal's **CDF** rises from 0 to 1; the risk field falls from 1 to
0. Both are correct, because they are complements of each other:

```
  d/σ |  CDF  F(d) = P(ε ≤ d) |  SF  S(d) = P(ε ≥ d) | F + S
      |  erf( d/(σ√2) )       |  erfc( d/(σ√2) )     |
 -----+-----------------------+----------------------+------
    0 |       0.00000         |       1.00000        |  1
    1 |       0.68269         |       0.31731        |  1
    2 |       0.95450         |       0.04550        |  1
    3 |       0.99730         |       0.00270        |  1
```

(`erf` / `erfc` — the "c" is literally *complementary*. Matches `scipy.stats.halfnorm.cdf`
and `.sf` exactly.)

**"Survival" here refers to ε, not to the robot.** `S_ε` is standard probability
terminology for the upper tail of a random variable — the inflation "surviving" past the
threshold d. It does **not** mean the robot survived. The inflation reaching past you *is*
the robot being hit, so:

```
value returned  =  P(ε ≥ d)  =  P(COLLISION)          ← this is a failure probability
1 − value       =  P(safe / no collision)
```

The check is the boundary: the field returns **1.0** at `d = 0`. That can only be collision
probability — standing on the obstacle you are certainly hit, and your safety there is 0.
This is why it feeds [`obstacle_risk`](../map/risk_map.py#L119) directly, and why the
combination step is written `keep = Π(1 − riskᵢ)` — `keep` is the safety probability, the
layers are failure probabilities.

**Why the upper tail is the right one.** The argument `d` is not "how much did the obstacle
inflate" — it is *your distance from the boundary*. You are hit when the inflation **reaches
out past you**, i.e. when ε is **large**. Large-ε events live in the upper tail, so the
probability you want is `P(ε ≥ d)`, which necessarily decreases as you move away.

The direction is the sanity check, not a red flag. Using the CDF instead would claim:

- standing on the obstacle boundary (`d = 0`) → risk **0.000**, perfectly safe
- three sigma clear (`d = 3σ`) → risk **0.997**, near-certain collision

i.e. collision probability *rising* with clearance. The falling curve is the correct one.

### 1.3 The interior value of 1 is derived, not imposed

Because ε ≥ 0 always, the event `ε ≥ d` is certain whenever `d ≤ 0`. So `P = 1` inside the
nominal obstacle is a *consequence* of "the sensor only under-reports", not a safety
override bolted on. And the two branches agree in the limit:

```
lim(d→0⁺) erfc( d/(σ√2) ) = erfc(0) = 1
```

so P is **continuous** at the boundary. Contrast the earlier draft, which paired a
symmetric-error CDF with a `d ≤ 0` clamp: those disagree at the boundary (0.5 vs 1.0) and
the field jumps. That combination is gone.

---

## 2. Deriving the formula

`P(ε ≥ d)` with ε = σ|Z| and d > 0:

```
P(σ|Z| ≥ d) = P(|Z| ≥ d/σ) = P(Z ≥ d/σ) + P(Z ≤ −d/σ) = 2·P(Z ≥ d/σ) = 2(1 − Φ(d/σ))
```

The two normal tails are equal by symmetry, so both-tails = 2 × one-tail. **This is exactly
the fact that a two-tailed p-value is twice the one-tailed one** — same identity, and the
reason the numbers below are the familiar two-sided z-values.

Verified symbolically: `P(|Z| ≥ t) − 2·P(Z ≥ t)` simplifies to **0**, not to something
small. In closed form `2(1 − Φ(d/σ)) = erfc(d/(σ√2))`, which is the form to implement.

**Picture it as folding.** Start from a symmetric Gaussian boundary error. Fold the
"obstacle turned out *smaller*" half over onto the "obstacle turned out *bigger*" side. All
that probability mass is reassigned to growth, so every outward distance receives exactly
double — and standing on the nominal boundary, *every* possible obstacle reaches you, which
is why P(0) = 1.

The doubling applies **only for d ≥ 0**. There `2(1−Φ) ≤ 1` automatically, reaching exactly
1 at d = 0; nothing is ever clipped. For d < 0 the doubling would exceed 1 and is
meaningless — which is precisely the region the `d ≤ 0` branch covers.

---

## 3. Confirmed by simulation

Draw ε, count the fraction of worlds in which the point is swallowed (4M samples, σ = 0.30):

```
  d/σ  |   MC      formula
 ------+------------------
  −1.0 | 1.00000   1.00000
   0.0 | 1.00000   1.00000
  +0.5 | 0.61676   0.61708
  +1.0 | 0.31668   0.31731
  +2.0 | 0.04545   0.04550
  +3.0 | 0.00268   0.00270
```

---

## 4. σ is a *scale*, not the standard deviation

The single easiest way to misuse this. For ε = σ|Z|:

```
mean(ε)   = σ√(2/π)     = 0.7979 σ
std(ε)    = σ√(1−2/π)   = 0.6028 σ      ← not σ
median(ε) =               0.6745 σ
```

σ is the std of the *underlying normal before folding*, not the spread of the realised
inflation. Two consistent ways to set it:

| you have | set |
|---|---|
| the std of the pre-fold boundary error (what `obstacle_sigma_min/max` currently mean in [ObstacleSigmaField](../map/sdf.py#L144-L199)) | `σ` directly |
| a measured standard deviation `s` of actual inflation amounts | `σ = s / 0.6028 = 1.659 s` |

Plugging a measured inflation-std straight in as σ gives a halo ~40% narrower than
intended, plus an unrequested 0.80σ mean bias. Decide which you mean and write it down at
the call site.

**One parameter, two jobs.** σ fixes the mean and the spread together — you cannot set them
independently, since `mean/std = 1.323` always. If your measured underestimation is
"5 cm ± 2 cm", no half-normal matches both (matching the mean forces std = 3.8 cm; matching
the std forces mean = 2.7 cm). §7 adds a floor Δ if you need both.

---

## 5. Numbers

```
   d      P(collision)
 ──────────────────────
  0.0σ      1.00000
  0.5σ      0.61708
  1.0σ      0.31731
  1.5σ      0.13361
  2.0σ      0.04550
  2.5σ      0.01242
  3.0σ      0.00270
```

**As a chance constraint.** Inverting `erfc(d/(σ√2)) ≤ δ` gives the minimum clearance for a
risk budget δ — and it comes out as exactly the two-tailed z-value, as §2 predicts:

| risk budget δ | required clearance |
|---|---|
| 0.5   | 0.6745 σ |
| 0.1   | 1.6449 σ |
| 0.05  | **1.9600 σ** |
| 0.01  | 2.5758 σ |
| 0.001 | 3.2905 σ |

So "keep collision probability under 5%" is literally "stay 1.96σ clear" — the risk field
and a hard geometric margin are the same constraint written two ways.

For comparison, the exponential currently in the code gives 0.607 at one sigma against this
model's 0.317 — nearly 2×, and that ratio is arbitrary rather than derived, because
`exp(−d²/2σ²)` is the *shape of a density* evaluated where a *tail probability* belongs.

---

## 6. What you may and may not multiply

This is where a per-point risk field is most often misused, and the model gives a sharp
answer in both directions.

### Along a trajectory: do NOT multiply — take the minimum

One obstacle has **one** ε. Every point near it shares that same draw, so the per-point
probabilities are perfectly correlated. The trajectory collides iff *some* point is
swallowed:

```
∃t : ε ≥ d_t   ⟺   ε ≥ min_t d_t
```

so the trajectory's collision probability is the point formula evaluated at the **minimum
clearance** — exact, no independence assumption:

```
P(trajectory hits obstacle k) = erfc( min_t d_t^(k) / (σ_k √2) )
```

*Verified:* a 40-point path skimming a disc, σ = 0.25, `d_min` = 0.3510 →
MC **0.16031** vs formula **0.16035**. Treating the 40 points as independent gives
**0.821** — wrong by 5×.

This is exactly why [`ensemble_worst_clearance`](../map/risk_sdf.py#L178-L188) reduces with
`min` over the time axis. That reduction is the correct one, not a conservative shortcut.

### Across obstacles: DO multiply

Different obstacles draw independent ε_k ([`sample_per_obstacle`](../map/sdf.py#L155-L168)
draws one per instance), so the product is valid:

```
P(any collision) = 1 − Π_k [ 1 − erfc( d_min^(k) / (σ_k √2) ) ]
```

*Verified:* 3 obstacles, MC **0.18562** vs formula **0.18578**.

The practical consequence: the probabilistic-OR in
[`compute`](../map/risk_map.py#L122-L126) is legitimate *across obstacles*, and the
per-cell risk map is a valid **marginal** field, but summing or multiplying it along a path
is not a collision probability.

---

## 7. Optional: a guaranteed minimum inflation Δ

If you know the obstacle is *always* under-reported by at least some amount — a systematic
ESDF bias rather than pure noise — put a floor on the inflation:

```
ε = Δ + σ|Z|
```

"inflated by at least Δ, plus a random half-normal amount." Then

```
P(d) = 1                              d ≤ Δ
     = erfc( (d − Δ) / (σ√2) )        d > Δ
```

Still exact, still continuous (both branches give 1 at d = Δ), and Δ = 0 recovers the bare
model at the top of this document. This
restores the ability to match a measured mean *and* spread, which a bare half-normal cannot
(§4).

**Δ is a radial offset — half the diameter deficit.** The easiest place to be off by 2×:

```
true obstacle:   20 cm diameter  →  radius 10 cm
ESDF reports:    10 cm diameter  →  radius  5 cm

diameter deficit = 10 cm
Δ = (D_true − D_obs)/2 = 5 cm = 0.05 m       ← use this, not 0.10
```

The SDF measures distance to the *boundary*, not across the obstacle. Using the diameter
deficit inflates every obstacle twice as much as intended.

Δ need not be a global scalar — the shrinkage from slicing a 3-D field depends on obstacle
shape and slice height. Calibrate it by measuring reported-vs-true widths on known objects,
and if the spread across obstacles is large, broadcast a per-obstacle Δ exactly as σ
already is ([ObstacleSigmaField](../map/sdf.py#L144-L199)).

---

## 8. Implementation

### 8.1 You need a genuinely signed SDF — and you must know its units

**Simulation path (synthetic maps).**
[`SignedDistanceField.compute`](../map/sdf.py#L103-L114) **clamps the interior to zero**:

```python
sdf = sdf_pos - sdf_neg
sdf[sdf < 0] = 0.0        # obstacles read as 0, not negative
```

With that field `d < 0` never occurs and the `d ≤ Δ` branch degenerates. Use
[`_true_signed_sdf`](../src/environment.py#L51-L56), which is what
[`Environment`](../src/environment.py#L142) already builds and passes in as `sdf=sdf_px`.
A `TerrainRiskMap` constructed standalone (`sdf=None`) silently gets the clamped one and
does **not** behave the same. This field is in **pixels**.

**Hardware path (nvblox ESDF).** A true ESDF is already signed (negative inside) and
already in **metres**, so it needs neither `_true_signed_sdf` nor the pixel conversion.

⚠ [risk_map.py:118](../map/risk_map.py#L118) hardcodes `d_m = self.sdf * self.res`. Feeding
a metric ESDF through that multiplies every distance by the resolution and **fails
silently** — no crash, no NaN, just saturation:

| true clearance | after a stray ×0.05 | P reported |
|---|---|---|
| 0.30 m | 0.015 m | 1.000 |
| 2.00 m | 0.100 m | 0.739 |

The whole map reads as solid obstacle. Make the units explicit rather than implicit:

```python
    def __init__(self, grid_map, sdf=None, sdf_units="px", ...):
        if sdf_units not in ("px", "m"):
            raise ValueError(f"sdf_units must be 'px' or 'm', got {sdf_units!r}")
        self.sdf_units = sdf_units

    # in compute():
        d_m = self.sdf.astype(np.float64)
        if self.sdf_units == "px":
            d_m = d_m * self.res
```

`sdf_units="px"` keeps every existing call site working; the nvblox path passes
`sdf_units="m"`.

**Unobserved cells are not free space.** An ESDF built from real sensor data has regions
that were never seen. nvblox reports those at the truncation distance, which this formula
reads as *far from any obstacle* — i.e. **safe** — when the correct statement is *unknown*.
Carry the validity mask alongside the ESDF and decide explicitly: either mark unobserved
cells as high risk, or exclude them from the planning region. Do not let them default to 0.

(ESDF **truncation** itself is harmless here: nvblox caps at ~2 m, and with σ of a few
centimetres P is already ~1e-40 well before that, so the far field is 0 either way.)

### 8.2 Use `erfc` directly

`erfc(x/√2)` *is* `2(1 − Φ(x))` — one call, and it keeps precision in the far tail where
`1 - norm.cdf(x)` underflows to exactly 0 past x ≈ 8. The branch is required because
`erfc` of a negative argument exceeds 1 (it ranges to 2), which is the doubling breaking
down exactly as §2 describes.

```python
from scipy.special import erfc

arg = (d_m - self.underestimation) / (self.obs_sigma * np.sqrt(2.0))
self.obstacle_risk = np.where(arg <= 0.0, 1.0, erfc(arg))
```

### 8.3 The layer weight forfeits the probability reading

`w_obs * P` is not a probability: with `w_obs = 0.5` a point deep inside an obstacle reads
0.5. Keep `w_obs = 1` for the obstacle layer if the output is to be called a collision
probability — which is the default in [`Environment`](../src/environment.py#L89)
(`risk_weights=(0.0, 0.0, 1.0)`). Treat `w_obs ≠ 1` as an explicit heuristic reweighting
and say so wherever the number is reported.

### 8.4 The change in `risk_map.py`

Replace [lines 118-120](../map/risk_map.py#L118-L120):

```python
        d_m = self.sdf.astype(np.float64) * self.res        # SDF is in pixels
        self.obstacle_risk = self.w_obs * np.where(
            d_m <= 0, 1.0, np.exp(-0.5 * (d_m / self.obs_sigma) ** 2))
```

with:

```python
        d_m = self.sdf.astype(np.float64)
        if self.sdf_units == "px":                          # 8.1 -- nvblox is already m
            d_m = d_m * self.res
        # P(collision) = P(eps >= d) for eps = Delta + sigma*|Z| :
        #   obstacle inflated outward by a random nonnegative amount.
        # obs_sigma is the half-normal SCALE, not std(eps) = 0.6028*scale.
        arg = (d_m - self.underestimation) / (self.obs_sigma * np.sqrt(2.0))
        self.obstacle_risk = self.w_obs * np.where(arg <= 0.0, 1.0, erfc(arg))
```

`from scipy.special import erfc` at module top, plus the constructor arguments:

```python
    def __init__(self, grid_map, sdf=None,
                 ...,
                 sdf_units="px",             # "px" (synthetic) or "m" (nvblox ESDF)
                 underestimation_m=0.0):     # radial floor, = (D_true - D_obs)/2
        ...
        self.sdf_units = sdf_units
        self.underestimation = float(underestimation_m)
```

The defaults (`sdf_units="px"`, `underestimation_m=0.0`) keep every existing call site
working. Downstream combination is unchanged. Call sites:

```python
# simulation
risk_map = TerrainRiskMap(
    grid_map,
    sdf=true_signed_sdf_px,     # 8.1 -- not SignedDistanceField.compute
    sdf_units="px",
    underestimation_m=0.0,      # synthetic obstacles are exact -- no bias to correct
    obstacle_sigma_min=0.03,    # 4   -- SCALE of the half-normal, not std(eps)
    obstacle_sigma_max=0.08,
)

# hardware: nvblox ESDF slice, already signed and already in metres
risk_map = TerrainRiskMap(
    grid_map,
    sdf=esdf_slice_m,           # negative inside, metres
    sdf_units="m",              # 8.1 -- do NOT let it be rescaled by resolution
    underestimation_m=0.05,     # 7   -- radial, half the measured diameter deficit
    obstacle_sigma_min=0.03,
    obstacle_sigma_max=0.08,
)
```

### 8.5 Tests worth writing

- monotone: `P(d₁) ≥ P(d₂)` whenever `d₁ ≤ d₂`
- `P(d) = 1` for all `d ≤ Δ`; `P → 0` as `d → ∞`
- continuity: `P(Δ⁺) → 1` — no jump at the boundary
- `P(Δ + σ) ≈ 0.31731`, `P(Δ + 1.96σ) ≈ 0.05`
- against Monte Carlo: `mean(σ|Z| ≥ d)` over ≥1e6 draws, to ~3 decimals
- trajectory: `P(path) == P(d_min)`, and that it is *not* the independent-OR product

---

## 9. Assumptions and limits

- **Isotropic inflation.** One scalar ε grows the whole obstacle uniformly along its normal.
  This is what makes §1.1 exact. If the true error varies *around* a single obstacle —
  plausible for a slice through an irregular rock — a point could be swallowed by a locally
  large bulge while the obstacle-wide ε is smaller than d, and the formula understates.
  Modelling that needs a random field on the boundary, not a scalar.
- **The half-normal is a choice.** §1.1–1.2 hold for *any* nonnegative ε; the half-normal is
  one pick. It puts its mode at ε = 0 ("most likely the ESDF was right"). If your measured
  underestimation clusters around a nonzero value, the floor Δ of §7 fits better, and a
  truncated normal or lognormal ε would fit better still — swap only `S_ε`, everything else
  stands.
- **σ is prescribed, not measured.** It is currently drawn per obstacle from
  `uniform(sigma_min, sigma_max)`. The numbers are only as good as that prior; calibrating
  σ against measured boundary error is what turns this from "a probability under an assumed
  model" into "the collision probability".
- **Layer fusion is not a joint probability.** The probabilistic-OR with slope and roughness
  in [`compute`](../map/risk_map.py#L122-L126) assumes independence across hazards that are
  correlated on real terrain. It is a bounded monotone surrogate; only the obstacle layer
  alone carries the interpretation derived here.
- **Point robot.** `d` is the distance for a point. For a disc robot of radius `r`, use
  `d − r`, which is exact for the same reason as §1.1 (dilating the obstacle by `r`).

---

## References

- Half-normal distribution: <https://en.wikipedia.org/wiki/Half-normal_distribution>
- Relation `erfc(x/√2) = 2(1 − Φ(x))`: <https://en.wikipedia.org/wiki/Error_function>
- CVaR inflation of an SDF — a related but different scheme already in this repo, which
  inflates by a fixed `κ·s_k` margin rather than integrating over ε:
  [risk_sdf.py](../map/risk_sdf.py#L39-L56)
