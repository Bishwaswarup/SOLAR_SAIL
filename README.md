<div align="center">

# ☀️ Solar Sail CR3BP

### Dissolution of the Collinear Structure at Finite Sail Lightness Number

*When does a solar sail stop having a Lagrange point? A computational study in the Sun–Earth and Earth–Moon circular restricted three-body problems.*

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://python.org)
[![NumPy](https://img.shields.io/badge/NumPy-Scientific-013243?logo=numpy)](https://numpy.org)
[![SciPy](https://img.shields.io/badge/SciPy-ODE%20Solver-8CAAE6?logo=scipy)](https://scipy.org)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-Figures-11557c)](https://matplotlib.org)
[![IISc](https://img.shields.io/badge/IISc-Bangalore-E04B00)](https://iisc.ac.in)

**Bishwaswarup** · Indian Institute of Science, Bangalore · independent project

</div>

---

## The Result

A face-on solar sail does not *displace* the collinear Lagrange point. It **dissolves** it.

Because a face-on sail (α = 0) produces a purely radial, conservative force, it simply rescales solar gravity, (1−μ) → (1−β)(1−μ). The on-axis equilibrium is therefore the root of a Kepler-like balance, and with the Earth **deleted entirely** it sits at

$$x_{\rm hover}(\beta) = \left[(1-\beta)(1-\mu)\right]^{1/3}$$

This is a *heliocentric hovering point* — the radius where reduced solar gravity balances centrifugal acceleration at the synchronous rate. It is not a perturbation of L₁.

The whole content of the study is the rate at which the collinear structure degenerates into this hovering point as β grows. Defining the **saddle strength**

$$s(\beta) = \mu / r_2^3, \qquad A(\beta) = \tfrac{(1-\beta)(1-\mu)}{r_1^{3}} + \tfrac{\mu}{r_2^{3}} = 1 + s$$

(a real hyperbolic pair requires A > 1, so *all* hyperbolicity is of Earth origin), we find two thresholds:

| Criterion | Condition | β_crit | Standoff |
|---|---|---|---|
| **Hill-sphere exit** | r₂ = r_H = (μ/3)^(1/3) | **2.98 × 10⁻⁴** | 1.00 r_H |
| **Tidal parity** | s = 1, i.e. r₂ = μ^(1/3) | **0.02865** | 3^(1/3) r_H = 1.442 r_H |

The tidal-parity threshold has an exact closed form. Imposing r₂ = μ^(1/3) on the on-axis balance cancels the (1−μ) factor identically, leaving

**β_crit = 1 − (1 − μ^(1/3))² = μ^(1/3)(2 − μ^(1/3))**

exactly — no expansion in μ, no root-finding. It therefore depends on the system *only* through μ^(1/3). For Sun–Earth it lands at **β ≈ 0.028646**.

Where that sits against real hardware is a checkable question, and the answer is **not** "already achievable". Reduced from primary specifications ([`src/sail_technology.py`](src/sail_technology.py)), every solar sail ever flown sits at β ≤ 0.0061:

| sail | β | note |
|---|---|---|
| IKAROS (2010) | 0.00062 | the only **measured** value — from JAXA's 1.12 mN |
| ACS3 (2024) | 0.0048 | 80 m², 16 kg |
| LightSail-2 (2019) | 0.0061 | 32 m², 5 kg — best flown |
| Solar Cruiser | 0.0202 | design only, cancelled 2022 |

So tidal parity is **≈ 3× beyond the best flown sail** and **within 40 % of the most ambitious funded design**. The defensible claim is that it lies at the edge of *near-term* capability — which is stronger than the unsourced version precisely because it can be checked.

<div align="center"><img src="fig8_structure_dissolution.png" width="95%"/></div>

*Panel (a): the full three-body equilibrium is indistinguishable from the Earth-free hovering law; the inset shows the Earth's entire contribution collapsing to a few thousand km. (b) the standoff crosses the Hill sphere almost immediately and reaches 20.6 r_H at β = 0.5. (c) the saddle strength collapses four orders of magnitude. (d) the in-plane and out-of-plane frequencies both converge on the mean motion — the Keplerian epicyclic degeneracy.*

---

## What β = 0.5 Actually Is

The β = 0.5 case, run in `main.py`, is **past** both thresholds by a wide margin. Its results are artifacts of the hovering geometry, not dynamical findings:

| Reported | Reality |
|---|---|
| "L₁ displaced sunward by 29.4 M km" | A heliocentric hovering point at 0.7937 AU. The Earth's **entire** contribution is 2.55 × 10⁻⁵ nd = **3,817 km**, against a 30.9 M km standoff — **20.6 Hill radii**, where the Earth is dynamically irrelevant. |
| "One-year period — resonance with Earth" | **Forced, not resonant.** In a 1/r² field the epicyclic frequency equals the mean motion identically. As A → 1 the planar roots become λ² = {−1, 0} and ν = √A → 1, so the period is pinned to exactly 2π nd. Verified: ω = 1.00042656, ν = 1.00021348. |
| "λ_u = 1.25 — the sail stabilised the orbit" | **No saddle was tamed; the orbit moved away from one.** The linear saddle exponent at the point is λ = 0.035781 nd, and exp(0.035781 × 6.281845) = **1.25203** — reproducing the reported monodromy eigenvalue 1.252011 to five figures. The residual is *entirely* the Earth's leftover tug: s(0.5) = 3.42 × 10⁻⁴. |
| "L₁→L₂ transfer for 0.05 m/s" | **Self-matching.** `main.py` builds *both* manifolds from the same `state0_class`. W^u and W^s of one orbit both contain that orbit, so the matcher returns its self-intersection: the two "matched" states are the same point (separation 0.000 km), each 137.9 km from the halo — the ε = 150 km seed perturbation. There is no second orbit and no transfer. |

Note that the saddle never strictly vanishes: A = 1 + s with s > 0 always, so there is no bifurcation — only a smooth, four-decade degeneration. Reporting the *collapse rate* is both more honest and more interesting than claiming a stability transition.

---

## Earth–Moon Heteroclinic Connections

The earlier claim — matched Jacobi constants at Az = 0.02 for both halos, ΔC = 5.7 × 10⁻⁵ — does not survive. Two independent problems:

**1. Equal Az does not give equal C.** The L₁ and L₂ families have different energy-vs-amplitude slopes. There is no reason for them to coincide, and they do not.

**2. The corrector was branch-hopping.** `compute_halo_orbit`'s free variables are `[x₀, vy₀, T_half]` with constraints `[vx_f, vz_f, y_f] = 0`. **z₀ is never a free variable and Az is never a constraint** — Az only seeds the Richardson guess. Calling it independently at each Az lands on different branches:

```
 Az     C(L2)  unseeded        C(L2)  continued        T
0.005   3.17207607             3.15200518          3.41532
0.010   3.15127516  ← jump     3.15167618          3.41471
0.020   3.17087868  ← jump     3.14870567          3.40910
0.040   3.13757644  ← jump     3.13757644          3.38692
```

The apparent match at Az = 0.02 was a **spurious orbit** (T = 3.5195 against the true family's 3.4091). Corrected:

| At Az = 0.02 | old | corrected |
|---|---|---|
| C_L1 | 3.17093543 | 3.17093543 |
| C_L2 | 3.17087868 | **3.14870567** |
| **ΔC** | 5.7 × 10⁻⁵ | **2.2 × 10⁻²** |

`src/jacobi_match.py` fixes this with natural-parameter continuation (each solution seeds the next, with z₀ overwritten to step Az explicitly) plus a period-continuity branch guard. Both families then track smoothly and monotonically, and **genuine energy-matched pairs require very unequal amplitudes**:

```
  C_target      Az_L1      Az_L2    ratio      T_L1      T_L2
------------------------------------------------------------------
3.14165258    0.06446    0.04949      1.3   2.76710   3.39527
3.14369698    0.06225    0.04427      1.4   2.76574   3.39936
3.14574137    0.05997    0.03842      1.6   2.76436   3.40338
3.14778577    0.05763    0.03158      1.8   2.76295   3.40734
3.14983017    0.05522    0.02288      2.4   2.76153   3.41124
3.15187456    0.05272    0.00740      7.1   2.76008   3.41508
```

L₁ family C ∈ [3.13128, 3.17349] over Az ∈ [0.010, 0.075]; L₂ family C ∈ [3.14144, 3.15209] over Az ∈ [0.0025, 0.050]. **Overlap: C ∈ [3.14144, 3.15209].** The most balanced pair is C ≈ 3.1417 with Az_L1 = 0.0645, Az_L2 = 0.0495.

### Does the halo family terminate? No — a checked negative result

The z-extrema asymmetry

$$\delta = \frac{|z_0| - |z(T/2)|}{|z_0| + |z(T/2)|}$$

is O(0.1) for a genuine halo and exactly 0 on the vertical-Lyapunov branch (which satisfies z(t + T/2) = −z(t)). It is what exposed the bogus L₂ orbit. Swept at **fixed** Az = 0.003 it appeared to fall off a cliff between β = 0.05 and 0.10 — 1.3×10⁻¹ to ~10⁻¹³ with nothing between — which looked like a symmetry-breaking bifurcation terminating the halo family.

It is an artifact. δ is a *nonlinear* quantity, vanishing identically for a linearised orbit, and holding Az fixed while the equilibrium's own length scale γ = (1−μ) − x_eq grows from 0.00997 to 0.0353 shrinks Az/γ four-fold, pushing the orbit into the linear regime. The continuation was also branch-hopping (period jumping 3.260 → 2.634 across β = 0.005 → 0.010). Holding Az/γ fixed instead, δ stays near 0.13 through β = 0.03.

Two tests reject the bifurcation:

- Fitting δ ~ (β_c − β)^p gives **p = 1.17 and 1.70** in two independent sweeps. A pitchfork requires **p = 0.5**.
- δ crosses zero **transversally** near β ≈ 0.107 and grows again with opposite sign (+2.0×10⁻³ at β = 0.110 rising to +1.38×10⁻² at β = 0.140). A bifurcation would end the branch there, not continue through it.

So the loss of halo character with β is the same smooth death of the Earth's tidal term already measured by s = μ/r₂³ — seen in the orbit family rather than the linearisation, not a separate dynamical event.

**Caveat, stated plainly:** the natural-parameter continuation here is *not* robust — sweeps started at different β land on different branches. The two conclusions above survive every sweep tried, but the fine structure does not, so `halo_asymmetry.py` ships no figure. A publishable statement needs pseudo-arclength continuation.

### On the 544 m/s figure

A genuine heteroclinic connection costs **zero ΔV by construction** — the unstable manifold of one orbit *is* the stable manifold of the other. 544 m/s with a 57 km residual is therefore not a heteroclinic connection and must not be described as "low-cost ballistic-like." It is at best a **manifold-guided two-impulse transfer**.

Worse, it was computed at Az = 0.02 for both halos, so its L₂ endpoint was the spurious vertical-Lyapunov orbit — the transfer was matching an L₁ *halo* against an L₂ *vertical Lyapunov orbit*. The two figures have been **deleted** rather than shipped with a caveat, and must be regenerated at a matched pair from the table above, using the self-intersection guard now available as `match_manifolds(..., exclude_states=..., min_sep=...)`.

---

## Sail Force Model

The implementation in `src/dynamics.py` is a correct **ideal specular reflector**:

$$\mathbf{a}_{\rm sail} = \beta \frac{1-\mu}{r_1^{2}} \cos^{2}\!\alpha \; \hat{\mathbf{n}}, \qquad \hat{\mathbf{n}} = \cos\alpha\,\hat{\mathbf{r}} + \sin\alpha\cos\delta\,\hat{\mathbf{t}} + \sin\alpha\sin\delta\,\hat{\mathbf{k}}$$

Force purely along the sail normal, magnitude ∝ cos²α. An earlier draft of this README wrote the bracketed form `cos α · [cos α n̂ + sin α t̂]`, which is the **perfect absorber** (force along the sun-line, magnitude ∝ cos α). The two agree only at α = 0. The code was always right; the documentation was wrong. Figures 4 and 5, which depend on off-axis behaviour, are unaffected.

| Parameter | Symbol | Meaning |
|---|---|---|
| Cone angle | α | Sun-line to sail normal. α = 0 → face-on (max thrust); α = π/2 → edge-on |
| Clock angle | δ | Azimuth of the normal about the sun-line |
| Lightness number | β | SRP force / solar gravity. β ≤ 0.006 flown, 0.02 designed (see `sail_technology.py`); β = 0.5 far-future |

Membrane billow, finite slew rate and self-shadowing are not modelled; see Dachwald et al. on why non-ideal optics matter more than that estimate suggests.

---

## Figures

<div align="center">

![Beta sweep animation](beta_sweep_animation.gif)

![Manifold deployment animation](manifold_deployment.gif)

</div>

**Figure 1 — β family of orbits.** The computation is sound; the interpretation is the dissolution above, not a family of displaced halos.

<div align="center"><img src="fig1_beta_family.png" width="85%"/></div>

**Figure 2 — Eigenvalue map.** The unstable eigenvalue collapses toward unity because the standoff grows and the Earth's tidal term dies, not because the sail stabilises a saddle.

<div align="center"><img src="fig2_stability.png" width="75%"/></div>

**Figure 3 — Floquet exponents.** Out-of-plane mode decouples and stays neutrally stable; both frequencies converge on the mean motion.

<div align="center"><img src="fig3_floquet.png" width="75%"/></div>

**Figure 5 — Sail control authority.** Built on the true 6×2 sail Jacobian ∂**a**/∂(α, δ), replacing the earlier LQR suite that was computed on a 6×3 unconstrained thruster and therefore described a spacecraft a sail cannot be.

The result: **α₀ = 0 is a singular nominal for sail station-keeping.** At face-on, ∂**a**/∂δ vanishes identically — the clock angle only rotates a normal already parallel to **r̂** — so the sail is a one-input system whose single direction is fixed by a constant δ. The Kalman rank is 4/6: the in-plane pair is reachable (Coriolis carries a transverse input into the radial direction, rank 4/4), but the decoupled vertical mode is not. Any α₀ > 0 restores rank 6.

Authority peaks at **α₀\* = arctan(1/√2) = 35.264°**, where σ₂/σ₁ = 1/3 and cos²α₀\* = 2/3, both exactly. That angle is the classical optimal sail cone angle (McInnes 1999 §2.6) — recovering it from the control Jacobian is a benchmark check, not a new result; what is new is that it also maximises out-of-plane control authority, and that the authority vanishes at α₀ = 0.

<div align="center"><img src="fig5_control_authority.png" width="88%"/></div>

`fig5_minimum_beta.png` and `fig5_simulation.png` have been **deleted** — both were LQR results on the fictitious thruster. `fig5_station_keeping.png` (the sensitivity-matrix corrector) survives: it perturbs the actual sail angles and is genuine sail control.

**Figures 6–7 — Earth–Moon, out of scope.** Regenerated at a properly energy-matched pair with the self-intersection guard, but the transfer ΔV does not converge with manifold sampling (1891 → 2637 → 1929 → 388 m/s), so no ΔV is quoted. Reachable via `python main.py earthmoon`; excluded from the paper set.

---

## Positioning Against the Literature

Artificial equilibria, sail halo orbits and sail station-keeping are all well-established. This work must be positioned against, not presented as independent of:

1. **McInnes, C. R.** (1999). *Solar Sailing: Technology, Dynamics and Mission Applications*. Springer–Praxis. — The standard reference.
2. **McInnes, C. R., McDonald, A. J., Simmons, J. F. L., MacDonald, E. W.** (1994). "Solar sail parking in restricted three-body systems." *J. Guidance, Control, and Dynamics* **17**(2), 399–406. [doi:10.2514/3.21211](https://arc.aiaa.org/doi/10.2514/3.21211) — Surfaces of artificial equilibria; the direct antecedent of the equilibrium calculation here.
3. **Baoyin, H. & McInnes, C. R.** (2006). "Solar sail halo orbits at the Sun–Earth artificial L₁ point." *Celestial Mechanics and Dynamical Astronomy* **94**(2), 155–171. [doi:10.1007/s10569-005-4626-3](https://link.springer.com/article/10.1007/s10569-005-4626-3) — Precisely the orbits of Figures 1–3.
4. **Waters, T. J. & McInnes, C. R.** (2007). "Periodic orbits above the ecliptic in the solar-sail restricted three-body problem." *J. Guidance, Control, and Dynamics* **30**(3), 687–693. [doi:10.2514/1.26232](https://arc.aiaa.org/doi/10.2514/1.26232)
5. **Farrés, A. & Jorba, À.** (2010). "Periodic and quasi-periodic motions of a solar sail close to SL₁ in the Earth–Sun system." *Celestial Mechanics and Dynamical Astronomy* **107**, 233–253. [doi:10.1007/s10569-010-9268-4](https://link.springer.com/article/10.1007/s10569-010-9268-4) — Station-keeping at SL₁; must be cited before any station-keeping claim.
6. **Farrés, A. & Jorba, À.** (2016). "Station keeping strategies for a solar sail in the solar system." [doi:10.1007/978-3-319-27464-5_3](https://link.springer.com/chapter/10.1007/978-3-319-27464-5_3)
7. **Dachwald, B. et al.** (2013). "Solar sails are not ideal, and yes it matters." [doi:10.1007/978-3-642-34907-2_55](https://link.springer.com/chapter/10.1007/978-3-642-34907-2_55) — On the non-ideal optics caveat.
8. **Heiligers, J. & McInnes, C. R.** "Solar sail Lyapunov and halo orbits in the Earth–Moon three-body problem." [eprints.gla.ac.uk/110808](https://eprints.gla.ac.uk/110808/1/110808.pdf) — Relevant to Figures 6–7.

**The novel contribution, as far as these establish, is the dissolution criterion itself**: the closed-form tidal-parity threshold r₂ = μ^(1/3) = 3^(1/3) r_H, and the observation that it falls inside current sail performance. That claim needs a literature check before submission.

---

## Repository Layout

```
SOLAR_SAIL/
├── main.py                     End-to-end pipeline (β = 0.5 case; see caveats above)
├── src/
│   ├── dynamics.py             CR3BP EOM + ideal-reflector sail acceleration
│   ├── equilibria.py           Newton solver for artificial equilibria
│   ├── orbits.py            ★  FIXED corrector: Az enforced, branch validation
│   ├── manifolds.py            Monodromy, Floquet vectors, manifold propagation
│   ├── transfer.py             Poincaré sections, matching (now with self-intersection guard)
│   ├── critical_beta.py     ★  NEW — dissolution analysis, thresholds, Figure 8
│   ├── jacobi_match.py      ★  NEW — family continuation + true energy matching
│   ├── jacobi.py            ★  NEW — Jacobi integral (+ face-on-sail variant)
│   ├── halo_asymmetry.py    ★  NEW — the checked negative result above
│   ├── paperstyle.py        ★  NEW — plain journal figure style
│   ├── sail_control.py         Reachable set, sensitivity-matrix corrector
│   ├── stationkeeping.py       LQR controller, minimum-β sweep
│   ├── heteroclinic.py         Earth–Moon manifolds (needs rerun at a matched pair)
│   ├── paper_extras.py         Figure generators (CLI)
│   ├── animations.py           GIF / MP4 generation
│   └── viz.py                  Plotting utilities
├── regression_test.py       ★  Covers every case the pipeline depends on
└── fig1–fig5, fig8, *.gif, *.mp4, results_corrected.txt
```

---

## Getting Started

```bash
python -m venv .venv && source .venv/bin/activate
pip install numpy scipy matplotlib

python -m src.critical_beta        # thresholds + Figure 8   (the headline result)
python -m src.jacobi_match         # family continuation + matched pairs
python -m src.halo_asymmetry       # the negative result (no figure, by design)
python regression_test.py          # full regression over the pipeline

python main.py                     # full β = 0.5 pipeline (read caveats first)
python -m src.paper_extras all     # figures 1–4
python -m src.paper_extras fig5sk  # station-keeping
```

---

## Open Items

1. **Pseudo-arclength continuation.** The corrector is now sound, but natural-parameter stepping still loses branches. This blocks any fine-structure claim about the β-dependence of the families.
2. **Regenerate Figures 6–7** at an energy-matched pair (C ≈ 3.1417, Az_L1 = 0.0645, Az_L2 = 0.0495), using `min_sep` to exclude self-intersections. Report as a manifold-guided two-impulse transfer with an honest ΔV.
3. **Literature check on the dissolution criterion** — confirm the r₂ = μ^(1/3) threshold is not already in McInnes (1994) or Farrés & Jorba. Until this is done, novelty is unestablished.
4. **Reframe the manuscript** around the dissolution result rather than "the orbit became stable."
5. **Push into β ∈ [0.001, 0.05]** — halo families and their stability across that window, real heteroclinic connections there, and the off-axis (α ≠ 0) case where the force is non-conservative and the effective-potential analysis used here breaks entirely.

---

<div align="center">
<sub>Bishwaswarup · Indian Institute of Science, Bangalore · an independent project</sub>
</div>
