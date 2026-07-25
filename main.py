"""
main.py — Solar Sail CR3BP: end-to-end computation and paper figure.

Sail model assumption (important — read before running)
────────────────────────────────────────────────────────
A real solar sail is a thin, highly reflective membrane — square, rectangular,
or spiral — whose physical shape is fixed by the deployment mechanism.  It does
NOT physically move like an antenna dish; what changes is the spacecraft's
attitude, which rotates the membrane relative to the sun-line.

In this model we do NOT simulate the geometry of that membrane at all.  Instead:

  • α (cone angle)   — angle between the sun-line and the membrane normal.
                       α = 0 → face-on (maximum thrust); α = π/2 → edge-on (zero thrust).
  • δ (clock angle)  — azimuth of the membrane normal around the sun-line.

These two numbers fully determine the force direction under the flat-panel,
specular-reflection assumption.  There is no square, no rectangle, no boom in
the EOM — β captures the sail's area, mass, and reflectivity in one number:

    β = (sail lightness number) = A · (2P_☉/c) · (1 − μ) / (m · au²)

So β = 0.5 means the sail's radiation pressure force equals half of the Sun's
gravitational force on the spacecraft.  This is a high-performance theoretical
sail (current technology is β ~ 0.01–0.05; β ~ 0.5 is a future-generation goal).

The flat-panel model is standard for trajectory design.  A real membrane would
require integrating the radiation pressure over the surface mesh (STL model),
which changes the effective force by < 1–5% and is used only for verification.

Usage
─────
    cd /path/to/SOLAR_SAIL
    python src/main.py

Outputs
───────
    paper_figure.png   — 4-panel publication figure
    results.txt        — key numerical results for the paper
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')   # no GUI needed — saves directly to file
import matplotlib.pyplot as plt

from src.dynamics        import sail_acceleration
from src.equilibria      import find_artificial_equilibrium
from src.orbits          import compute_halo_orbit
from src.manifolds       import compute_monodromy, compute_manifold
from src.sail_control    import reachable_set
from src.transfer        import poincare_section, match_manifolds, transfer_dv
from src.viz             import make_paper_figure
from src.stationkeeping  import (minimum_beta_for_stability,
                                  simulate_stationkeeping)

# ── Global parameters ──────────────────────────────────────────────────────────

MU   = 3.003e-6    # Sun-Earth CR3BP mass parameter  (dimensionless)

# Solar sail parameters
# β = 0.5: the sail's SRP force = 50% of solar gravity on the spacecraft.
# This shifts L₁ significantly sunward — that's the whole point of the study.
BETA        = 0.5
ALPHA_SAIL  = 0.0  # cone angle [rad]  — face-on: maximum radial thrust
DELTA_SAIL  = 0.0  # clock angle [rad] — irrelevant when α = 0 (sin 0 = 0)

# Halo orbit amplitudes (Az, out-of-plane) in CR3BP non-dimensional units.
# 0.003 ≈ 450 000 km — a realistic Sun-Earth L₁ northern halo size.
AZ_CLASS = 0.003
AZ_SAIL  = 0.003

# Manifold computation settings
N_STRANDS  = 30           # strands per branch (increase for smoother tubes)
T_MAX_MAN  = 2.5 * np.pi  # max integration time per strand [non-dim]
EPS_MAN    = 1e-6          # perturbation size along eigenvector

# ─────────────────────────────────────────────────────────────────────────────

def _sep(title=""):
    bar = "─" * 60
    print(f"\n{bar}")
    if title:
        print(f"  {title}")
        print(bar)


def main():

    # ── Step 1: Equilibria ────────────────────────────────────────────────────
    _sep("Step 1  Finding equilibria")

    # Classical L₁ (β = 0): start near x ≈ 0.99
    eq_class = find_artificial_equilibrium(0.0, 0.0, 0.0, MU,
                                           [0.99, 0.0, 0.0])

    # Sail L₁ (β = 0.5): equilibrium shifts significantly sunward.
    # For large β the classical guess [0.99, 0, 0] may be too far from the root;
    # we bias the guess sunward (smaller x) to help the solver.
    x_guess_sail = eq_class[0] - 0.05   # conservative sunward offset
    eq_sail = find_artificial_equilibrium(ALPHA_SAIL, DELTA_SAIL, BETA, MU,
                                          [x_guess_sail, 0.0, 0.0])

    shift_km = (eq_class[0] - eq_sail[0]) * 1.496e8   # 1 AU = 1.496e8 km
    print(f"  Classical L₁   x = {eq_class[0]:.8f} [non-dim]")
    print(f"  Sail L₁ (β={BETA})  x = {eq_sail[0]:.8f} [non-dim]")
    print(f"  Sunward displacement = {shift_km:,.0f} km")

    # ── Step 2: Halo orbits ───────────────────────────────────────────────────
    _sep("Step 2  Computing halo orbits")

    state0_class, T_class = compute_halo_orbit(
        eq_class, Az=AZ_CLASS, mu=MU,
        alpha=0.0, delta=0.0, beta=0.0)

    state0_sail, T_sail = compute_halo_orbit(
        eq_sail, Az=AZ_SAIL, mu=MU,
        alpha=ALPHA_SAIL, delta=DELTA_SAIL, beta=BETA)

    # 1 non-dim time = 1/(2π) years for Sun-Earth
    days_per_nondim = 365.25 / (2 * np.pi)
    T_class_d = T_class * days_per_nondim
    T_sail_d  = T_sail  * days_per_nondim

    print(f"  Classical  T = {T_class:.6f}  ({T_class_d:.1f} days)"
          f"   x₀ = {state0_class[0]:.8f}   vy₀ = {state0_class[4]:.8f}")
    print(f"  Sail       T = {T_sail:.6f}  ({T_sail_d:.1f} days)"
          f"   x₀ = {state0_sail[0]:.8f}   vy₀ = {state0_sail[4]:.8f}")

    # ── Step 3: Monodromy matrix & stability ──────────────────────────────────
    _sep("Step 3  Monodromy matrix & stability exponents")

    M_class = compute_monodromy(state0_class, T_class, MU, 0.0, 0.0, 0.0)
    M_sail  = compute_monodromy(state0_sail,  T_sail,  MU,
                                ALPHA_SAIL, DELTA_SAIL, BETA)

    def _stability(M, T):
        w   = np.sort(np.abs(np.linalg.eigvals(M)))
        lam_u = w[-1]
        lam_s = w[0]
        # e-folding time: how long it takes the unstable manifold to grow by e
        tau   = T / np.log(lam_u)
        return lam_u, lam_s, tau

    lu_c, ls_c, tau_c = _stability(M_class, T_class)
    lu_s, ls_s, tau_s = _stability(M_sail,  T_sail)

    print(f"  Classical  λ_u = {lu_c:.4e}   λ_s = {ls_c:.4e}"
          f"   τ_e = {tau_c:.4f}  ({tau_c*days_per_nondim:.1f} days)")
    print(f"  Sail       λ_u = {lu_s:.4e}   λ_s = {ls_s:.4e}"
          f"   τ_e = {tau_s:.4f}  ({tau_s*days_per_nondim:.1f} days)")
    print()
    print("  Interpretation: larger λ_u → more unstable → faster divergence.")
    print("  The sail can tune this by adjusting β at the cost of orbit size.")

    # ── Step 4: Manifold tubes ────────────────────────────────────────────────
    _sep(f"Step 4  Computing manifold tubes  ({N_STRANDS} strands per branch)")

    strands_u = compute_manifold(
        state0_class, T_class, MU, 0.0, 0.0, 0.0,
        direction='unstable', branch='+',
        n_points=N_STRANDS, eps=EPS_MAN, t_max=T_MAX_MAN)

    strands_s = compute_manifold(
        state0_class, T_class, MU, 0.0, 0.0, 0.0,
        direction='stable', branch='+',
        n_points=N_STRANDS, eps=EPS_MAN, t_max=T_MAX_MAN)

    print(f"  Unstable: {len(strands_u)} strands  shape {strands_u[0].shape}")
    print(f"  Stable:   {len(strands_s)} strands  shape {strands_s[0].shape}")

    # ── Step 5: Poincaré section & best ΔV transfer ───────────────────────────
    _sep("Step 5  Poincaré section  y = 0  →  best transfer")

    cross_u = poincare_section(strands_u, section='y', value=0.0, direction=0)
    cross_s = poincare_section(strands_s, section='y', value=0.0, direction=0)
    print(f"  Crossings:  unstable = {len(cross_u)}   stable = {len(cross_s)}")

    dv_mag = pos_res = np.nan
    su = ss = None
    if cross_u and cross_s:
        (ib, jb), su, ss, dv_vec = match_manifolds(cross_u, cross_s)
        dv_mag, _, pos_res = transfer_dv(su, ss)
        # 1 non-dim velocity = 29.784 km/s for Sun-Earth
        dv_ms = dv_mag * 29_784.7
        print(f"  Best pair  ({ib}, {jb})")
        print(f"  |Δv|      = {dv_mag:.6f} [non-dim]  ≈ {dv_ms:.2f} m/s")
        print(f"  pos_res   = {pos_res:.6f} [non-dim]")
    else:
        print("  WARNING: no crossings — increase N_STRANDS or T_MAX_MAN")

    # ── Step 6: Reachable sail acceleration set ───────────────────────────────
    _sep("Step 6  Reachable acceleration cloud at sail L₁")

    # Sample on a coarser grid for speed — (α, δ) ∈ [0, π/2] × [0, 2π)
    cloud = reachable_set(list(eq_sail), BETA, MU, n_alpha=40, n_delta=48)
    a_norms = np.linalg.norm(cloud, axis=1)
    print(f"  Samples:  {cloud.shape[0]}   (n_alpha=40, n_delta=48)")
    print(f"  |a| range:  [{a_norms.min():.4f}, {a_norms.max():.4f}] [non-dim]")
    print(f"  Max |a| ≈ {a_norms.max() * 29_784.7 / days_per_nondim:.2f} km/s per day")

    # ── Step 7: Compose the paper figure ─────────────────────────────────────
    _sep("Step 7  Composing paper figure")

    fig = make_paper_figure(
        state0_class, T_class, MU,
        state0_sail,  T_sail,  ALPHA_SAIL, DELTA_SAIL, BETA,
        strands_u, strands_s,
        cross_u, cross_s,
        cloud,
        eq_pos_sail=eq_sail,
    )

    out_fig = 'paper_figure.png'
    fig.savefig(out_fig, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved → {out_fig}")

    # ── Step 9: LQR station-keeping — sub-L₁ sentinel ────────────────────────
    _sep("Step 9  LQR station-keeping  (sub-L₁ solar-storm sentinel)")

    # Find the minimum sail lightness that the LQR controller can stabilise.
    # These are sub-L₁ equilibria displaced sunward from the classical L₁ —
    # the sentinel concept for increased space-weather warning time.
    SK_BETAS = np.linspace(0.001, 0.10, 20)
    sk_results, sk_beta_min = minimum_beta_for_stability(
        betas=SK_BETAS, verbose=False)

    print(f"  Minimum β for LQR stability : {sk_beta_min:.4f}")

    # Simulate 30-day arc at β = 0.05 (near-term realistic sail)
    SK_SIM_BETA = 0.05
    sk_sim = simulate_stationkeeping(
        SK_SIM_BETA, duration_days=30,
        perturbation_km=100, verbose=True)

    sk_shift_km = (0.990027 - sk_sim['eq'][0]) * 1.496e8
    sk_warn_min = sk_shift_km / (750 * 60)   # extra minutes at 750 km/s
    print(f"  Sub-L₁ shift  = {sk_shift_km/1e6:.2f} × 10⁶ km")
    print(f"  Extra warning = {sk_warn_min:.1f} min  (@ 750 km/s solar wind)")
    print(f"  30-day ΔV     = {sk_sim['delta_v_ms'][-1]:.4f} m/s  "
          f"(propellantless — attitude only)")

    # ── Step 8: Write results summary ─────────────────────────────────────────
    _sep("Step 8  Writing results")

    summary = f"""\
Solar Sail CR3BP — Numerical Results
=====================================
Run date : (see file timestamp)

─── Sail model ─────────────────────────────────────────────────────────────
The sail is modelled as an ideal flat panel with instantaneous attitude control.
α and δ are mathematical parameters, NOT the physical motion of a membrane.
A real rectangular or square sail has fixed geometry; what changes is the
spacecraft attitude (which rotates the membrane relative to the sun-line).
β = {BETA} means the SRP force equals {int(BETA*100)}% of solar gravity on the craft.
Real-sail corrections (billowing, finite slew rate, shadowing) are < ~5% and
would be captured by a small β perturbation or an STL-based force model.

─── Parameters ─────────────────────────────────────────────────────────────
  μ (Sun-Earth)     = {MU:.4e}
  β (sail lightness) = {BETA}
  α, δ (sail angles) = {np.degrees(ALPHA_SAIL):.1f}°, {np.degrees(DELTA_SAIL):.1f}°
  Az (halo amplitude) = {AZ_CLASS:.3f} non-dim  ≈ {AZ_CLASS * 1.496e8 / 1e3:.0f} 000 km

─── Equilibria ─────────────────────────────────────────────────────────────
  Classical L₁  x = {eq_class[0]:.8f} [non-dim]
  Sail L₁       x = {eq_sail[0]:.8f} [non-dim]
  Sunward shift   = {shift_km:,.0f} km

─── Halo orbits ────────────────────────────────────────────────────────────
  Classical  T = {T_class:.6f} non-dim  ({T_class_d:.1f} days)
             state0 = {np.round(state0_class, 8)}
  Sail       T = {T_sail:.6f} non-dim  ({T_sail_d:.1f} days)
             state0 = {np.round(state0_sail, 8)}

─── Monodromy / stability ──────────────────────────────────────────────────
  Classical  λ_u = {lu_c:.6e}   λ_s = {ls_c:.6e}
             e-fold τ = {tau_c:.4f} non-dim  ({tau_c * days_per_nondim:.1f} days)
  Sail       λ_u = {lu_s:.6e}   λ_s = {ls_s:.6e}
             e-fold τ = {tau_s:.4f} non-dim  ({tau_s * days_per_nondim:.1f} days)

─── Poincaré / transfer ────────────────────────────────────────────────────
  Unstable crossings : {len(cross_u)}
  Stable  crossings  : {len(cross_s)}
  Best |Δv|          = {dv_mag:.6f} [non-dim]  ≈ {dv_mag * 29_784.7:.2f} m/s
  Position residual  = {pos_res:.6f} [non-dim]

─── Reachable set ──────────────────────────────────────────────────────────
  Samples            : {cloud.shape[0]}
  Max |a|            = {a_norms.max():.6f} [non-dim]
  |a| range          = [{a_norms.min():.4f}, {a_norms.max():.4f}]

─── LQR Station-keeping (sub-L₁ sentinel) ──────────────────────────────────
  Minimum β for LQR stability : {sk_beta_min:.4f}
  Simulation β                : {SK_SIM_BETA}
  Sub-L₁ sunward shift        : {sk_shift_km/1e6:.2f} × 10⁶ km
  Extra warning time          : {sk_warn_min:.1f} min  (@ 750 km/s solar wind)
  30-day ΔV (attitude only)   : {sk_sim['delta_v_ms'][-1]:.4f} m/s
  Closed-loop max Re(λ)       : {np.max(sk_sim['eigs_cl'].real):.5f}
"""

    out_txt = 'results.txt'
    with open(out_txt, 'w') as f:
        f.write(summary)
    print(f"  Saved → {out_txt}")

    print("\n" + "─" * 60)
    print("  Done.  Check paper_figure.png and results.txt.")
    print("─" * 60)


if __name__ == "__main__":
    main()