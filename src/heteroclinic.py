"""
heteroclinic.py — Earth-Moon L1/L2 heteroclinic connection figures.

Computes halo orbits around the Earth-Moon L1 and L2 points, propagates
their unstable/stable manifold tubes, and finds the near-heteroclinic
connection at the Moon Poincaré section (x = 1 − μ).

Physical background
───────────────────
At the same amplitude Az = 0.02 (≈ 7,688 km), the L1 and L2 halos have
essentially the same Jacobi constant (ΔC < 10⁻⁴), allowing manifold
strands from L1 and L2 to nearly intersect in phase space.  The resulting
transfer requires a ΔV ≈ 500 m/s at the Moon — orders of magnitude less
than a direct burn from low Earth orbit.

Poincaré section: x = 1 − μ  (the Moon's x-position)

Figures
───────
  fig_poincare_map()      → fig6_poincare_map.png   (y–ẏ phase portrait)
  fig_manifold_transfer() → fig7_manifold_transfer.png  (x–y rotating frame)
"""

import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from src.equilibria import find_artificial_equilibrium
from src.orbits     import compute_halo_orbit
from src.manifolds  import compute_manifold
from src.dynamics   import cr3bp_sail_eom
from src.transfer   import poincare_section, match_manifolds, transfer_dv

# ── Earth-Moon constants ───────────────────────────────────────────────────────
MU_EM   = 0.01215       # Moon / (Earth + Moon)
EM_KM   = 384_400.0     # 1 non-dim length [km]
EM_VEL  = 1.023         # characteristic velocity [km/s] = 2π·EM_KM / T_moon

# Default halo amplitude — matched so L1 and L2 have the same Jacobi constant
_AZ_DEFAULT = 0.02      # non-dim  (≈ 7,688 km out-of-plane amplitude)

_DARK   = '#0d1117'
_ORANGE = '#FF6B35'
_TEAL   = '#4ECDC4'
_YELLOW = '#FFE66D'
_WHITE  = '#e8e8e8'
_GREY   = '#555555'


# ── helpers ────────────────────────────────────────────────────────────────────

def find_em_lagrange(which: str = 'L1') -> np.ndarray:
    """Return the exact Earth-Moon L1 or L2 collinear equilibrium (no sail)."""
    gamma = (MU_EM / 3.0) ** (1.0 / 3.0)
    moon  = 1.0 - MU_EM
    x0    = [moon - gamma, 0.0, 0.0] if which == 'L1' else [moon + gamma, 0.0, 0.0]
    return find_artificial_equilibrium(0.0, 0.0, 0.0, MU_EM, x0)


def _full_orbit(state0: np.ndarray, period: float, n: int = 600) -> np.ndarray:
    """Integrate one halo period; return (6, n) array."""
    res = solve_ivp(
        cr3bp_sail_eom, [0.0, period], state0,
        args=(0.0, 0.0, 0.0, MU_EM),
        t_eval=np.linspace(0.0, period, n),
        rtol=1e-11, atol=1e-11,
    )
    return res.y


def _compute_manifolds(Az=_AZ_DEFAULT, n_strands=60, t_max=5.0*np.pi, verbose=True):
    """
    Compute all manifold strands and Poincaré crossings at x = 1−μ (Moon).

    Returns a dict with all intermediate quantities needed by both figure
    functions so they can share computation when called together.

    Notes
    -----
    • L1 unstable '+' branch travels toward the Moon.  The '−' branch goes
      toward Earth and is not useful for an L1→L2 transfer, so we skip it.
    • L2 stable '+' and '−' branches both cross the Moon section.
    • Filter: keep crossings with |y| < 0.3 and |z| < 0.2 to exclude
      strands that escape the Earth-Moon system.
    """
    if verbose:
        print("  Finding Earth-Moon L1 and L2 …")
    L1 = find_em_lagrange('L1')
    L2 = find_em_lagrange('L2')
    if verbose:
        print(f"    L1 = ({L1[0]:.5f}, 0)    L2 = ({L2[0]:.5f}, 0)")

    if verbose:
        print("  Computing L1 halo …")
    s1, T1 = compute_halo_orbit(L1, Az, MU_EM, 0., 0., 0.)
    if verbose:
        days1 = T1 * 27.32 / (2 * np.pi)
        print(f"    period = {T1:.4f} nd  ({days1:.1f} days)")

    if verbose:
        print("  Computing L2 halo …")
    s2, T2 = compute_halo_orbit(L2, Az, MU_EM, 0., 0., 0.)
    if verbose:
        days2 = T2 * 27.32 / (2 * np.pi)
        print(f"    period = {T2:.4f} nd  ({days2:.1f} days)")

    if verbose:
        print("  Propagating L1 unstable manifold (Moon-bound '+' branch) …")
    u_p = compute_manifold(s1, T1, MU_EM, 0., 0., 0.,
                           'unstable', '+', n_strands, t_max=t_max)

    if verbose:
        print("  Propagating L2 stable manifold (both branches) …")
    s_p = compute_manifold(s2, T2, MU_EM, 0., 0., 0.,
                           'stable', '+', n_strands, t_max=t_max)
    s_m = compute_manifold(s2, T2, MU_EM, 0., 0., 0.,
                           'stable', '-', n_strands, t_max=t_max)

    # ── Poincaré section at the Moon's x-position ────────────────────────────
    moon_x = 1.0 - MU_EM
    if verbose:
        print(f"  Finding crossings at x = {moon_x:.5f} (Moon) …")

    cross_u  = poincare_section(u_p, 'x', moon_x, direction=0)
    cross_sp = poincare_section(s_p, 'x', moon_x, direction=0)
    cross_sm = poincare_section(s_m, 'x', moon_x, direction=0)
    cross_s  = cross_sp + cross_sm

    # Filter: discard strands that leave the Earth-Moon system
    cross_u = [c for c in cross_u if abs(c[1]) < 0.3 and abs(c[2]) < 0.2]
    cross_s = [c for c in cross_s if abs(c[1]) < 0.3 and abs(c[2]) < 0.2]
    if verbose:
        print(f"    L1 unstable: {len(cross_u)} crossings, "
              f"L2 stable: {len(cross_s)} crossings")

    return dict(L1=L1, L2=L2, s1=s1, T1=T1, s2=s2, T2=T2,
                u_p=u_p, s_p=s_p, s_m=s_m,
                cross_u=cross_u, cross_s=cross_s, moon_x=moon_x)


# ── figure 6 — Poincaré map ────────────────────────────────────────────────────

def fig_poincare_map(
    output: str    = 'fig6_poincare_map.png',
    Az: float      = _AZ_DEFAULT,
    n_strands: int = 60,
    t_max: float   = 5.0 * np.pi,
    cache: dict    = None,
) -> dict:
    """
    Poincaré portrait (y, ẏ) at the Moon's x-position, showing the L1
    unstable and L2 stable manifold families.

    Returns the data dict so fig_manifold_transfer() can reuse it.
    """
    data = cache or _compute_manifolds(Az=Az, n_strands=n_strands,
                                       t_max=t_max, verbose=True)
    cross_u = data['cross_u']
    cross_s = data['cross_s']
    L1 = data['L1'];  L2 = data['L2']

    # ── plot (y, ẏ) at the Moon section ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor(_DARK)
    ax.set_facecolor(_DARK)

    if cross_u:
        yu  = [c[1] for c in cross_u]
        dyu = [c[4] for c in cross_u]
        ax.scatter(yu, dyu, c=_ORANGE, s=22, alpha=0.85, zorder=5,
                   label='L₁ unstable manifold')

    if cross_s:
        ys  = [c[1] for c in cross_s]
        dys = [c[4] for c in cross_s]
        ax.scatter(ys, dys, c=_TEAL, s=22, alpha=0.85, zorder=5,
                   label='L₂ stable manifold')

    # Overlap region (intersection → heteroclinic candidates)
    ax.axvline(0, color=_WHITE, lw=0.5, ls=':', alpha=0.4)
    ax.axhline(0, color=_WHITE, lw=0.5, ls=':', alpha=0.4)

    ax.set_xlabel('y  [Earth–Moon non-dim]', color=_WHITE, fontsize=12)
    ax.set_ylabel('ẏ  [non-dim]',             color=_WHITE, fontsize=12)
    ax.set_title(f'Poincaré Section  (x = Moon,  Az = {Az:.2f} nd)\n'
                 'Earth–Moon  L₁ Unstable ∩ L₂ Stable Manifolds',
                 color=_WHITE, fontsize=13, pad=10)

    ax.tick_params(colors=_WHITE)
    for sp in ax.spines.values():
        sp.set_color(_GREY)
    ax.legend(facecolor='#1a1a2e', labelcolor=_WHITE,
              framealpha=0.85, fontsize=11, loc='best')

    # Annotate equilibrium x positions for context
    moon_x = data['moon_x']
    ax.text(0.98, 0.97,
            f'Section: x = {moon_x:.4f}  (Moon)\n'
            f'L₁ @ x = {L1[0]:.4f}   L₂ @ x = {L2[0]:.4f}',
            transform=ax.transAxes, ha='right', va='top',
            color='#aaaaaa', fontsize=8)

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches='tight', facecolor=_DARK)
    plt.close(fig)
    print(f"  ✓ Saved → {output}")
    return data


# ── figure 7 — heteroclinic transfer ──────────────────────────────────────────

def fig_manifold_transfer(
    output: str    = 'fig7_manifold_transfer.png',
    Az: float      = _AZ_DEFAULT,
    n_strands: int = 60,
    t_max: float   = 5.0 * np.pi,
    cache: dict    = None,
) -> None:
    """
    Plot the best near-heteroclinic transfer trajectory in the Earth-Moon
    rotating frame (x-y projection), with the manifold tubes as context.
    """
    data = cache or _compute_manifolds(Az=Az, n_strands=n_strands,
                                       t_max=t_max, verbose=True)

    L1 = data['L1'];  L2 = data['L2']
    s1 = data['s1'];  T1 = data['T1']
    s2 = data['s2'];  T2 = data['T2']
    u_p = data['u_p']
    s_p = data['s_p'];  s_m = data['s_m']
    cross_u = data['cross_u']
    cross_s = data['cross_s']

    if not cross_u or not cross_s:
        print("  WARNING: no crossings found — cannot plot transfer.")
        return

    # ── find best near-heteroclinic connection ────────────────────────────────
    # High w_pos: minimise position residual first, then velocity
    (i_u, j_s), state_u, state_s, dv_vec = match_manifolds(
        cross_u, cross_s, w_pos=1e6)
    dv_mag, _, pos_res = transfer_dv(state_u, state_s)
    dv_ms = dv_mag * EM_VEL * 1000          # m/s
    dr_km = pos_res * EM_KM                 # km
    print(f"  Best match: |ΔV| = {dv_ms:.0f} m/s   "
          f"position residual = {dr_km:.0f} km")

    # ── full halo orbit trajectories ──────────────────────────────────────────
    orb1 = _full_orbit(s1, T1)
    orb2 = _full_orbit(s2, T2)

    # ── plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 9))
    fig.patch.set_facecolor(_DARK)
    ax.set_facecolor(_DARK)

    # Manifold tubes — faint
    for traj in u_p:
        ax.plot(traj[0], traj[1], color=_ORANGE, alpha=0.13, lw=0.6)
    for traj in s_p:
        ax.plot(traj[0], traj[1], color=_TEAL,   alpha=0.13, lw=0.6)
    for traj in s_m:
        ax.plot(traj[0], traj[1], color=_TEAL,   alpha=0.13, lw=0.6)

    # Halo orbits — bright
    ax.plot(orb1[0], orb1[1], color=_ORANGE, lw=2.2, zorder=7,
            label='L₁ halo orbit')
    ax.plot(orb2[0], orb2[1], color=_TEAL,   lw=2.2, zorder=7,
            label='L₂ halo orbit')

    # Moon's orbit (dotted circle, radius = 1−μ in dimensionless coords,
    # but in the rotating frame the Moon is a fixed point)
    # Just show its position:
    moon_x = 1.0 - MU_EM
    ax.axvline(moon_x, color='#888', lw=0.6, ls=':', alpha=0.5)

    # Primaries
    ax.scatter([-MU_EM],    [0], s=140, color='#4488ff', zorder=10, label='Earth')
    ax.scatter([moon_x],    [0], s=70,  color='#cccccc', zorder=10, label='Moon')
    ax.text(-MU_EM - 0.025, -0.07, 'Earth', color='#88aaff', fontsize=9, ha='right')
    ax.text(moon_x + 0.025, -0.07, 'Moon',  color='#cccccc', fontsize=9, ha='left')

    # Equilibrium points
    ax.scatter([L1[0], L2[0]], [0, 0], s=40, color=_WHITE,
               marker='x', zorder=9, linewidths=1.5)
    ax.text(L1[0],  0.05, 'L₁', color=_WHITE, fontsize=9, ha='center')
    ax.text(L2[0],  0.05, 'L₂', color=_WHITE, fontsize=9, ha='center')

    # ΔV connection marker
    ax.scatter([state_u[0]], [state_u[1]], s=200, color=_YELLOW,
               marker='*', zorder=12, linewidths=1,
               label=f'Transfer ΔV ≈ {dv_ms:.0f} m/s')
    ax.annotate(f'ΔV ≈ {dv_ms:.0f} m/s\n({dr_km:.0f} km residual)',
                xy=(state_u[0], state_u[1]),
                xytext=(state_u[0] + 0.06, state_u[1] + 0.08),
                color=_YELLOW, fontsize=9, arrowprops=dict(
                    arrowstyle='->', color=_YELLOW, lw=1.2))

    ax.set_xlabel('x  [Earth–Moon non-dim]', color=_WHITE, fontsize=12)
    ax.set_ylabel('y  [Earth–Moon non-dim]', color=_WHITE, fontsize=12)
    ax.set_title('Near-Heteroclinic Transfer  L₁ → L₂\nEarth–Moon Rotating Frame',
                 color=_WHITE, fontsize=13, pad=10)
    ax.set_aspect('equal', adjustable='datalim')
    ax.tick_params(colors=_WHITE)
    for sp in ax.spines.values():
        sp.set_color(_GREY)
    ax.legend(facecolor='#1a1a2e', labelcolor=_WHITE,
              framealpha=0.85, fontsize=10, loc='upper left')

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches='tight', facecolor=_DARK)
    plt.close(fig)
    print(f"  ✓ Saved → {output}")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    _args = [a for a in sys.argv[1:] if not a.startswith('#')]
    which = _args[0] if _args else 'both'

    Az    = _AZ_DEFAULT
    n     = 60
    t_max = 5.0 * np.pi

    if which in ('poincare', 'fig6', 'both'):
        print("\n── Figure 6: Poincaré map ───────────────────────────")
        cache = fig_poincare_map('fig6_poincare_map.png',
                                  Az=Az, n_strands=n, t_max=t_max)
    else:
        cache = None

    if which in ('transfer', 'fig7', 'both'):
        print("\n── Figure 7: Heteroclinic transfer ──────────────────")
        fig_manifold_transfer('fig7_manifold_transfer.png',
                               Az=Az, n_strands=n, t_max=t_max,
                               cache=cache)
