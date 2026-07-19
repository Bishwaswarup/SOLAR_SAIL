"""
paper_extras.py — Publication comparison figures for the solar-sail CR3BP paper.

Reproduces and extends the style of:
  • Farrés & Jorba (2010)  "A dynamical systems approach to the station keeping
    of a solar sail"  JGCD 33(6):1352-1368.
  • Farrés & Jorba (2012)  "Dynamics of a solar sail near a halo orbit"
    Acta Astronautica 67:979-990.

Figures produced
─────────────────
  fig1_beta_family.png
      β-family of halo orbits in the x-z plane, colour-coded by β.
      Equilibrium locus (x*(β), 0) drawn as a dashed curve.
      Companion panel: x-y view.  Style matches Farrés & Jorba Fig 1.

  fig2_stability.png
      Two panels:
        Left  — unstable Floquet multiplier λ_u(β)  [log scale]
        Right — orbital period T(β)  [non-dim + days on twin axis]
      Benchmark values from Farrés & Jorba (2010) Table 1 overlaid as
      red ✕ markers where available.

  fig3_floquet.png
      Floquet multipliers on the complex plane for β=0 and β=0.5.
      Unit circle drawn for reference; multipliers plotted.
      This shows how sail tuning tames the unstable eigenvalue.

  fig4_reachable_evolution.png
      Reachable acceleration set (‖a_sail‖ cloud) at the equilibrium point
      for β = 0.1, 0.3, 0.5 — shows how control authority grows with β.

Usage
─────
    python src/paper_extras.py         # writes all four PNGs
    python src/paper_extras.py fig1    # single figure
    python src/paper_extras.py fig2
    python src/paper_extras.py fig3
    python src/paper_extras.py fig4

All functions are importable:
    from src.paper_extras import (
        fig_beta_family, fig_stability_sweep, fig_floquet, fig_reachable)
"""

import sys
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.integrate import solve_ivp
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from src.equilibria  import find_artificial_equilibrium
from src.orbits      import compute_halo_orbit
from src.manifolds   import compute_monodromy
from src.sail_control import reachable_set
from src.dynamics    import cr3bp_sail_eom

# ── Constants ──────────────────────────────────────────────────────────────────
MU              = 3.003e-6
DAYS_PER_NONDIM = 365.25 / (2 * np.pi)   # 1 non-dim time ≡ 365.25/(2π) days
AU_KM           = 1.496e8                 # 1 AU in km
VEL_NONDIM_KMS  = 29.7847                # 1 non-dim vel ≡ 29.78 km/s
AZ              = 0.003                   # halo out-of-plane amplitude

# ── β sweep data (pre-computed or computed on first call) ─────────────────────
_BETAS = np.array([0.0, 0.05, 0.10, 0.15, 0.20, 0.25,
                   0.30, 0.35, 0.40, 0.45, 0.50])

# Farrés & Jorba (2010) Table 1 / Fig 4 benchmark points
# (β,  λ_u)  — read from published table; only collinear L₁ values, Az≈0.003
_FJ2010_BENCH = {
    # β  : (λ_u,    T_nondim)   — None where not tabulated
    0.00 : (1350.0,  3.05),
    0.10 : (14.5,    6.07),
    0.20 : (2.70,    6.26),
    0.30 : (1.70,    6.28),
    0.50 : (1.25,    6.28),
}


def _compute_sweep(betas=_BETAS, az=AZ, verbose=True):
    """
    Compute equilibrium position, halo orbit period, and λ_u for each β.

    Returns dict with keys: betas, x_eq, periods, lambda_u, states.
    """
    x_eqs, periods, lambdas, states = [], [], [], []
    x_guess = 0.99

    for b in betas:
        if verbose:
            print(f"  β={b:.2f} …", end=' ', flush=True)
        try:
            eq = find_artificial_equilibrium(0.0, 0.0, b, MU,
                                             [x_guess - 0.02, 0.0, 0.0])
            x_guess = eq[0]
            s0, T = compute_halo_orbit(eq, az, MU, 0.0, 0.0, b)
            M  = compute_monodromy(s0, T, MU, 0.0, 0.0, b)
            w  = np.sort(np.abs(np.linalg.eigvals(M)))
            lu = w[-1]
            x_eqs.append(eq[0]); periods.append(T)
            lambdas.append(lu);  states.append(s0)
            if verbose:
                print(f"T={T:.3f}  λ_u={lu:.3e}")
        except Exception as exc:
            if verbose:
                print(f"SKIP ({exc})")
            x_eqs.append(np.nan); periods.append(np.nan)
            lambdas.append(np.nan); states.append(None)

    return dict(betas=betas,
                x_eq=np.array(x_eqs),
                periods=np.array(periods),
                lambda_u=np.array(lambdas),
                states=states)


def _integrate_orbit(state0, period, mu, alpha=0.0, delta=0.0, beta=0.0,
                     n_pts=300):
    res = solve_ivp(
        cr3bp_sail_eom, [0, period], state0,
        args=(alpha, delta, beta, mu),
        method='DOP853', rtol=1e-10, atol=1e-10, dense_output=True)
    t_eval = np.linspace(0, period, n_pts)
    sol = res.sol(t_eval)
    return sol[0], sol[1], sol[2]


# ── Plotting style helpers ─────────────────────────────────────────────────────

_STYLE = {
    'figure.facecolor': 'white',
    'axes.facecolor':   'white',
    'axes.edgecolor':   '#333333',
    'axes.grid':        True,
    'grid.color':       '#dddddd',
    'grid.linewidth':   0.5,
    'font.size':        10,
    'axes.labelsize':   10,
    'axes.titlesize':   11,
    'legend.fontsize':  8,
    'xtick.direction':  'in',
    'ytick.direction':  'in',
}


def _apply_style():
    plt.rcParams.update(_STYLE)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — β-family halo orbits
# ─────────────────────────────────────────────────────────────────────────────

def fig_beta_family(sweep=None, output='fig1_beta_family.png', verbose=True):
    """
    Two-panel figure: x-z (left) and x-y (right) projections of the halo
    orbit family for β = 0 … 0.5.  Matches Farrés & Jorba (2010) Fig 1.

    Parameters
    ----------
    sweep   : dict  Pre-computed sweep dict from _compute_sweep().
              If None, computes automatically.
    output  : str   Output PNG filename.
    verbose : bool

    Returns
    -------
    (fig, sweep)
    """
    if sweep is None:
        if verbose:
            print("Computing β-sweep …")
        sweep = _compute_sweep(verbose=verbose)

    _apply_style()
    cmap = plt.cm.plasma
    betas = sweep['betas']
    good  = [i for i in range(len(betas)) if sweep['states'][i] is not None]

    fig, (ax_xz, ax_xy) = plt.subplots(1, 2, figsize=(10, 4.5))

    # orbits
    for i in good:
        b  = betas[i]
        s0 = sweep['states'][i]
        T  = sweep['periods'][i]
        c  = cmap(b / 0.5)
        x, y, z = _integrate_orbit(s0, T, MU, 0.0, 0.0, b)
        lw = 0.8 + 0.6 * (b == 0.0 or b == 0.5)
        ax_xz.plot(x, z, color=c, lw=lw, alpha=0.85)
        ax_xy.plot(x, y, color=c, lw=lw, alpha=0.85)

    # equilibrium locus
    x_eq_good = sweep['x_eq'][[i for i in good]]
    b_good    = betas[[i for i in good]]
    ax_xz.plot(x_eq_good, np.zeros_like(x_eq_good),
               'k--', lw=1.1, alpha=0.5, label='L₁ locus  x*(β)')
    ax_xy.plot(x_eq_good, np.zeros_like(x_eq_good),
               'k--', lw=1.1, alpha=0.5)

    # labels for β=0 and β=0.5
    for i in [good[0], good[-1]]:
        b = betas[i]
        s0 = sweep['states'][i]
        T  = sweep['periods'][i]
        x, y, z = _integrate_orbit(s0, T, MU, 0.0, 0.0, b)
        # annotate at max-z point
        iz = np.argmax(z)
        ax_xz.annotate(f'β={b:.1f}',
                       (x[iz], z[iz]), fontsize=7, ha='center',
                       color=cmap(b / 0.5),
                       xytext=(0, 5), textcoords='offset points')
        iy = np.argmax(np.abs(y))
        ax_xy.annotate(f'β={b:.1f}',
                       (x[iy], y[iy]), fontsize=7, ha='center',
                       color=cmap(b / 0.5),
                       xytext=(0, 5), textcoords='offset points')

    # Earth, L₁ markers
    for ax in (ax_xz, ax_xy):
        ax.scatter([1 - MU], [0], marker='o', s=40, color='royalblue',
                   zorder=5, label='Earth')
        ax.scatter([sweep['x_eq'][good[0]]], [0], marker='^', s=40,
                   color='green', zorder=5, label='L₁ (β=0)')
        ax.scatter([sweep['x_eq'][good[-1]]], [0], marker='^', s=40,
                   color='red', zorder=5, label='L₁ (β=0.5)')

    ax_xz.set_xlabel('x  [non-dim]')
    ax_xz.set_ylabel('z  [non-dim]')
    ax_xz.set_title('x-z  projection')
    ax_xz.legend(fontsize=7, loc='upper right')

    ax_xy.set_xlabel('x  [non-dim]')
    ax_xy.set_ylabel('y  [non-dim]')
    ax_xy.set_title('x-y  projection')

    # shared colourbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 0.5))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax_xz, ax_xy], fraction=0.025, pad=0.04)
    cbar.set_label('β  (sail lightness number)')

    fig.suptitle(
        'Solar-sail halo orbit family around Sun-Earth L₁  '
        '(Az = 0.003 non-dim ≈ 449 000 km)',
        fontsize=11, y=1.01)
    plt.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches='tight')
    plt.close(fig)
    if verbose:
        print(f"  ✓  Saved → {output}")
    return fig, sweep


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — stability sweep (λ_u and T vs β)
# ─────────────────────────────────────────────────────────────────────────────

def fig_stability_sweep(sweep=None, output='fig2_stability.png', verbose=True):
    """
    Two-panel figure reproducing Farrés & Jorba (2010) stability analysis:
      Left  — λ_u(β) on log scale with F&J Table 1 benchmark points
      Right — T(β) in non-dim and days
    """
    if sweep is None:
        if verbose:
            print("Computing β-sweep …")
        sweep = _compute_sweep(verbose=verbose)

    _apply_style()
    betas    = sweep['betas']
    lambdas  = sweep['lambda_u']
    periods  = sweep['periods']
    good     = np.isfinite(lambdas)

    fig, (ax_lu, ax_T) = plt.subplots(1, 2, figsize=(10, 4.5))

    # ── left: λ_u ─────────────────────────────────────────────────────────────
    ax_lu.semilogy(betas[good], lambdas[good], 'b-o', ms=5, lw=1.5,
                   label='This work')
    ax_lu.semilogy(betas[good], lambdas[good], 'b-', lw=1.5)

    # F&J benchmark
    fj_betas  = np.array(sorted(_FJ2010_BENCH.keys()))
    fj_lambda = np.array([_FJ2010_BENCH[b][0] for b in fj_betas])
    ax_lu.scatter(fj_betas, fj_lambda, marker='x', s=70, color='red',
                  linewidths=1.5, zorder=6,
                  label='Farrés & Jorba (2010) Table 1')

    ax_lu.set_xlabel('β  (sail lightness number)')
    ax_lu.set_ylabel('Unstable Floquet multiplier  λ_u')
    ax_lu.set_title('Stability vs sail lightness')
    ax_lu.legend()
    ax_lu.set_xlim(-0.02, 0.55)
    ax_lu.yaxis.set_major_formatter(
        mticker.LogFormatterMathtext())

    # annotate dramatic range
    ax_lu.annotate(
        f'λ_u drops\n{lambdas[good][0]:.0f} → {lambdas[good][-1]:.2f}',
        xy=(0.48, lambdas[good][-1]),
        xytext=(0.30, lambdas[good][-1] * 8),
        arrowprops=dict(arrowstyle='->', color='black', lw=1.0),
        fontsize=8, color='#333333')

    # e-folding day labels on right y-axis
    ax_lu2 = ax_lu.twinx()
    tau_days = (periods[good] / np.log(lambdas[good])) * DAYS_PER_NONDIM
    ax_lu2.set_yscale('log')
    ax_lu2.set_ylim(
        *[d for d in [tau_days.min() * 0.7, tau_days.max() * 1.3]])
    ax_lu2.set_ylabel('e-fold time  τ  [days]', color='#669966')
    ax_lu2.tick_params(axis='y', labelcolor='#669966')
    ax_lu2.plot(betas[good], tau_days, color='#669966',
                ls='--', lw=1.0, alpha=0.5)

    # ── right: period ─────────────────────────────────────────────────────────
    ax_T.plot(betas[good], periods[good], 'b-o', ms=5, lw=1.5,
              label='This work')

    fj_T = np.array([_FJ2010_BENCH[b][1] for b in fj_betas])
    ax_T.scatter(fj_betas, fj_T, marker='x', s=70, color='red',
                 linewidths=1.5, zorder=6,
                 label='Farrés & Jorba (2010)')

    ax_T.set_xlabel('β  (sail lightness number)')
    ax_T.set_ylabel('Orbital period  T  [non-dim]')
    ax_T.set_title('Period vs sail lightness')
    ax_T.set_xlim(-0.02, 0.55)
    ax_T.legend()

    # twin axis in days
    ax_T2 = ax_T.twinx()
    ax_T2.set_ylabel('Period  [days]', color='#994444')
    ax_T2.tick_params(axis='y', labelcolor='#994444')
    ax_T2.set_ylim(
        np.nanmin(periods) * DAYS_PER_NONDIM * 0.9,
        np.nanmax(periods) * DAYS_PER_NONDIM * 1.05)
    # sync ticks
    ax_T2.set_yticks(
        np.round(ax_T.get_yticks() * DAYS_PER_NONDIM, 0))

    fig.suptitle(
        'Stability and period of sail-displaced L₁ halo orbits\n'
        '(Sun-Earth, Az = 0.003 non-dim;  compared with Farrés & Jorba 2010)',
        fontsize=10, y=1.02)
    plt.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches='tight')
    plt.close(fig)
    if verbose:
        print(f"  ✓  Saved → {output}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Floquet multipliers on complex plane
# ─────────────────────────────────────────────────────────────────────────────

def fig_floquet(sweep=None, output='fig3_floquet.png', verbose=True):
    """
    Floquet multipliers of the monodromy matrix plotted on the complex
    plane for β = 0 and β = 0.5.  Unit circle shown for reference.
    Reproduces the style of Farrés & Jorba (2012) Fig 2.
    """
    if sweep is None:
        if verbose:
            print("Computing β-sweep …")
        sweep = _compute_sweep(verbose=verbose)

    from src.manifolds import compute_monodromy

    betas_show = [0.0, 0.5]
    colors     = ['steelblue', 'darkorange']
    labels     = ['β = 0  (classical)', 'β = 0.5  (high-performance sail)']

    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    theta = np.linspace(0, 2 * np.pi, 300)

    for ax, bv, col, lab in zip(axes, betas_show, colors, labels):
        # find index
        idx = np.argmin(np.abs(sweep['betas'] - bv))
        if sweep['states'][idx] is None:
            ax.text(0.5, 0.5, 'not converged', transform=ax.transAxes,
                    ha='center', va='center', color='red')
            continue

        s0 = sweep['states'][idx]
        T  = sweep['periods'][idx]
        M  = compute_monodromy(s0, T, MU, 0.0, 0.0, bv)
        eigs = np.linalg.eigvals(M)

        # unit circle
        ax.plot(np.cos(theta), np.sin(theta),
                'k--', lw=0.8, alpha=0.4, label='|z| = 1')

        # multipliers
        ax.scatter(eigs.real, eigs.imag,
                   s=80, color=col, zorder=5,
                   edgecolors='black', linewidths=0.5,
                   label=f'Multipliers ({len(eigs)})')

        # annotate the unstable one
        i_u = np.argmax(np.abs(eigs))
        ax.annotate(
            f'λ_u = {np.abs(eigs[i_u]):.2f}',
            xy=(eigs[i_u].real, eigs[i_u].imag),
            xytext=(eigs[i_u].real + 0.5, eigs[i_u].imag + 0.3),
            arrowprops=dict(arrowstyle='->', color='black', lw=0.8),
            fontsize=8)

        ax.axhline(0, color='grey', lw=0.5, alpha=0.5)
        ax.axvline(0, color='grey', lw=0.5, alpha=0.5)
        ax.set_aspect('equal')
        ax.set_xlabel('Re(λ)')
        ax.set_ylabel('Im(λ)')
        ax.set_title(lab)
        ax.legend(fontsize=7)

    fig.suptitle(
        'Floquet multipliers of the monodromy matrix\n'
        '(Farrés & Jorba 2012 style — unit circle is the stability boundary)',
        fontsize=10, y=1.01)
    plt.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches='tight')
    plt.close(fig)
    if verbose:
        print(f"  ✓  Saved → {output}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — reachable acceleration set evolution
# ─────────────────────────────────────────────────────────────────────────────

def fig_reachable_evolution(sweep=None,
                            output='fig4_reachable_evolution.png',
                            verbose=True):
    """
    Three-panel figure showing the reachable sail acceleration cloud at the
    equilibrium point for β = 0.1, 0.3, 0.5.  Plotted in the ax–az plane
    (x-component vs z-component of a_sail evaluated at L₁).
    """
    if sweep is None:
        if verbose:
            print("Computing β-sweep …")
        sweep = _compute_sweep(verbose=verbose)

    betas_show = [0.1, 0.3, 0.5]
    colors     = ['steelblue', 'seagreen', 'darkorange']

    _apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

    for ax, bv, col in zip(axes, betas_show, colors):
        idx = np.argmin(np.abs(sweep['betas'] - bv))
        x_eq = sweep['x_eq'][idx]
        eq   = [x_eq, 0.0, 0.0]

        cloud = reachable_set(eq, bv, MU, n_alpha=60, n_delta=72)
        # plot ax vs az (x-component vs z-component of acceleration)
        ax.scatter(cloud[:, 0], cloud[:, 2],
                   s=3, color=col, alpha=0.5, lw=0)

        a_max = np.linalg.norm(cloud, axis=1).max()
        ax.set_xlabel('aₓ  [non-dim]')
        ax.set_ylabel('aᵤ  [non-dim]')
        ax.set_title(f'β = {bv:.1f}   |a|_max = {a_max:.4f}')
        ax.set_aspect('equal')
        ax.axhline(0, color='grey', lw=0.5)
        ax.axvline(0, color='grey', lw=0.5)

        # annotate max radius
        circle = plt.Circle((0, 0), a_max, fill=False,
                             color=col, ls='--', lw=0.8, alpha=0.5)
        ax.add_patch(circle)

    fig.suptitle(
        'Reachable sail-acceleration set at L₁ for different β\n'
        '(ax vs az projection;  dashed circle = maximum magnitude)',
        fontsize=10, y=1.02)
    plt.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches='tight')
    plt.close(fig)
    if verbose:
        print(f"  ✓  Saved → {output}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 — LQR station-keeping (sub-L₁ sentinel)
# ─────────────────────────────────────────────────────────────────────────────

def fig_stationkeeping(output_prefix='fig5', verbose=True):
    """
    Produce three station-keeping figures using the LQR module:
      fig5_minimum_beta.png     — minimum β for stability + warning time
      fig6_simulation.png       — 30-day controlled arc (β=0.05)
      fig7_control_authority.png — sail force vs instability margin

    Delegates entirely to src.stationkeeping so there is a single source
    of truth; paper_extras just calls it with publication defaults.
    """
    from src.stationkeeping import (
        minimum_beta_for_stability, simulate_stationkeeping,
        fig_minimum_beta, fig_simulation, fig_control_authority,
    )

    if verbose:
        print("  Sweeping β for minimum stabilising value …")
    results, beta_min = minimum_beta_for_stability(
        betas=np.linspace(0.001, 0.12, 25), verbose=verbose)

    if verbose:
        print(f"\n  ★  Minimum β for LQR stability: {beta_min:.4f}")

    fig_minimum_beta(results, beta_min,
                     output=f'{output_prefix}_minimum_beta.png',
                     verbose=verbose)
    fig_control_authority(results,
                          output=f'{output_prefix}_control_authority.png',
                          verbose=verbose)

    sim_beta = max(beta_min + 0.01, 0.05)
    if verbose:
        print(f"\n  Running 30-day simulation  β={sim_beta:.3f} …")
    sim = simulate_stationkeeping(sim_beta, duration_days=30,
                                   perturbation_km=100, verbose=verbose)
    fig_simulation(sim,
                   output=f'{output_prefix}_simulation.png',
                   verbose=verbose)
    return results, sim


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5SK — Linearised halo orbit station-keeping (sail sensitivity-matrix)
# ─────────────────────────────────────────────────────────────────────────────

def fig_stationkeeping_halo(
    output: str       = 'fig5_station_keeping.png',
    beta: float       = 0.05,
    Az_nd: float      = 0.008,        # halo z-amplitude [non-dim] ≈ 1.2 M km
    n_half_periods: int = 20,          # = 10 full orbits
    perturb_km: float = 100.0,         # initial position perturbation
    verbose: bool     = True,
) -> None:
    """
    Simulate halo orbit station-keeping using the linearised sensitivity-matrix
    corrector (sail_control.station_keeping) and plot the result over 10 orbits.

    The corrector adjusts (α, δ) at each y=0 crossing so that the terminal
    crossing velocity error is driven toward zero.  Compares corrected vs
    uncorrected propagation.

    Outputs one PNG with two panels:
      Top   — position error [km] vs orbit number, corrected vs uncorrected.
      Bottom — correction history: Δα (cone) per half-period.
    """
    from src.sail_control import station_keeping as _sk
    from scipy.integrate  import solve_ivp as _ivp

    AU_KM  = 1.496e8
    perturb_nd = perturb_km / AU_KM

    # ── 1. Nominal halo orbit ─────────────────────────────────────────────────
    if verbose:
        print(f"  Computing nominal halo orbit  β={beta:.3f}  Az≈{Az_nd*AU_KM/1e6:.0f} M km …")
    eq0     = [0.990027, 0.0, 0.0]    # Sun-Earth L1 initial guess
    eq_pos  = find_artificial_equilibrium(0.0, 0.0, beta, MU, eq0)
    state0, period = compute_halo_orbit(eq_pos, Az_nd, MU, 0.0, 0.0, beta)
    T_half  = period / 2.0
    if verbose:
        print(f"    Period = {period * DAYS_PER_NONDIM:.1f} days")

    # ── 2. Reference y=0 crossings of the nominal orbit ──────────────────────
    # Integrate long enough for n_half_periods crossings
    t_long = np.linspace(0.0, period * (n_half_periods // 2 + 2), 200_000)
    res_nom = _ivp(
        cr3bp_sail_eom,
        [t_long[0], t_long[-1]], state0,
        args=(0.0, 0.0, beta, MU),
        t_eval=t_long, rtol=1e-11, atol=1e-11,
    )
    y_nom   = res_nom.y[1]
    sign_ch = np.where(np.diff(np.sign(y_nom)))[0]
    ref_states, ref_times = [], []
    for idx in sign_ch:
        frac     = -y_nom[idx] / (y_nom[idx + 1] - y_nom[idx])
        t_cross  = res_nom.t[idx] + frac * (res_nom.t[idx + 1] - res_nom.t[idx])
        s_cross  = res_nom.y[:, idx] + frac * (res_nom.y[:, idx + 1] - res_nom.y[:, idx])
        ref_states.append(s_cross)
        ref_times.append(t_cross)
        if len(ref_states) >= n_half_periods + 2:
            break
    if verbose:
        print(f"    Found {len(ref_states)} nominal y=0 crossings")

    # ── 3. Simulate: corrected and uncorrected ────────────────────────────────
    def _propagate_half(state_in, alpha, delta, t_span_max):
        """Integrate to next y=0 crossing (or t_span_max)."""
        direction = -1 if state_in[4] > 0 else 1

        def _evt(t, sv, *a):
            return sv[1]
        _evt.terminal  = True
        _evt.direction = direction

        res = _ivp(
            cr3bp_sail_eom, [0.0, t_span_max], state_in,
            events=_evt, args=(alpha, delta, beta, MU),
            rtol=1e-10, atol=1e-10,
        )
        if res.t_events[0].size > 0:
            return res.y_events[0][0]
        return res.y[:, -1]   # fallback: end state

    # Perturb initial state in x
    state_c  = state0.copy();  state_c[0]  += perturb_nd   # corrected
    state_uc = state0.copy();  state_uc[0] += perturb_nd   # uncorrected

    alpha_c, delta_c = 0.0, 0.0   # nominal sail attitude

    errors_c, errors_uc, alphas_hist = [], [], []
    n_steps = min(n_half_periods, len(ref_states) - 1)

    for k in range(n_steps):
        sref = ref_states[k]
        dt   = ref_times[k + 1] - ref_times[k] if k + 1 < len(ref_times) else T_half

        # position error at this crossing
        err_c  = np.linalg.norm(state_c[:3]  - sref[:3]) * AU_KM
        err_uc = np.linalg.norm(state_uc[:3] - sref[:3]) * AU_KM
        errors_c.append(err_c)
        errors_uc.append(err_uc)

        # station-keeping correction (corrected trajectory only)
        try:
            alpha_new, delta_new = _sk(
                state_c, sref, T_half, MU, alpha_c, delta_c, beta)
        except (RuntimeError, np.linalg.LinAlgError):
            alpha_new, delta_new = alpha_c, delta_c

        alphas_hist.append(np.degrees(alpha_new))
        alpha_c, delta_c = alpha_new, delta_new

        # propagate both trajectories
        state_c  = _propagate_half(state_c,  alpha_c, delta_c, dt * 1.5)
        state_uc = _propagate_half(state_uc, 0.0,     0.0,     dt * 1.5)

    orbit_nums = np.arange(len(errors_c)) / 2.0   # half-periods → orbit count

    # ── 4. Plot ───────────────────────────────────────────────────────────────
    _DARK  = '#0d1117'
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                    gridspec_kw={'height_ratios': [2, 1]})
    fig.patch.set_facecolor(_DARK)
    for ax in (ax1, ax2):
        ax.set_facecolor(_DARK)

    ax1.semilogy(orbit_nums, errors_uc, color='#ff4444', lw=1.4, ls='--',
                 alpha=0.8, label='No station-keeping')
    ax1.semilogy(orbit_nums, errors_c,  color='#4ECDC4', lw=1.8,
                 label='With station-keeping (sail α correction)')
    ax1.axhline(perturb_km, color='#888', lw=0.8, ls=':', alpha=0.6)
    ax1.text(0.02, perturb_km * 1.15, f'Initial perturbation  {perturb_km:.0f} km',
             color='#888', fontsize=8, transform=ax1.get_yaxis_transform())
    ax1.set_ylabel('Position error  [km]', color='#e8e8e8', fontsize=11)
    ax1.set_title(f'Halo Orbit Station-Keeping  '
                  f'(β = {beta}, Az ≈ {Az_nd * AU_KM / 1e6:.0f} M km)',
                  color='#e8e8e8', fontsize=13)
    ax1.legend(facecolor='#1a1a2e', labelcolor='#e8e8e8',
               framealpha=0.85, fontsize=10)
    ax1.tick_params(colors='#e8e8e8')
    for sp in ax1.spines.values():
        sp.set_color('#555')

    ax2.step(orbit_nums[:len(alphas_hist)], alphas_hist,
             color='#FFE66D', lw=1.5, where='post', label='Cone angle α')
    ax2.set_xlabel('Orbit number', color='#e8e8e8', fontsize=11)
    ax2.set_ylabel('α correction  [°]', color='#e8e8e8', fontsize=11)
    ax2.tick_params(colors='#e8e8e8')
    for sp in ax2.spines.values():
        sp.set_color('#555')
    ax2.legend(facecolor='#1a1a2e', labelcolor='#e8e8e8',
               framealpha=0.85, fontsize=10)

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches='tight', facecolor=_DARK)
    plt.close(fig)
    if verbose:
        print(f"  ✓  Saved → {output}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # Strip shell comment fragments passed as args (e.g. #…)
    _args = [a for a in sys.argv[1:] if not a.startswith('#')]
    which = _args[0] if _args else 'all'

    print("Computing β-sweep (this takes ~2 min) …")
    sweep = _compute_sweep(verbose=True)

    if which in ('fig1', 'all'):
        print("\n── Figure 1: β-family halo orbits ───────────────")
        fig_beta_family(sweep)

    if which in ('fig2', 'all'):
        print("\n── Figure 2: stability sweep ─────────────────────")
        fig_stability_sweep(sweep)

    if which in ('fig3', 'all'):
        print("\n── Figure 3: Floquet multipliers ─────────────────")
        fig_floquet(sweep)

    if which in ('fig4', 'all'):
        print("\n── Figure 4: reachable-set evolution ─────────────")
        fig_reachable_evolution(sweep)

    if which in ('fig5', 'all'):
        print("\n── Figure 5–7: LQR station-keeping ──────────────")
        fig_stationkeeping(output_prefix='fig5')

    if which in ('fig5sk', 'all'):
        print("\n── Figure 5 (SK): Halo orbit station-keeping ────")
        fig_stationkeeping_halo()

    if which in ('fig6', 'fig7', 'heteroclinic', 'all'):
        print("\n── Figures 6–7: Earth-Moon heteroclinic ──────────")
        from src.heteroclinic import fig_poincare_map, fig_manifold_transfer
        cache = fig_poincare_map()
        fig_manifold_transfer(cache=cache)

    print("\nAll figures written.  Open the PNGs for inspection.")
