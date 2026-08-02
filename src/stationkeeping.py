"""
stationkeeping.py — LQR station-keeping for the solar-storm warning sentinel.

Physical setup
──────────────
A solar sail is parked at an artificial equilibrium point sunward of the
classical Sun-Earth L₁.  The sail's radiation-pressure force exactly balances
the gravity gradient, holding the spacecraft closer to the Sun than SOHO/DSCOVR
and providing ~2 h advance warning instead of ~45 min.

The problem: these sub-L₁ equilibria are highly unstable (λ_u ~ 100 at β=0.05).
Without active control the spacecraft escapes on a timescale of days.

Solution: use the sail itself as the actuator.  By tilting the membrane slightly
(changing α and δ in real time) the sail produces a corrective acceleration.
We design the feedback gain using a Linear Quadratic Regulator (LQR) applied to
the linearised CR3BP dynamics around the equilibrium.

Key results (paper Section IV)
──────────────────────────────
  1. Minimum β for stabilisation — smallest sail that can outrun the instability.
  2. Control authority margin — reachable |u| vs required |u|.
  3. 30-day closed-loop simulation — position error and fuel-equivalent ΔV.
  4. Sensitivity — how Q/R weighting trades accuracy vs control effort.

Usage (from project root)
──────────────────────────
    python src/stationkeeping.py           # all figures
    python src/stationkeeping.py lqr       # LQR + minimum-β figure only
    python src/stationkeeping.py sim       # 30-day simulation only

Importable API
──────────────
    from src.stationkeeping import (
        lqr_gain, closed_loop_eigenvalues,
        minimum_beta_for_stability, simulate_stationkeeping,
        fig_minimum_beta, fig_simulation, fig_control_authority)
"""

import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.linalg   import solve_continuous_are, eigvals
from scipy.integrate import solve_ivp

from src.equilibria import find_artificial_equilibrium
from src.orbits     import _jacobian_analytic
from src.sail_control import reachable_set
from src.dynamics   import cr3bp_sail_eom

# ── Constants ──────────────────────────────────────────────────────────────────
MU              = 3.003e-6
AU_KM           = 1.496e8
DAYS_PER_NONDIM = 365.25 / (2 * np.pi)
VEL_KMS         = 29.7847            # 1 non-dim vel = 29.78 km/s

# B matrix: sail adds acceleration to the last three state components (ẍ, ÿ, z̈)
# 6×3 — control is a 3-vector [uₓ, u_y, u_z]
B_CTRL = np.zeros((6, 3))
B_CTRL[3, 0] = 1.0
B_CTRL[4, 1] = 1.0
B_CTRL[5, 2] = 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Core LQR functions
# ─────────────────────────────────────────────────────────────────────────────

def lqr_gain(A, B=None, Q=None, R=None):
    """
    Solve the continuous-time LQR problem:
        min ∫ (x'Qx + u'Ru) dt   subject to  ẋ = Ax + Bu

    Returns
    -------
    K : ndarray (m, n)   Gain matrix;  u* = −K x
    P : ndarray (n, n)   Riccati solution
    eigs_cl : ndarray    Closed-loop eigenvalues of (A − BK)
    """
    if B is None:
        B = B_CTRL
    n = A.shape[0]
    m = B.shape[1]
    if Q is None:
        Q = np.diag([1e4, 1e4, 1e4, 1.0, 1.0, 1.0])   # penalise position more
    if R is None:
        R = np.eye(m) * 1.0                             # equal control cost

    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.solve(R, B.T @ P)
    eigs_cl = eigvals(A - B @ K)
    return K, P, eigs_cl


def closed_loop_eigenvalues(eq_pos, beta, alpha=0.0, delta=0.0,
                             Q=None, R=None):
    """
    Compute open-loop and closed-loop eigenvalues for a given β.

    Returns
    -------
    eigs_ol  : open-loop  eigenvalues of A
    eigs_cl  : closed-loop eigenvalues of (A − BK)
    K        : LQR gain matrix
    a_max    : maximum sail acceleration magnitude at this equilibrium [non-dim]
    """
    state_eq = np.array([eq_pos[0], eq_pos[1], eq_pos[2], 0., 0., 0.])
    A = _jacobian_analytic(state_eq, MU, alpha, delta, beta)

    eigs_ol = eigvals(A)

    try:
        K, _, eigs_cl = lqr_gain(A, Q=Q, R=R)
    except np.linalg.LinAlgError:
        # Riccati solver failed (system not stabilisable at this β)
        K      = None
        eigs_cl = np.full(6, np.nan, dtype=complex)

    # maximum sail acceleration at this equilibrium (face-on, β)
    r1 = abs(eq_pos[0] + MU)
    a_max = beta * (1 - MU) / r1**2

    return eigs_ol, eigs_cl, K, a_max


def minimum_beta_for_stability(alpha=0.0, delta=0.0,
                                betas=None, Q=None, R=None,
                                verbose=True):
    """
    Sweep β and find the smallest value for which the LQR closed-loop system
    is fully stable (all eigenvalue real parts < 0).

    Returns
    -------
    results : list of dicts, one per β
    beta_min : float   smallest stable β found (or np.nan)
    """
    if betas is None:
        betas = np.linspace(0.005, 0.15, 30)

    results = []
    x_guess = 0.99

    for b in betas:
        try:
            eq = find_artificial_equilibrium(alpha, delta, b, MU,
                                             [x_guess - 0.01, 0.0, 0.0])
            x_guess = eq[0]
            eigs_ol, eigs_cl, K, a_max = closed_loop_eigenvalues(
                eq, b, alpha, delta, Q=Q, R=R)

            ol_max_re  = np.max(eigs_ol.real)
            cl_max_re  = np.max(eigs_cl.real) if K is not None else np.nan
            stable     = (K is not None) and (cl_max_re < -1e-10)

            # required control: rough estimate = λ_u * perturbation
            # here we store the ratio a_max / ol_max_re as "authority margin"
            margin = a_max / ol_max_re if ol_max_re > 0 else np.inf

            results.append(dict(
                beta     = b,
                x_eq     = eq[0],
                shift_km = (0.990027 - eq[0]) * AU_KM,   # sunward of classical L₁
                ol_max_re = ol_max_re,
                cl_max_re = cl_max_re,
                stable    = stable,
                a_max     = a_max,
                margin    = margin,
                K         = K,
                eq        = eq,
            ))
            if verbose:
                tag = '✓ STABLE' if stable else '✗ unstable'
                print(f"  β={b:.3f}  x_eq={eq[0]:.5f}"
                      f"  λ_u={ol_max_re:.3f}  cl_max_re={cl_max_re:.3f}"
                      f"  {tag}")
        except Exception as exc:
            if verbose:
                print(f"  β={b:.3f}  SKIP ({exc})")

    stable_betas = [r['beta'] for r in results if r['stable']]
    beta_min = min(stable_betas) if stable_betas else np.nan
    return results, beta_min


# ─────────────────────────────────────────────────────────────────────────────
# Closed-loop simulation
# ─────────────────────────────────────────────────────────────────────────────

def simulate_stationkeeping(beta, duration_days=30.0,
                             perturbation_km=100.0,
                             alpha=0.0, delta=0.0,
                             Q=None, R=None,
                             verbose=True):
    """
    Simulate 30 days of LQR station-keeping around the sub-L₁ sentinel orbit.

    The spacecraft starts with a 100 km displacement in x (representative of
    an insertion error or unmodelled solar-wind push) and the LQR controller
    drives it back.

    The full nonlinear CR3BP EOM are integrated; the LQR gain (computed from
    the linearised dynamics at the equilibrium) provides the feedback.

    Parameters
    ----------
    beta           : float  Sail lightness number.
    duration_days  : float  Simulation duration in days.
    perturbation_km: float  Initial x-displacement [km].
    alpha, delta   : float  Nominal sail angles [rad].
    Q, R           : ndarrays  LQR weight matrices (defaults used if None).
    verbose        : bool

    Returns
    -------
    dict with keys:
        t_days, pos_error_km, vel_error_ms, control_accel_ms2,
        delta_v_ms (cumulative), eq, K, stable
    """
    # ── equilibrium ───────────────────────────────────────────────────────────
    if verbose:
        print(f"  Finding equilibrium for β={beta:.3f} …", end=' ', flush=True)
    eq = find_artificial_equilibrium(alpha, delta, beta, MU,
                                     [0.99 - beta * 0.5, 0.0, 0.0])
    x_eq = np.array([eq[0], 0.0, 0.0, 0.0, 0.0, 0.0])
    if verbose:
        print(f"done  x*={eq[0]:.6f}")

    # ── LQR gain ──────────────────────────────────────────────────────────────
    _, eigs_cl, K, a_max = closed_loop_eigenvalues(eq, beta, alpha, delta, Q=Q, R=R)
    if K is None:
        raise RuntimeError(
            f"LQR Riccati solver failed for β={beta:.3f}. "
            "Increase β or relax Q/R.")
    if np.max(eigs_cl.real) > 0:
        raise RuntimeError(
            f"Closed-loop system unstable at β={beta:.3f} "
            f"(max Re(λ_cl) = {np.max(eigs_cl.real):.4f}). "
            "Increase β or tune Q/R.")

    if verbose:
        shift_km = (0.990027 - eq[0]) * AU_KM
        print(f"  Sunward shift = {shift_km:,.0f} km  "
              f"  a_max = {a_max:.5f} [non-dim]")
        print(f"  Closed-loop max Re(λ) = {np.max(eigs_cl.real):.5f}  "
              f"(all < 0 → stable)")

    # ── initial state: perturb x by perturbation_km ───────────────────────────
    dx0_nd = perturbation_km / AU_KM   # non-dim displacement
    state0  = x_eq.copy()
    state0[0] += dx0_nd

    # ── maximum control magnitude (1 non-dim accel = VEL_KMS/DAYS_PER_NONDIM km/s per day)
    u_max = a_max   # non-dim; sail can provide at most this

    # ── EOM with LQR feedback ─────────────────────────────────────────────────
    def eom_controlled(t, state):
        # CR3BP + sail at nominal (α, δ, β) + LQR correction
        dx = state - x_eq
        u  = -K @ dx                          # LQR control [non-dim accel]
        # Saturate at physical maximum
        u_norm = np.linalg.norm(u)
        if u_norm > u_max:
            u = u * (u_max / u_norm)

        dstate = np.array(cr3bp_sail_eom(t, state, alpha, delta, beta, MU))
        dstate[3] += u[0]
        dstate[4] += u[1]
        dstate[5] += u[2]
        return dstate

    # ── integrate ─────────────────────────────────────────────────────────────
    T_nd     = duration_days / DAYS_PER_NONDIM
    t_eval   = np.linspace(0, T_nd, int(duration_days * 48))  # every 30 min

    if verbose:
        print(f"  Integrating {duration_days:.0f} days …", end=' ', flush=True)

    res = solve_ivp(eom_controlled, [0, T_nd], state0,
                    method='DOP853', rtol=1e-9, atol=1e-9,
                    t_eval=t_eval, dense_output=False)

    if verbose:
        print("done")

    t_days   = res.t * DAYS_PER_NONDIM
    pos_nd   = res.y[:3, :] - x_eq[:3, np.newaxis]   # position error [non-dim]
    vel_nd   = res.y[3:, :] - x_eq[3:, np.newaxis]   # velocity error [non-dim]

    pos_km   = np.linalg.norm(pos_nd, axis=0) * AU_KM
    vel_ms   = np.linalg.norm(vel_nd, axis=0) * VEL_KMS * 1e3   # m/s

    # Control acceleration at each time step (re-evaluate from trajectory)
    ctrl_nd  = np.zeros(len(t_days))
    for i in range(len(t_days)):
        dx = res.y[:, i] - x_eq
        u  = -K @ dx
        u_norm = np.linalg.norm(u)
        if u_norm > u_max:
            u = u * (u_max / u_norm)
        ctrl_nd[i] = np.linalg.norm(u)

    ctrl_ms2 = ctrl_nd * VEL_KMS * 1e3 / DAYS_PER_NONDIM / 86400  # m/s²

    # Cumulative ΔV = ∫|u| dt  (trapezoidal)
    dt_s     = np.diff(t_days) * 86400    # seconds between steps
    dv_ms    = np.concatenate([[0],
                np.cumsum(0.5 * (ctrl_ms2[:-1] + ctrl_ms2[1:]) * dt_s)])

    if verbose:
        print(f"  Final pos error : {pos_km[-1]:.2f} km")
        print(f"  Total ΔV        : {dv_ms[-1]:.4f} m/s  "
              f"(over {duration_days:.0f} days)")

    return dict(
        t_days         = t_days,
        pos_error_km   = pos_km,
        vel_error_ms   = vel_ms,
        control_accel_ms2 = ctrl_ms2,
        delta_v_ms     = dv_ms,
        eq             = eq,
        K              = K,
        a_max          = a_max,
        eigs_cl        = eigs_cl,
        beta           = beta,
        stable         = True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Paper figures
# ─────────────────────────────────────────────────────────────────────────────

_STYLE = dict(figure_facecolor='white', axes_facecolor='white',
              axes_grid=True, grid_color='#dddddd', grid_linewidth=0.5,
              font_size=10, axes_labelsize=10, axes_titlesize=11,
              legend_fontsize=8)


def fig_minimum_beta(results, beta_min,
                     output='fig5_minimum_beta.png', verbose=True):
    """
    Three-panel figure:
      Left  — open-loop unstable eigenvalue Re(λ_u) vs β
      Centre — closed-loop max Re(λ) vs β  (crosses zero at β_min)
      Right  — sunward shift of equilibrium vs β  (contextualises β_min)
    """
    betas    = np.array([r['beta']      for r in results])
    ol_re    = np.array([r['ol_max_re'] for r in results])
    cl_re    = np.array([r['cl_max_re'] for r in results])
    shift_km = np.array([r['shift_km']  for r in results])
    stable   = np.array([r['stable']    for r in results])

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    ax_ol, ax_cl, ax_sh = axes

    # ── Left: open-loop instability ───────────────────────────────────────────
    ax_ol.semilogy(betas, ol_re, 'b-o', ms=4, lw=1.5)
    ax_ol.set_xlabel('β  (sail lightness)')
    ax_ol.set_ylabel('max Re(λ)  open-loop  [non-dim/TU]')
    ax_ol.set_title('Open-loop instability growth rate')
    ax_ol.axvline(beta_min, color='red', ls='--', lw=1.0, alpha=0.7,
                  label=f'β_min = {beta_min:.3f}')
    ax_ol.legend()

    # ── Centre: closed-loop stability ─────────────────────────────────────────
    valid = np.isfinite(cl_re)
    ax_cl.plot(betas[valid], cl_re[valid], 'g-o', ms=4, lw=1.5)
    ax_cl.axhline(0, color='black', lw=0.8, ls='--', alpha=0.5)
    ax_cl.axvline(beta_min, color='red', ls='--', lw=1.0, alpha=0.7,
                  label=f'β_min = {beta_min:.3f}')
    ax_cl.fill_between(betas[valid & stable], cl_re[valid & stable],
                       0, alpha=0.15, color='green', label='stable region')
    ax_cl.set_xlabel('β  (sail lightness)')
    ax_cl.set_ylabel('max Re(λ)  closed-loop')
    ax_cl.set_title('LQR closed-loop stability')
    ax_cl.legend()

    # ── Right: equilibrium shift ───────────────────────────────────────────────
    ax_sh.plot(betas, shift_km / 1e6, 'purple', lw=1.5, marker='o', ms=4)
    ax_sh.axvline(beta_min, color='red', ls='--', lw=1.0, alpha=0.7,
                  label=f'β_min ≈ {beta_min:.3f}')
    # annotate SOHO/DSCOVR reference
    ax_sh.axhline(0, color='grey', lw=0.5, ls=':', alpha=0.5)
    ax_sh.text(betas[-1]*0.92, 0.3,
               'classical L₁\n(SOHO/DSCOVR)', ha='right',
               fontsize=7.5, color='grey')
    ax_sh.set_xlabel('β  (sail lightness)')
    ax_sh.set_ylabel('Sunward shift from classical L₁  [10⁶ km]')
    ax_sh.set_title('Warning-time gain vs β')
    ax_sh.legend()

    # secondary axis: warning time gain
    # classical L₁ ≈ 1.5×10⁶ km from Earth → 45 min warning at solar-wind speed
    # extra shift / 750 km s⁻¹ = extra seconds, /60 = extra minutes
    def shift_to_extra_min(shift_mkm):
        return shift_mkm * 1e6 / (750 * 60)  # 750 km/s solar wind

    ax_sh2 = ax_sh.twinx()
    ax_sh2.set_ylabel('Extra warning time  [min]  (750 km/s solar wind)',
                      color='darkorange')
    ax_sh2.tick_params(axis='y', labelcolor='darkorange')
    y2 = shift_to_extra_min(shift_km / 1e6)
    ax_sh2.set_ylim(shift_to_extra_min(0), shift_to_extra_min(shift_km.max() / 1e6))

    fig.suptitle(
        'Minimum sail lightness for autonomous sub-L₁ station-keeping\n'
        '(Sun-Earth CR3BP,  LQR feedback via sail attitude control)',
        fontsize=10, y=1.02)
    plt.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches='tight')
    plt.close(fig)
    if verbose:
        print(f"  ✓  Saved → {output}")
    return fig


def fig_simulation(sim, output='fig6_simulation.png', verbose=True):
    """
    Four-panel figure showing the 30-day controlled station-keeping arc:
      (A) Position error [km]
      (B) Control acceleration [mm/s²]
      (C) Cumulative ΔV [m/s]
      (D) Phase portrait in x–vx  (perturbation space)
    """
    fig = plt.figure(figsize=(12, 9))
    gs  = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.35)
    ax_pos  = fig.add_subplot(gs[0, 0])
    ax_ctrl = fig.add_subplot(gs[0, 1])
    ax_dv   = fig.add_subplot(gs[1, 0])
    ax_ph   = fig.add_subplot(gs[1, 1])

    td  = sim['t_days']
    pos = sim['pos_error_km']
    vel = sim['vel_error_ms']
    acc = sim['control_accel_ms2'] * 1e3   # → mm/s²
    dv  = sim['delta_v_ms']
    b   = sim['beta']

    # ── A: position error ─────────────────────────────────────────────────────
    ax_pos.semilogy(td, np.maximum(pos, 1e-3), color='steelblue', lw=1.2)
    ax_pos.set_xlabel('Time  [days]')
    ax_pos.set_ylabel('|δr|  [km]')
    ax_pos.set_title(f'(A)  Position error  (β = {b:.3f})')
    ax_pos.axhline(1.0, color='grey', ls='--', lw=0.8, alpha=0.6,
                   label='1 km threshold')
    ax_pos.legend(fontsize=8)

    # ── B: control acceleration ───────────────────────────────────────────────
    ax_ctrl.plot(td, acc, color='darkorange', lw=1.0)
    a_max_mm = sim['a_max'] * VEL_KMS * 1e3 / DAYS_PER_NONDIM / 86400 * 1e3
    ax_ctrl.axhline(a_max_mm, color='red', ls='--', lw=0.9,
                    label=f'a_max = {a_max_mm:.3f} mm/s²')
    ax_ctrl.set_xlabel('Time  [days]')
    ax_ctrl.set_ylabel('|u|  [mm/s²]')
    ax_ctrl.set_title('(B)  Control acceleration  (sail tilt)')
    ax_ctrl.legend(fontsize=8)

    # ── C: cumulative ΔV ─────────────────────────────────────────────────────
    ax_dv.plot(td, dv, color='seagreen', lw=1.2)
    ax_dv.set_xlabel('Time  [days]')
    ax_dv.set_ylabel('Cumulative ΔV  [m/s]')
    ax_dv.set_title('(C)  Fuel-equivalent ΔV  (propellantless)')
    ax_dv.text(td[-1] * 0.05, dv[-1] * 0.85,
               f'Total: {dv[-1]:.4f} m/s\n(attitude-only, no propellant)',
               fontsize=8, color='seagreen',
               bbox=dict(boxstyle='round,pad=0.3', fc='#eeffee', ec='none'))

    # ── D: phase portrait δx vs δvx ───────────────────────────────────────────
    ax_ph.plot(pos, vel, color='#9933cc', lw=0.8, alpha=0.8)
    ax_ph.scatter([pos[0]], [vel[0]], color='blue',  s=50, zorder=5,
                  label='start')
    ax_ph.scatter([pos[-1]], [vel[-1]], color='red', s=50, zorder=5,
                  label='end')
    ax_ph.set_xlabel('|δr|  [km]')
    ax_ph.set_ylabel('|δv|  [m/s]')
    ax_ph.set_title('(D)  Phase portrait  (error space)')
    ax_ph.legend(fontsize=8)

    shift_km = (0.990027 - sim['eq'][0]) * AU_KM
    fig.suptitle(
        f'30-day LQR station-keeping simulation  '
        f'(β = {b:.3f},  sub-L₁ shift = {shift_km/1e6:.2f} × 10⁶ km)\n'
        f'Initial perturbation: 100 km in x;  '
        f'closed-loop max Re(λ) = {np.max(sim["eigs_cl"].real):.4f}',
        fontsize=10, y=1.01)

    fig.savefig(output, dpi=200, bbox_inches='tight')
    plt.close(fig)
    if verbose:
        print(f"  ✓  Saved → {output}")
    return fig


def fig_control_authority(results, output='fig7_control_authority.png',
                           verbose=True):
    """
    Control authority figure:
      Left  — a_max(β) vs the open-loop instability growth |λ_u|·ε
               (where ε = 100 km perturbation) — shows when sail force
               can outrun the unstable mode.
      Right  — margin = a_max / λ_u vs β  (> 1 means sail wins).
    """
    betas   = np.array([r['beta']      for r in results])
    ol_re   = np.array([r['ol_max_re'] for r in results])
    a_max   = np.array([r['a_max']     for r in results])
    margin  = np.array([r['margin']    for r in results])
    stable  = np.array([r['stable']    for r in results])

    fig, (ax_a, ax_m) = plt.subplots(1, 2, figsize=(10, 4.5))

    # ── Left: force race ──────────────────────────────────────────────────────
    eps_nd = 100 / AU_KM   # 100 km in non-dim
    ax_a.semilogy(betas, a_max, 'b-o', ms=4, lw=1.5, label='a_max(β)  [sail]')
    ax_a.semilogy(betas, ol_re * eps_nd, 'r--s', ms=4, lw=1.5,
                  label='λ_u · ε  [instability growth]')
    ax_a.set_xlabel('β  (sail lightness)')
    ax_a.set_ylabel('Acceleration  [non-dim]')
    ax_a.set_title('Sail force vs instability growth\n(100 km perturbation)')
    ax_a.legend()
    ax_a.fill_between(
        betas[stable], a_max[stable], ol_re[stable] * eps_nd,
        where=a_max[stable] > ol_re[stable] * eps_nd,
        alpha=0.15, color='green', label='sail wins')

    # ── Right: margin ─────────────────────────────────────────────────────────
    ax_m.plot(betas, margin, 'purple', lw=1.5, marker='o', ms=4)
    ax_m.axhline(1.0, color='red', ls='--', lw=1.0,
                 label='margin = 1  (breakeven)')
    ax_m.fill_between(betas, margin, 1,
                      where=margin > 1, alpha=0.15, color='green',
                      label='stable region')
    ax_m.set_xlabel('β  (sail lightness)')
    ax_m.set_ylabel('Control margin  =  a_max / λ_u')
    ax_m.set_title('Control authority margin')
    ax_m.legend()

    fig.suptitle(
        'Control authority of a solar sail for sub-L₁ station-keeping\n'
        '(margin > 1 means the sail can outrun the unstable eigenvalue)',
        fontsize=10, y=1.02)
    plt.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches='tight')
    plt.close(fig)
    if verbose:
        print(f"  ✓  Saved → {output}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    _args = [a for a in sys.argv[1:] if not a.startswith('#')]
    which = _args[0] if _args else 'all'

    if which in ('lqr', 'all'):
        print("\n── Minimum-β sweep ───────────────────────────────")
        results, beta_min = minimum_beta_for_stability(
            betas=np.linspace(0.005, 0.12, 24), verbose=True)
        print(f"\n  ★  Minimum β for LQR stability: {beta_min:.4f}")
        fig_minimum_beta(results, beta_min)
        fig_control_authority(results)

    if which in ('sim', 'all'):
        # Use β slightly above minimum for a clean simulation
        sim_beta = 0.05
        print(f"\n── 30-day simulation  β={sim_beta} ────────────────────")
        sim = simulate_stationkeeping(sim_beta, duration_days=30,
                                       perturbation_km=100, verbose=True)
        fig_simulation(sim)

    print("\nAll done.")