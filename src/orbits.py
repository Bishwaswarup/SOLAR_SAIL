# Richardson guess + correction (adapted)
import numpy as np
from scipy.integrate import solve_ivp
from .dynamics import cr3bp_sail_eom


# ── Jacobians ──────────────────────────────────────────────────────────────────

def _jacobian_analytic(state, mu, alpha=0.0, delta=0.0, beta=0.0):
    """
    6×6 Jacobian of the CR3BP + sail EOM.

    Gravity terms are computed analytically.  Sail terms (∂a_sail/∂r) are
    added via three finite-difference calls on sail_acceleration when beta > 0.
    This keeps the analytic structure for the dominant gravity part while
    correctly handling any (alpha, delta, beta) combination for the sail.

    Omitting the sail Jacobian (beta=0 default) is fine for low-β designs
    (< ~0.05) but causes the differential corrector to oscillate for β ~ 0.5
    because the sail force is then comparable to solar gravity.
    """
    from .dynamics import sail_acceleration  # relative import — works as src package

    x, y, z = state[0], state[1], state[2]
    r1  = np.sqrt((x + mu)**2      + y**2 + z**2)
    r2  = np.sqrt((x - (1-mu))**2  + y**2 + z**2)
    r1_3, r1_5 = r1**3, r1**5
    r2_3, r2_5 = r2**3, r2**5

    # Second partial derivatives of the gravitational effective potential
    Oxx = (1
           - (1-mu)/r1_3 + 3*(1-mu)*(x+mu)**2/r1_5
           - mu/r2_3      + 3*mu*(x-(1-mu))**2/r2_5)
    Oyy = (1
           - (1-mu)/r1_3 + 3*(1-mu)*y**2/r1_5
           - mu/r2_3      + 3*mu*y**2/r2_5)
    Ozz = (- (1-mu)/r1_3 + 3*(1-mu)*z**2/r1_5
           -  mu/r2_3     + 3*mu*z**2/r2_5)
    Oxy = (3*(1-mu)*(x+mu)*y/r1_5 + 3*mu*(x-(1-mu))*y/r2_5)
    Oxz = (3*(1-mu)*(x+mu)*z/r1_5 + 3*mu*(x-(1-mu))*z/r2_5)
    Oyz = (3*(1-mu)*y*z/r1_5       + 3*mu*y*z/r2_5)

    J = np.array([
        [0.,   0.,   0.,   1.,  0.,  0.],
        [0.,   0.,   0.,   0.,  1.,  0.],
        [0.,   0.,   0.,   0.,  0.,  1.],
        [Oxx,  Oxy,  Oxz,  0.,  2.,  0.],
        [Oxy,  Oyy,  Oyz, -2.,  0.,  0.],
        [Oxz,  Oyz,  Ozz,  0.,  0.,  0.],
    ])

    # ── Sail contribution ∂a_sail/∂r (columns 0–2, rows 3–5) ────────────────
    # The sail force does not depend on velocity, so only the position columns
    # of J[3:6, 0:3] need to be updated.
    if beta != 0.0:
        eps = 1e-7
        s = list(state)
        a0 = np.array(sail_acceleration(s, alpha, delta, beta, mu))
        for i in range(3):          # differentiate w.r.t. x, y, z
            sp = list(state); sp[i] += eps
            da = (np.array(sail_acceleration(sp, alpha, delta, beta, mu)) - a0) / eps
            J[3, i] += da[0]        # ∂(ẍ)/∂rᵢ
            J[4, i] += da[1]        # ∂(ÿ)/∂rᵢ
            J[5, i] += da[2]        # ∂(z̈)/∂rᵢ

    return J


def _jacobian_numerical(t, state, alpha, delta, beta, mu, eps=1e-7):
    """Numerical Jacobian — kept for validation only; not used in the hot path."""
    f0 = np.array(cr3bp_sail_eom(t, state, alpha, delta, beta, mu))
    J  = np.zeros((6, 6))
    for i in range(6):
        sp = state.copy(); sp[i] += eps
        J[:, i] = (np.array(cr3bp_sail_eom(t, sp, alpha, delta, beta, mu)) - f0) / eps
    return J


def _eom_stm(t, sv, alpha, delta, beta, mu):
    """Augmented EOM for state + STM (42 variables).

    Uses the analytic gravity Jacobian plus the sail Jacobian (3 extra
    sail_acceleration calls) when beta != 0.  This is exact for any (α, δ, β)
    and is critical for convergence at high β (e.g. β = 0.5).
    """
    state = sv[:6]
    Phi   = sv[6:].reshape(6, 6)
    dstate = np.array(cr3bp_sail_eom(t, state, alpha, delta, beta, mu))
    A      = _jacobian_analytic(state, mu, alpha, delta, beta)
    dPhi   = A @ Phi
    return np.concatenate([dstate, dPhi.flatten()])


# ── Richardson guess ───────────────────────────────────────────────────────────

def _richardson_guess(eq_pos, Az, mu, alpha=0.0, delta=0.0, beta=0.0):
    """
    Returns an approximate initial state [x,y,z,vx,vy,vz] and half-period T/2
    for a northern halo orbit of z-amplitude Az around eq_pos.

    Uses the analytic Jacobian evaluated at the equilibrium point to compute
    the effective c2, so sail-displaced artificial equilibria are handled
    correctly (gravity-only c2 is wrong when the sail modifies the curvature).
    """
    state_eq = np.array([eq_pos[0], eq_pos[1], eq_pos[2], 0., 0., 0.])
    J  = _jacobian_analytic(state_eq, mu)   # fast analytic at equilibrium
    # ∂ax/∂x at a collinear L-point = 1 + 2·c2  →  c2 = (J[3,0] − 1) / 2
    c2 = (J[3, 0] - 1.0) / 2.0

    if c2 <= 0:
        raise ValueError(
            f"Effective c2={c2:.4f} at equilibrium is non-positive. "
            "Check eq_pos and sail parameters."
        )

    # In-plane oscillatory frequency  ωp
    lambda_sq = ((c2 - 2.0) - np.sqrt(9.0*c2**2 - 8.0*c2)) / 2.0
    omega_p   = np.sqrt(-lambda_sq)

    # Amplitude ratio  κ
    kappa = (omega_p**2 + 1.0 + 2.0*c2) / (2.0*omega_p)
    Ax    = Az / kappa

    state0 = np.array([
        eq_pos[0] - Ax,   # x₀ (minimum-x crossing)
        0.,               # y₀ = 0
        Az,               # z₀ = Az
        0.,               # vx₀ = 0
        kappa*omega_p*Ax, # vy₀ > 0  (northern halo)
        0.,               # vz₀ = 0
    ])
    T_half = np.pi / omega_p

    return state0, T_half


# ── Differential corrector ────────────────────────────────────────────────────

def compute_halo_orbit(eq_pos: list, Az: float, mu: float,
                       alpha: float, delta: float, beta: float,
                       x0: list = None) -> tuple:
    """
    Computes a periodic halo orbit for a solar sail around an artificial
    equilibrium point in the CR3BP using differential correction.

    Uses a 3-variable Newton corrector:
        free:         [x₀, vy₀, T_half]
        constraints:  [vx_f = 0, vz_f = 0, y_f = 0]

    Integrates for a FIXED time T each iteration (no event detection), so the
    corrector converges even when the Richardson guess has an unstable component
    (typical for Sun-Earth L1 whose hyperbolic eigenvalue is ~2.5 / TU).

    Parameters
    ----------
    eq_pos : list[float]   3-element equilibrium position [x*, y*, z*].
    Az     : float         Out-of-plane (z) amplitude in non-dim units.
    mu     : float         CR3BP mass parameter.
    alpha  : float         Sail cone angle (rad).
    delta  : float         Sail clock angle (rad).
    beta   : float         Sail lightness number.
    x0     : list, optional  External initial guess [x,y,z,vx,vy,vz].

    Returns
    -------
    (state0, period) : (ndarray shape (6,), float)
        Converged initial state and full non-dimensional period.
    """
    state0_rich, T_half = _richardson_guess(eq_pos, Az, mu, alpha, delta, beta)
    state0 = np.array(x0, dtype=float) if x0 is not None else state0_rich.copy()
    T      = T_half          # floating half-period

    for iteration in range(50):
        # Build 42-element vector  [state | Φ₀ = I (flattened)]
        sv0 = np.concatenate([state0, np.eye(6).flatten()])

        # Integrate for exactly T  (fixed time — no event)
        res = solve_ivp(
            _eom_stm,
            [0.0, T],
            sv0,
            args=(alpha, delta, beta, mu),
            method='DOP853',
            rtol=1e-9,
            atol=1e-9,
        )

        sv_f    = res.y[:, -1]
        state_f = sv_f[:6]
        Phi_f   = sv_f[6:].reshape(6, 6)

        vx_f = state_f[3]   # want → 0
        vz_f = state_f[5]   # want → 0
        y_f  = state_f[1]   # want → 0

        if abs(vx_f) + abs(vz_f) + abs(y_f) < 1e-10:
            return state0, 2.0 * T

        # EOM at final state (∂/∂T column)
        sdot_f = np.array(cr3bp_sail_eom(T, state_f, alpha, delta, beta, mu))

        # 3×3 Newton matrix
        # M · [δx₀, δvy₀, δT] = −[vx_f, vz_f, y_f]
        M = np.array([
            [Phi_f[3, 0], Phi_f[3, 4], sdot_f[3]],   # δvx_f
            [Phi_f[5, 0], Phi_f[5, 4], sdot_f[5]],   # δvz_f
            [Phi_f[1, 0], Phi_f[1, 4], sdot_f[1]],   # δy_f
        ])

        corr     = np.linalg.solve(M, [-vx_f, -vz_f, -y_f])

        # Step-size guard on T: never let a single Newton step change T by
        # more than 40 % of its current value, and never let T go below 10 % of
        # the Richardson half-period.  Without this, a poor initial guess can
        # push T to ≈ 0 on the first iteration and the orbit collapses.
        max_dT = 0.40 * abs(T)
        if abs(corr[2]) > max_dT:
            corr *= max_dT / abs(corr[2])

        state0[0] += corr[0]   # x₀
        state0[4] += corr[1]   # vy₀
        T         += corr[2]   # T_half
        T          = max(T, 0.10 * T_half)   # floor: can't undershoot by >90%

    raise RuntimeError(
        f"Differential corrector did not converge after 50 iterations. "
        f"Last residual: vx={vx_f:.2e}, vz={vz_f:.2e}, y={y_f:.2e}"
    )
