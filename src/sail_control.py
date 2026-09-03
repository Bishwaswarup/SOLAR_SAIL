# Cone/clock angle scheduling, optimal law
import numpy as np
from scipy.integrate import solve_ivp
from .dynamics import sail_acceleration, cr3bp_sail_eom, sail_frame


def optimal_sail_angles(state: list[float], target_dir: list[float], beta: float, mu: float) -> tuple[
    float, float, np.ndarray]:
    """
    Computes the optimal cone and clock angles (α*, δ*) to maximise the sail's
    acceleration component along a given target direction vector.

    This uses the closed-form analytical derivation from Colin McInnes (2004).

    Parameters
    ----------
    state : list[float] or np.ndarray
        The 6-element state vector [x, y, z, vx, vy, vz].
    target_dir : list[float] or np.ndarray
        The 3-element desired acceleration direction vector.
    beta, mu : float
        Lightness number and CR3BP mass parameter.

    Returns
    -------
    tuple[float, float, np.ndarray]
        (alpha_opt, delta_opt, a_sail_opt_vec)
    """

    x, y, z = state[0], state[1], state[2]

    # Normalise target direction
    d_hat = np.array(target_dir, dtype=float)
    d_norm = np.linalg.norm(d_hat)
    if d_norm < 1e-12:
        return 0.0, 0.0, np.zeros(3)
    d_hat = d_hat / d_norm

    # Build the local ORTHONORMAL sail frame {r_hat, p_hat, q_hat}.
    # Previously this used {r_hat, t_hat, k_hat} with k_hat the global z-axis,
    # which is not orthonormal off the ecliptic (r_hat . k_hat = z/r1) -- bug
    # A1.  Projecting a target direction onto a non-orthogonal triad gives
    # components that do not reconstruct the vector, so the "optimal" angles
    # were wrong off-plane.  See dynamics.sail_frame for the full statement.
    r_hat, p_hat, q_hat, r1 = sail_frame((x, y, z), mu)

    # Project target direction into the local frame
    d_r = np.dot(d_hat, r_hat)
    d_t = np.dot(d_hat, p_hat)
    d_k = np.dot(d_hat, q_hat)

    # Optimal Clock Angle (delta*)
    # delta* maximizes the transverse/normal projection: sin(delta)*d_k + cos(delta)*d_t
    delta_opt = np.arctan2(d_k, d_t)

    # Optimal Cone Angle (alpha*)
    A = d_r
    B = np.sqrt(d_t ** 2 + d_k ** 2)

    if B < 1e-12:
        # Target is purely radial.
        # If A > 0, point flat to sun. If A < 0, edge-on (no negative radial thrust possible).
        alpha_opt = 0.0 if A > 0 else np.pi / 2.0
    else:
        # Solve quadratic for tan(alpha): 2B*tan^2(a) + 3A*tan(a) - B = 0
        tan_alpha = (-3.0 * A + np.sqrt(9.0 * A ** 2 + 8.0 * B ** 2)) / (4.0 * B)
        # alpha is bounded between [0, pi/2]
        alpha_opt = np.arctan(max(0.0, tan_alpha))

    # Compute resulting acceleration vector
    a_opt = sail_acceleration(state, alpha_opt, delta_opt, beta, mu)

    return alpha_opt, delta_opt, a_opt


def station_keeping(state: list[float], state_ref: list[float], T_half: float, mu: float,
                    alpha: float, delta: float, beta: float, eps: float = 1e-5) -> tuple[float, float]:
    """
    Performs a single station-keeping correction at a y=0 crossing event.

    Uses finite-differencing to compute a 2x2 sensitivity matrix mapping
    changes in sail angles (Δα, Δδ) to errors in the terminal velocities (vx_f, vz_f)
    at the next half-period crossing.

    Parameters
    ----------
    state : list[float] or np.ndarray
        Current true spacecraft state at y=0 crossing.
    state_ref : list[float] or np.ndarray
        Target reference state at the current crossing.
    T_half : float
        Time until the next half-period crossing.
    mu, alpha, delta, beta : float
        CR3BP mass parameter and current nominal sail/spacecraft parameters.
    eps : float, optional
        Finite-difference perturbation step size, default 1e-5.

    Returns
    -------
    tuple[float, float]
        (alpha_new, delta_new): The updated sail control angles for the next half-orbit.
    """

    # Define the terminal event
    def y_crossing(t, sv, *args):
        return sv[1]
    y_crossing.terminal = True
    y_crossing.direction = -1 if state[4] > 0 else 1

    # Helper to integrate a given state and control to the next crossing
    def integrate_to_plane(s0, a_val, d_val):
        res = solve_ivp(
            cr3bp_sail_eom,
            [0, T_half * 1.5],
            s0,
            events=y_crossing,
            args=(a_val, d_val, beta, mu),
            rtol=1e-10,
            atol=1e-10
        )
        if not res.t_events[0].size:
            raise RuntimeError("Station-keeping trajectory failed to reach y=0 plane.")
        return res.y_events[0][0]

    # 1. Integrate nominal trajectory starting from the CURRENT state
    # (The error is baked into the initial conditions)
    state_f_nom = integrate_to_plane(state, alpha, delta)
    vx_f_nom = state_f_nom[3]
    vz_f_nom = state_f_nom[5]

    # Target terminal velocities are whatever the reference trajectory
    # dictates at the NEXT crossing. For a halo, this is [0, 0].
    # But to be robust, we assume the reference tells us where to aim:
    # Here, we assume the goal is perpendicular crossing:
    err_vx = -vx_f_nom
    err_vz = -vz_f_nom

    # 2. Perturb alpha
    state_f_alpha = integrate_to_plane(state, alpha + eps, delta)
    dvx_dalpha = (state_f_alpha[3] - vx_f_nom) / eps
    dvz_dalpha = (state_f_alpha[5] - vz_f_nom) / eps

    # 3. Perturb delta
    state_f_delta = integrate_to_plane(state, alpha, delta + eps)
    dvx_ddelta = (state_f_delta[3] - vx_f_nom) / eps
    dvz_ddelta = (state_f_delta[5] - vz_f_nom) / eps

    # 4. Form sensitivity matrix and solve for control changes
    S = np.array([
        [dvx_dalpha, dvx_ddelta],
        [dvz_dalpha, dvz_ddelta]
    ])

    err_vec = np.array([err_vx, err_vz])

    # Guard: if alpha ≈ 0, delta has no effect (sin(alpha)=0 ⟹ column 2 of S = 0).
    # In that case reduce to a 1-variable correction on alpha only.
    if alpha < 1e-4:
        da = err_vx / dvx_dalpha if abs(dvx_dalpha) > 1e-14 else 0.0
        alpha_new = np.clip(alpha + da, 0.0, np.pi / 2.0)
        return float(alpha_new), float(delta)

    try:
        delta_control = np.linalg.solve(S, err_vec)
    except np.linalg.LinAlgError:
        # Singular: fall back to alpha-only correction
        da = err_vx / dvx_dalpha if abs(dvx_dalpha) > 1e-14 else 0.0
        alpha_new = np.clip(alpha + da, 0.0, np.pi / 2.0)
        return float(alpha_new), float(delta)

    # Apply updates
    alpha_new = alpha + delta_control[0]
    delta_new = delta + delta_control[1]

    # Constrain alpha back to [0, pi/2] bounds (delta can freely wrap)
    alpha_new = np.clip(alpha_new, 0.0, np.pi / 2.0)

    return float(alpha_new), float(delta_new)


def reachable_set(state: list, beta: float, mu: float,
                  n_alpha: int = 30, n_delta: int = 36) -> np.ndarray:
    """
    Samples the full envelope of achievable sail accelerations at a given state
    by sweeping (alpha, delta) over a grid.

    The result is the "control bubble" — a closed surface in acceleration space
    that shows what the sail can and cannot achieve from this position.
    Key paper figure: compare the bubble at L1 vs. the sail-displaced L1.

    Parameters
    ----------
    state   : 6-element state vector [x, y, z, vx, vy, vz]
    beta    : lightness number
    mu      : CR3BP mass parameter
    n_alpha : number of cone-angle samples in [0, π/2]
    n_delta : number of clock-angle samples in [0, 2π)

    Returns
    -------
    np.ndarray, shape (n_alpha * n_delta, 3)
        Each row is one achievable acceleration vector [ax, ay, az].
    """
    alphas = np.linspace(0.0, np.pi / 2.0, n_alpha)
    deltas = np.linspace(0.0, 2.0 * np.pi, n_delta, endpoint=False)

    points = []
    for a in alphas:
        for d in deltas:
            points.append(sail_acceleration(state, a, d, beta, mu))

    return np.array(points)   # shape (n_alpha * n_delta, 3)
