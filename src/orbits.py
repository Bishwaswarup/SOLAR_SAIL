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

class HaloConvergenceError(RuntimeError):
    """The Newton corrector failed to converge."""


class HaloValidationError(RuntimeError):
    """The corrector converged, but to something that is not the requested halo."""


def _propagate_with_stm(state0, T, alpha, delta, beta, mu, n_dense=0,
                        rtol=1e-11, atol=1e-11):
    """
    Integrate state + STM to time T.  Returns (state_f, Phi_f, ts, Y).

    Cost note: for beta != 0 the STM's Jacobian adds three finite-difference
    sail_acceleration calls per RHS evaluation, so tightening rtol multiplies
    the step count *and* the per-step cost.  At beta ~ 0.5, rtol=1e-12 makes the
    Newton loop roughly an order of magnitude slower than rtol=1e-11 with no
    useful gain — the corrector tolerance is the binding constraint, not the
    integrator.  Keep rtol at or above the corrector's `tol`.
    """
    sv0 = np.concatenate([state0, np.eye(6).flatten()])
    res = solve_ivp(
        _eom_stm, [0.0, T], sv0,
        args=(alpha, delta, beta, mu),
        method='DOP853', rtol=rtol, atol=atol,
        dense_output=bool(n_dense),
    )
    sv_f = res.y[:, -1]
    ts = Y = None
    if n_dense:
        ts = np.linspace(0.0, T, n_dense)
        Y = res.sol(ts)[:6, :]
    return sv_f[:6], sv_f[6:].reshape(6, 6), ts, Y


def compute_halo_orbit(eq_pos: list, Az: float, mu: float,
                       alpha: float, delta: float, beta: float,
                       x0: list = None,
                       seed: tuple = None,
                       amplitude: str = 'z0',
                       tol: float = 1e-10,
                       max_iter: int = 60,
                       rtol: float = None,
                       validate: bool = True,
                       require_halo: bool = False,
                       retry_unseeded: bool = True,
                       verbose: bool = False,
                       return_info: bool = False):
    """
    Periodic halo orbit about an (artificial) collinear equilibrium in the CR3BP.

    Amplitude handling
    ──────────────────
    The initial state is always of the symmetric form

        [x0, 0, z0, 0, vy0, 0]

    so t = 0 sits on the x-z plane at a z-extremum (vz = 0).  Two conventions are
    offered for what `Az` means:

      amplitude='z0'   (default, the literature convention)
          z0 is PINNED to Az.  Free variables [x0, vy0, T_half]; constraints
          [vx_f, vz_f, y_f] = 0.  A 3x3 Newton solve.

      amplitude='max'
          The true out-of-plane amplitude max|z| over the orbit is driven to Az.
          Free variables [x0, z0, vy0, T_half]; constraints
          [vx_f, vz_f, y_f, max|z| - Az] = 0.  A 4x4 Newton solve.
          Because t=0 and t=T_half are both z-extrema, max|z| = max(|z0|, |z_f|).

    Why this matters.  In the previous implementation z0 was neither a free
    variable nor a constraint: `Az` only seeded the Richardson guess, and passing
    a full 6-state `x0` overwrote z0 as well, making `Az` a silent no-op.  Newton
    then walked to whichever periodic orbit was nearest in (x0, vy0, T) space,
    which for Earth-Moon L2 meant landing on a different branch (period 3.5195
    against the family's 3.4091) and reporting it as "Az = 0.02".

    Seeding
    ───────
    `seed=(x0, vy0, T_half)` supplies a continuation guess for the *free*
    variables only; z0 is always (re)set from `Az`.  The legacy `x0=` 6-state is
    still accepted — its x and vy are used, its z is ignored — so existing
    callers keep working while the no-op bug disappears.

    Validation
    ──────────
    With validate=True the converged orbit is checked for
      * Newton residual within `tol`
      * a sane period (T_half within a factor of 3 of the Richardson estimate)
      * genuine halo character: z must not change sign around the orbit
        (a sign change means a vertical-Lyapunov / planar branch, not a halo)
      * the requested amplitude actually achieved to 1 %
    and HaloValidationError is raised rather than silently returning a spurious
    orbit.

    Returns
    -------
    (state0, period)                     if return_info is False
    (state0, period, info)               if return_info is True

    `info` carries residual, n_iter, amplitude_achieved, z_min, z_max,
    T_half_richardson, and mode.
    """
    if amplitude not in ('z0', 'max'):
        raise ValueError("amplitude must be 'z0' or 'max'")

    # A continuation seed can be worse than no seed (it may sit in a different
    # family's basin).  Try it, then fall back to the pure Richardson guess.
    # Integrator tolerance: one decade tighter than the corrector tolerance is
    # enough to measure the residual; tighter than that only costs time (badly
    # so at high beta, where the sail Jacobian is finite-differenced).
    if rtol is None:
        rtol = min(1e-11, 0.1 * tol)

    if retry_unseeded and (seed is not None or x0 is not None):
        kw = dict(amplitude=amplitude, tol=tol, max_iter=max_iter, rtol=rtol,
                  validate=validate, require_halo=require_halo,
                  retry_unseeded=False, verbose=verbose,
                  return_info=return_info)
        try:
            return compute_halo_orbit(eq_pos, Az, mu, alpha, delta, beta,
                                      x0=x0, seed=seed, **kw)
        except (HaloConvergenceError, HaloValidationError):
            return compute_halo_orbit(eq_pos, Az, mu, alpha, delta, beta,
                                      x0=None, seed=None, **kw)

    state0_rich, T_half_rich = _richardson_guess(eq_pos, Az, mu, alpha, delta, beta)

    # ── assemble the initial iterate ────────────────────────────────────────
    state0 = state0_rich.copy()
    T = T_half_rich

    if seed is not None:
        state0[0] = float(seed[0])
        state0[4] = float(seed[1])
        if len(seed) > 2 and seed[2]:
            T = float(seed[2])
    elif x0 is not None:
        # Legacy path: take x and vy from the seed, but NEVER its z.
        xs = np.asarray(x0, dtype=float)
        state0[0] = xs[0]
        state0[4] = xs[4]

    # z0 is always the requested amplitude on entry, in both modes.
    state0[1] = state0[3] = state0[5] = 0.0
    state0[2] = float(Az)

    n_free = 3 if amplitude == 'z0' else 4
    residual = np.inf
    n_iter = 0

    for n_iter in range(1, max_iter + 1):
        state_f, Phi_f, _, _ = _propagate_with_stm(
            state0, T, alpha, delta, beta, mu, rtol=rtol, atol=rtol)

        vx_f, vz_f, y_f, z_f = state_f[3], state_f[5], state_f[1], state_f[2]
        F = [vx_f, vz_f, y_f]

        if amplitude == 'max':
            A_now = max(abs(state0[2]), abs(z_f))
            F = F + [A_now - Az]

        residual = float(np.max(np.abs(F)))
        if verbose:
            print(f"      it {n_iter:2d}  |F|={residual:.3e}  "
                  f"T_half={T:.6f}  x0={state0[0]:.8f}")
        if residual < tol:
            break

        sdot_f = np.array(cr3bp_sail_eom(T, state_f, alpha, delta, beta, mu))

        if amplitude == 'z0':
            #  [dx0, dvy0, dT]
            M = np.array([
                [Phi_f[3, 0], Phi_f[3, 4], sdot_f[3]],
                [Phi_f[5, 0], Phi_f[5, 4], sdot_f[5]],
                [Phi_f[1, 0], Phi_f[1, 4], sdot_f[1]],
            ])
        else:
            #  [dx0, dz0, dvy0, dT]
            if abs(state0[2]) >= abs(z_f):
                amp_row = [0.0, np.sign(state0[2]), 0.0, 0.0]
            else:
                s = np.sign(z_f)
                amp_row = [s * Phi_f[2, 0], s * Phi_f[2, 2],
                           s * Phi_f[2, 4], s * sdot_f[2]]
            M = np.array([
                [Phi_f[3, 0], Phi_f[3, 2], Phi_f[3, 4], sdot_f[3]],
                [Phi_f[5, 0], Phi_f[5, 2], Phi_f[5, 4], sdot_f[5]],
                [Phi_f[1, 0], Phi_f[1, 2], Phi_f[1, 4], sdot_f[1]],
                amp_row,
            ])

        try:
            corr = np.linalg.solve(M, -np.asarray(F, dtype=float))
        except np.linalg.LinAlgError:
            corr = np.linalg.lstsq(M, -np.asarray(F, dtype=float), rcond=None)[0]

        # Trust region on the period step: no single step may move T_half by
        # more than 40 %, and T_half may never fall below 10 % of Richardson.
        max_dT = 0.40 * abs(T)
        if abs(corr[-1]) > max_dT:
            corr = corr * (max_dT / abs(corr[-1]))

        state0[0] += corr[0]
        if amplitude == 'max':
            state0[2] += corr[1]
            state0[4] += corr[2]
        else:
            state0[4] += corr[1]
        T += corr[-1]
        T = max(T, 0.10 * T_half_rich)

    if residual >= tol:
        raise HaloConvergenceError(
            f"Corrector did not converge in {max_iter} iterations "
            f"(mode='{amplitude}', Az={Az:g}). Final residual {residual:.2e} "
            f"> tol {tol:.1e}."
        )

    period = 2.0 * T

    # ── post-convergence validation ─────────────────────────────────────────
    # Diagnostic pass for max|z| only; the amplitude check has a 1 % band, so a
    # loose tolerance is ample and keeps the high-beta cost down.
    _, _, _, Y = _propagate_with_stm(state0, period, alpha, delta, beta, mu,
                                     n_dense=400, rtol=1e-9, atol=1e-9)
    z_track = Y[2, :]
    z_min, z_max = float(z_track.min()), float(z_track.max())
    A_ach = float(np.max(np.abs(z_track)))

    # z at the two x-z plane crossings (t = 0 and t = T_half): both are
    # z-extrema.  A halo generically has |z0| != |z_f| and the two may differ in
    # sign — that is normal and is NOT a defect (e.g. the Sun-Earth Az=0.003
    # halo has z0 = +0.003000, z_f = -0.002311).  A z-symmetric orbit
    # (|z0| == |z_f|) is the signature of a vertical-Lyapunov-type branch.
    # state_f is the t = T_half state of the *converged* iterate (the loop exits
    # after evaluating it), so no extra propagation is needed here.
    z0_c = float(state0[2])
    zf_c = float(state_f[2])
    z_asym = abs(abs(z0_c) - abs(zf_c)) / max(abs(z0_c), 1e-15)

    info = dict(residual=residual, n_iter=n_iter, mode=amplitude,
                amplitude_requested=float(Az), amplitude_achieved=A_ach,
                z_min=z_min, z_max=z_max, z0=z0_c, z_half=zf_c,
                z_asymmetry=float(z_asym),
                z_symmetric=bool(z_asym < 1e-6),
                T_half=float(T), T_half_richardson=float(T_half_rich))

    if validate:
        if not (T_half_rich / 3.0 < T < T_half_rich * 3.0):
            raise HaloValidationError(
                f"Converged to T_half={T:.5f}, which is off the Richardson "
                f"estimate {T_half_rich:.5f} by more than 3x — almost certainly "
                f"a different branch, not the Az={Az:g} halo."
            )
        if A_ach < 0.1 * abs(Az):
            raise HaloValidationError(
                f"Out-of-plane motion collapsed: max|z| = {A_ach:.3e} against a "
                f"requested {abs(Az):.3e} — this is a planar (Lyapunov) branch."
            )
        if require_halo and z_asym > 5.0:
            raise HaloValidationError(
                f"z(T/2) = {zf_c:+.3e} is near zero against z0 = {z0_c:+.6f} "
                f"(asymmetry {z_asym:.1f}): the orbit crosses the z=0 plane at "
                f"the half-period crossing, so it is not a halo. Genuine halos "
                f"have asymmetry of order 0.1-0.5 (Earth-Moon L1 ~ 0.13, "
                f"L2 ~ 0.38)."
            )
        if require_halo and z_asym < 1e-6:
            raise HaloValidationError(
                f"Orbit is z-antisymmetric (z0 = {z0_c:+.6f}, "
                f"z(T/2) = {zf_c:+.6f}, asymmetry {z_asym:.1e}), so it is NOT a "
                f"halo: a halo has |z0| != |z(T/2)| (e.g. +0.003000 / -0.002311). "
                f"Two distinct things produce this signature — a "
                f"vertical-Lyapunov / axial branch (the corrector landed on the "
                f"wrong family), or genuine Keplerian degeneracy at high beta, "
                f"where no distinct halo family exists. Check `z_asymmetry` in "
                f"the returned info and pass require_halo=False if the "
                f"degenerate orbit is what you want."
            )
        # What "Az achieved" means depends on the convention in force.
        #   mode 'z0'  — Az is the x-z CROSSING value z0, pinned by construction.
        #                max|z| may legitimately exceed it, because z0 is only
        #                ONE of the orbit's two z-extrema.  (Earth-Moon L2:
        #                z0 = 0.010 gives max|z| = 0.0138.)
        #   mode 'max' — Az is the true out-of-plane amplitude max|z|.
        if amplitude == 'z0':
            if abs(z0_c - Az) > 1e-12 * max(abs(Az), 1.0):
                raise HaloValidationError(
                    f"z0 drifted off the pinned amplitude: requested {Az:.6g}, "
                    f"got {z0_c:.6g}."
                )
        else:
            if abs(A_ach - Az) > 0.01 * max(abs(Az), 1e-12):
                raise HaloValidationError(
                    f"Amplitude not achieved: requested max|z| = {Az:.6g}, got "
                    f"{A_ach:.6g}."
                )

    if return_info:
        return state0, period, info
    return state0, period


def halo_family(eq_pos: list, mu: float,
                az_grid,
                alpha: float = 0.0, delta: float = 0.0, beta: float = 0.0,
                amplitude: str = 'z0',
                require_halo: bool = True,
                verbose: bool = False) -> dict:
    """
    Track a halo family by natural-parameter continuation in Az.

    Each converged orbit supplies the (x0, vy0, T_half) seed for the next, with
    a linear tangent prediction once two points are in hand.  Non-convergent or
    invalid steps are skipped (and counted) rather than poisoning the track.

    Returns
    -------
    dict with arrays  Az, C, T, state0 (list of ndarray), plus n_failed/failures.
    """
    from .jacobi import jacobi_constant_sail  # local: keeps orbits.py import-light

    az_grid = np.asarray(az_grid, dtype=float)
    Az_ok, C_ok, T_ok, S_ok = [], [], [], []
    failures = []

    for Az in az_grid:
        if len(S_ok) >= 2:
            # linear tangent prediction in (x0, vy0, T_half)
            dA = Az_ok[-1] - Az_ok[-2]
            w = (Az - Az_ok[-1]) / dA if dA != 0 else 0.0
            pred = (S_ok[-1][0] + w * (S_ok[-1][0] - S_ok[-2][0]),
                    S_ok[-1][4] + w * (S_ok[-1][4] - S_ok[-2][4]),
                    T_ok[-1] / 2.0 + w * (T_ok[-1] - T_ok[-2]) / 2.0)
        elif S_ok:
            pred = (S_ok[-1][0], S_ok[-1][4], T_ok[-1] / 2.0)
        else:
            pred = None

        try:
            s0, P = compute_halo_orbit(eq_pos, float(Az), mu, alpha, delta, beta,
                                       seed=pred, amplitude=amplitude,
                                       require_halo=require_halo)
        except (HaloConvergenceError, HaloValidationError, ValueError) as e:
            failures.append((float(Az), type(e).__name__, str(e)[:70]))
            continue

        if T_ok and abs(P - T_ok[-1]) > 0.15 * T_ok[-1]:
            failures.append((float(Az), 'BranchJump',
                             f'period {P:.5f} vs {T_ok[-1]:.5f}'))
            continue

        Az_ok.append(float(Az))
        C_ok.append(jacobi_constant_sail(s0, mu, beta))
        T_ok.append(float(P))
        S_ok.append(s0.copy())

    if verbose:
        print(f"    {len(Az_ok)}/{len(az_grid)} on-branch, "
              f"{len(failures)} skipped")
        for f in failures[:4]:
            print(f"      Az={f[0]:.5f}  {f[1]}: {f[2]}")

    return dict(Az=np.array(Az_ok), C=np.array(C_ok), T=np.array(T_ok),
                state0=S_ok, n_failed=len(failures), failures=failures)
