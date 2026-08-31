"""
continuation.py — pseudo-arclength continuation of periodic orbits in the
                  solar-sail CR3BP.

Why natural-parameter stepping is not enough
────────────────────────────────────────────
The corrector in orbits.py solves, for FIXED z0 = Az,

    F(u) = [ vx(T_h), vz(T_h), y(T_h) ] = 0,      u = [x0, vy0, T_h]     (1)

Stepping Az and re-solving ("natural-parameter continuation") fails wherever the
branch has a FOLD in Az — a turning point where dAz/ds = 0.  Past a fold there
is no solution at the requested Az at all, so Newton either diverges or jumps to
a different branch.  That is exactly the branch-loss seen in halo_asymmetry.py,
where sweeps started at different beta ended up on different families.

The fix: treat the continuation parameter as an unknown
──────────────────────────────────────────────────────
Let lambda be the continuation parameter (here z0, or beta) and

    v = (u, lambda) in R^4

Pseudo-arclength appends an arclength condition to (1):

    F(u, lambda)                       = 0        (3 equations)
    (v - v_prev) . t_prev  -  ds       = 0        (1 equation)              (2)

where t_prev is the unit tangent to the branch at the previous solution.  The
augmented Jacobian

    J = [ F_u    F_lambda ]
        [ t_u    t_lambda ]                                                (3)

is non-singular AT a fold, because the arclength row supplies the direction the
branch is actually moving in.  So folds are traversed rather than tripped over,
and lambda is free to decrease.

Tangent.  Solve F_u du + F_lambda dlambda = 0 with dlambda = 1, giving the raw
tangent (-F_u^-1 F_lambda, 1); normalise, and pick the sign with
t . t_prev > 0 so the walk keeps its direction.

Derivatives.  F_u is the 3x3 matrix orbits.py already forms from the STM and the
end-point flow.  F_lambda is
    lambda = z0  :  [Phi[3,2], Phi[5,2], Phi[1,2]]      exact, from the STM
    lambda = beta:  finite-differenced (the STM does not carry d/dbeta)

Conventions
───────────
The initial state is always the symmetric form [x0, 0, z0, 0, vy0, 0], so t = 0
lies on the x-z plane at a z-extremum, and the full period is 2*T_h.  Along an
Az-continuation z0 VARIES — it is the unknown lambda, not a pinned input, which
is what makes folds passable.
"""

from __future__ import annotations

import numpy as np

from src.dynamics import cr3bp_sail_eom
from src.orbits import _propagate_with_stm, _richardson_guess
from src.jacobi import jacobi_constant_sail


# ── residual, Jacobian, and parameter sensitivity ─────────────────────────────

def _unpack(u, lam, param, other):
    """Build state0 and beta from (u, lambda) given which parameter is free."""
    x0, vy0, T_h = u
    if param == 'Az':
        z0, beta = lam, other
    elif param == 'beta':
        z0, beta = other, lam
    else:
        raise ValueError("param must be 'Az' or 'beta'")
    return np.array([x0, 0.0, z0, 0.0, vy0, 0.0]), float(beta), float(T_h)


def _eval(u, lam, param, other, mu, alpha, delta, rtol, fd_beta=1e-6):
    """
    Returns F (3,), F_u (3,3), F_lam (3,), state_f (6,), Phi (6,6).
    """
    state0, beta, T_h = _unpack(u, lam, param, other)
    state_f, Phi, _, _ = _propagate_with_stm(
        state0, T_h, alpha, delta, beta, mu, rtol=rtol, atol=rtol)

    F = np.array([state_f[3], state_f[5], state_f[1]])
    sdot = np.asarray(cr3bp_sail_eom(T_h, state_f, alpha, delta, beta, mu))

    # d[vx_f, vz_f, y_f] / d[x0, vy0, T_h]
    F_u = np.array([
        [Phi[3, 0], Phi[3, 4], sdot[3]],
        [Phi[5, 0], Phi[5, 4], sdot[5]],
        [Phi[1, 0], Phi[1, 4], sdot[1]],
    ])

    if param == 'Az':
        # exact: z0 is state index 2, so the STM already has it
        F_lam = np.array([Phi[3, 2], Phi[5, 2], Phi[1, 2]])
    else:
        sf2, _, _, _ = _propagate_with_stm(
            state0, T_h, alpha, delta, beta + fd_beta, mu,
            rtol=rtol, atol=rtol)
        F_lam = (np.array([sf2[3], sf2[5], sf2[1]]) - F) / fd_beta

    return F, F_u, F_lam, state_f, Phi


def _tangent(F_u, F_lam, t_prev=None):
    """Unit tangent to the branch, oriented to continue the previous direction."""
    try:
        du = -np.linalg.solve(F_u, F_lam)
    except np.linalg.LinAlgError:
        du = -np.linalg.lstsq(F_u, F_lam, rcond=None)[0]
    t = np.concatenate([du, [1.0]])
    t /= np.linalg.norm(t)
    if t_prev is not None and float(t @ t_prev) < 0.0:
        t = -t
    return t


# ── the corrector on the augmented system ─────────────────────────────────────

def _correct(v_pred, v_prev, t_prev, ds, param, other, mu, alpha, delta,
             rtol, tol, max_iter):
    """
    Solve the augmented system (2) by Newton from the predictor `v_pred`.
    Returns (v, n_iter, residual) or raises RuntimeError.
    """
    v = v_pred.copy()
    for it in range(1, max_iter + 1):
        F, F_u, F_lam, _, _ = _eval(v[:3], v[3], param, other,
                                    mu, alpha, delta, rtol)
        g = float((v - v_prev) @ t_prev - ds)
        R = np.concatenate([F, [g]])
        res = float(np.max(np.abs(R)))
        if res < tol:
            return v, it, res

        J = np.zeros((4, 4))
        J[:3, :3] = F_u
        J[:3, 3] = F_lam
        J[3, :] = t_prev
        try:
            dv = np.linalg.solve(J, -R)
        except np.linalg.LinAlgError:
            dv = np.linalg.lstsq(J, -R, rcond=None)[0]

        # trust region: cap the step at half the arclength
        n = np.linalg.norm(dv)
        if n > 0.5 * abs(ds) and n > 0:
            dv *= (0.5 * abs(ds)) / n
        v = v + dv

    raise RuntimeError(f"augmented corrector stalled at residual {res:.2e}")


# ── stability ─────────────────────────────────────────────────────────────────

def stability_indices(state0, period, mu, alpha, delta, beta,
                      rtol: float = 1e-11) -> dict:
    """
    Floquet analysis of one orbit.

    The CR3BP monodromy has a trivial reciprocal pair at +1.  The remaining four
    eigenvalues form two reciprocal pairs; for each we report the stability index

        nu = (lam + 1/lam) / 2

    which is real, and satisfies |nu| <= 1 exactly when that pair lies on the
    unit circle.  So max|nu| <= 1 is linear stability.
    """
    _, Phi, _, _ = _propagate_with_stm(state0, period, alpha, delta, beta, mu,
                                       rtol=rtol, atol=rtol)
    w = np.linalg.eigvals(Phi)

    # drop the two eigenvalues closest to +1 (the trivial pair)
    order = np.argsort(np.abs(w - 1.0))
    rest = w[order[2:]]

    # pair them by reciprocity
    used, nus, lam_max = set(), [], 0.0
    for i in range(len(rest)):
        if i in used:
            continue
        j, best = None, np.inf
        for k in range(i + 1, len(rest)):
            if k in used:
                continue
            d = abs(rest[i] * rest[k] - 1.0)
            if d < best:
                best, j = d, k
        if j is None:
            continue
        used.update({i, j})
        lam = rest[i] if abs(rest[i]) >= abs(rest[j]) else rest[j]
        nus.append(complex(0.5 * (lam + 1.0 / lam)).real)
        lam_max = max(lam_max, abs(lam))

    nus = sorted(nus, key=abs, reverse=True)
    return dict(eigenvalues=w, nu=nus,
                nu_max=(max(abs(n) for n in nus) if nus else np.nan),
                lambda_max=lam_max,
                stable=bool(nus and max(abs(n) for n in nus) <= 1.0))


# Seed amplitudes as fractions of the local length scale gamma (the
# equilibrium's standoff from the secondary).  A halo only exists while its
# amplitude is a modest fraction of gamma; beyond that the corrector leaves the
# local family altogether.  See AZ_OVER_GAMMA below.
AZ_OVER_GAMMA = np.array([0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.50])

# A family member whose x0 sits further than this many gamma from the
# equilibrium is not a local halo — it is a far-field escape.
MAX_DX_OVER_GAMMA = 3.0


def local_scale(eq_pos, mu: float) -> float:
    """
    Local length scale gamma = |(1-mu) - x_eq|, the equilibrium's standoff from
    the secondary.

    Taken from eq_pos rather than re-solving, so this works for any collinear
    point in any system: the absolute value covers L2-type equilibria, which sit
    beyond the secondary and give a negative signed standoff.
    """
    return abs((1.0 - mu) - float(eq_pos[0]))


def find_halo_seed(eq_pos, mu: float,
                   az_candidates=None,
                   az_over_gamma=None,
                   alpha: float = 0.0, delta: float = 0.0,
                   beta: float = 0.0,
                   max_dx_over_gamma: float = MAX_DX_OVER_GAMMA,
                   verbose: bool = False) -> tuple:
    """
    Scan candidate amplitudes for a VALIDATED halo to seed a continuation from.

    The Richardson guess does not land on the halo branch everywhere.  At
    Earth-Moon L2 it reaches the halo only for Az ~ 0.008-0.014; elsewhere it
    converges to the vertical-Lyapunov branch (T ~ 3.518, delta ~ 1e-11) or to
    spurious short-period orbits (T ~ 1.34, delta ~ 42).  So rather than trusting
    one guess, scan and take the first amplitude that passes `require_halo`.

    Amplitudes are DIMENSIONLESS by default
    ───────────────────────────────────────
    Candidates are `az_over_gamma * gamma`, not absolute values.  The former
    absolute default (Az from 0.004 to 0.060) was implicitly tuned for
    Earth-Moon, where gamma ~ 0.15 makes it Az/gamma ~ 0.03-0.40 — a sensible
    band.  Applied unchanged to Sun-Earth, where gamma ~ 0.010-0.020, the very
    same numbers mean Az/gamma ~ 0.2-6.0: amplitudes up to six times the entire
    local length scale.  The corrector then converged to far-field orbits with
    |x0 - x_eq| as large as 70 gamma, and those entered the atlas as "halos".
    Scaling by gamma reproduces the good Earth-Moon band in every system.

    Parameters
    ----------
    az_over_gamma   candidate amplitudes in units of gamma (default AZ_OVER_GAMMA)
    az_candidates   ABSOLUTE amplitudes; overrides az_over_gamma when given
    max_dx_over_gamma
                    reject a converged seed whose |x0 - x_eq| exceeds this many
                    gamma, so a far-field orbit can never be returned as a seed

    Returns (state0, T_half, Az) or raises RuntimeError.
    """
    from src.orbits import (compute_halo_orbit, HaloValidationError,
                            HaloConvergenceError)
    gamma = local_scale(eq_pos, mu)
    if az_candidates is None:
        ratios = AZ_OVER_GAMMA if az_over_gamma is None else np.asarray(
            az_over_gamma, dtype=float)
        az_candidates = gamma * ratios

    n_far = 0
    for Az in az_candidates:
        try:
            s0, P, info = compute_halo_orbit(eq_pos, float(Az), mu,
                                             alpha, delta, beta,
                                             require_halo=True,
                                             return_info=True)
        except (HaloValidationError, HaloConvergenceError):
            continue
        dx_g = abs(s0[0] - float(eq_pos[0])) / gamma
        if dx_g > max_dx_over_gamma:
            # Converged, validated as a halo, but far outside the local
            # neighbourhood: a far-field orbit, not a member of this family.
            n_far += 1
            if verbose:
                print(f"    rejected Az={Az:.6f} (Az/gamma={Az/gamma:.2f}): "
                      f"|x0-x_eq| = {dx_g:.1f} gamma > {max_dx_over_gamma:g}")
            continue
        if verbose:
            print(f"    seed: Az={Az:.6f} (Az/gamma={Az/gamma:.3f})  T={P:.5f}  "
                  f"delta={info['z_asymmetry']:.4f}  "
                  f"|x0-x_eq|={dx_g:.3f} gamma")
        return s0, P / 2.0, float(Az)

    raise RuntimeError(
        f"no validated halo seed found among {len(az_candidates)} amplitudes "
        f"({n_far} converged but were rejected as far-field, "
        f"|x0-x_eq| > {max_dx_over_gamma:g} gamma); the halo branch may not "
        f"exist for these parameters (beta={beta:g}, mu={mu:g}, "
        f"gamma={gamma:.6g})")


# ── the branch walker ─────────────────────────────────────────────────────────

def continue_branch(eq_pos, mu: float,
                    param: str = 'Az',
                    lam0: float = None,
                    other: float = 0.0,
                    alpha: float = 0.0, delta: float = 0.0,
                    ds: float = 2e-3,
                    n_steps: int = 200,
                    lam_bounds: tuple = None,
                    ds_min: float = 1e-5, ds_max: float = 2e-2,
                    rtol: float = 1e-11, tol: float = 1e-10,
                    max_iter: int = 25,
                    with_stability: bool = True,
                    seed_state: tuple = None,
                    max_dx_over_gamma: float = MAX_DX_OVER_GAMMA,
                    verbose: bool = False) -> dict:
    """
    Walk a periodic-orbit branch by pseudo-arclength continuation.

    Parameters
    ----------
    eq_pos      equilibrium position, only used for the Richardson seed
    param       'Az' (continue in z0) or 'beta'
    lam0        starting value of the continuation parameter
    other       the parameter held fixed (beta if param='Az', Az if param='beta')
    ds          initial arclength step; halved on failure, grown on easy steps
    lam_bounds  (lo, hi) — stop when lambda leaves this interval
    with_stability  compute Floquet indices at each point (roughly doubles cost)
    seed_state  optional (state0, T_half) to start the walk from a KNOWN orbit.
                Essential when the Richardson guess lands on the wrong family:
                at Earth-Moon L2 it converges to the vertical-Lyapunov branch
                (T ~ 3.518) rather than the halo branch (T ~ 3.409).  Pass a
                validated halo from orbits.compute_halo_orbit(require_halo=True).

    max_dx_over_gamma
                HARD GUARD.  A member whose |x0 - x_eq| exceeds this many local
                length scales gamma = |(1-mu) - x_eq| is not a local halo but a
                far-field orbit, and the walk stops there.  The check runs
                BEFORE the member is recorded, so a far-field state can never
                enter the returned arrays — and therefore never the exported
                CSV.  Set to np.inf to disable.

                This is not hypothetical: with the former absolute seed
                amplitudes, 155 of 270 atlas members were far-field, reaching
                70 gamma.  The stop reason is reported in `out['stopped']`.

    Returns
    -------
    dict of arrays: lam, Az, beta, x0, vy0, T (full period), C, nu_max,
    lambda_max, stable, plus `state0` (list), `n_failed`, `folds`, `gamma`,
    `n_far_field` and `stopped`.
    """
    if param not in ('Az', 'beta'):
        raise ValueError("param must be 'Az' or 'beta'")
    if lam0 is None:
        lam0 = 0.005 if param == 'Az' else 0.001

    # ── seed: solve the ordinary 3x3 problem once at lambda = lam0 ──────────
    Az_seed = lam0 if param == 'Az' else other
    beta_seed = other if param == 'Az' else lam0
    s_rich, Th_rich = _richardson_guess(eq_pos, Az_seed, mu, alpha, delta,
                                        beta_seed)
    if seed_state is not None:
        st_s, Th_s = seed_state
        st_s = np.asarray(st_s, dtype=float)
        u = np.array([st_s[0], st_s[4], float(Th_s)])
        if param == 'Az':
            lam0 = float(st_s[2])
        Th_rich = float(Th_s)
    else:
        u = np.array([s_rich[0], s_rich[4], Th_rich])

    for _ in range(60):
        F, F_u, F_lam, _, _ = _eval(u, lam0, param, other, mu, alpha, delta, rtol)
        if np.max(np.abs(F)) < tol:
            break
        try:
            du = np.linalg.solve(F_u, -F)
        except np.linalg.LinAlgError:
            du = np.linalg.lstsq(F_u, -F, rcond=None)[0]
        if abs(du[2]) > 0.4 * abs(u[2]):
            du *= 0.4 * abs(u[2]) / abs(du[2])
        u = u + du
        u[2] = max(u[2], 0.1 * Th_rich)
    else:
        raise RuntimeError(f"could not converge the seed orbit at {param}={lam0}")

    v = np.concatenate([u, [lam0]])
    _, F_u, F_lam, _, _ = _eval(v[:3], v[3], param, other, mu, alpha, delta, rtol)
    t = _tangent(F_u, F_lam)
    if t[3] < 0:                    # start by increasing lambda
        t = -t

    rows, folds, n_failed = [], [], 0
    t_prev = t

    # Local length scale and the far-field guard.
    x_eq = float(eq_pos[0])
    gamma = local_scale(eq_pos, mu)
    n_far_field, stopped = 0, None

    def _is_far_field(v_) -> float | None:
        """Return |x0 - x_eq|/gamma when it breaches the guard, else None."""
        d = abs(float(v_[0]) - x_eq) / gamma
        return d if d > max_dx_over_gamma else None

    def _record(v_):
        state0, beta_, T_h = _unpack(v_[:3], v_[3], param, other)
        P = 2.0 * T_h
        sf, _, _, _ = _propagate_with_stm(state0, T_h, alpha, delta, beta_, mu,
                                          rtol=rtol, atol=rtol)
        a_, b_ = abs(state0[2]), abs(sf[2])
        dlt = (a_ - b_) / (a_ + b_) if (a_ + b_) > 0 else 0.0
        rec = dict(lam=float(v_[3]), delta=float(dlt),
                   Az=float(state0[2]), beta=float(beta_),
                   x0=float(state0[0]), vy0=float(state0[4]),
                   T=float(P), C=jacobi_constant_sail(state0, mu, beta_),
                   state0=state0.copy())
        if with_stability:
            st = stability_indices(state0, P, mu, alpha, delta, beta_, rtol)
            rec.update(nu_max=st['nu_max'], lambda_max=st['lambda_max'],
                       stable=st['stable'])
        else:
            rec.update(nu_max=np.nan, lambda_max=np.nan, stable=False)
        return rec

    d_seed = _is_far_field(v)
    if d_seed is not None:
        raise RuntimeError(
            f"seed orbit is far-field: |x0 - x_eq| = {d_seed:.1f} gamma "
            f"> {max_dx_over_gamma:g} (gamma = {gamma:.6g}, x_eq = {x_eq:.8f}, "
            f"x0 = {v[0]:.8f}).  This is not a member of the local family; "
            f"seed from find_halo_seed(), which applies the same guard.")
    rows.append(_record(v))

    step = ds
    for k in range(n_steps):
        v_pred = v + step * t_prev
        try:
            v_new, it, _ = _correct(v_pred, v, t_prev, step, param, other,
                                    mu, alpha, delta, rtol, tol, max_iter)
        except RuntimeError:
            n_failed += 1
            step *= 0.5
            if abs(step) < ds_min:
                stopped = f"arclength fell below ds_min={ds_min:g}"
                if verbose:
                    print(f"    stopped: {stopped}")
                break
            continue

        # HARD GUARD, before anything about v_new is recorded.  Pseudo-arclength
        # walks a connected curve, so once the branch has left the local
        # neighbourhood it does not come back along the same branch — stop
        # rather than skip.
        d_far = _is_far_field(v_new)
        if d_far is not None:
            n_far_field += 1
            stopped = (f"far-field guard: |x0 - x_eq| = {d_far:.2f} gamma "
                       f"> {max_dx_over_gamma:g} at {param}={v_new[3]:.6f}")
            if verbose:
                print(f"    stopped: {stopped}")
            break

        _, F_u, F_lam, _, _ = _eval(v_new[:3], v_new[3], param, other,
                                    mu, alpha, delta, rtol)
        t_new = _tangent(F_u, F_lam, t_prev)

        # a sign change in the lambda-component of the tangent is a fold
        if t_new[3] * t_prev[3] < 0:
            folds.append(float(v_new[3]))
            if verbose:
                print(f"    fold in {param} at {v_new[3]:.6f}")

        # Bounds check also runs BEFORE recording, for the same reason as the
        # far-field guard: a post-record check admits exactly one out-of-bounds
        # member per branch, which is how Az/gamma values of 1.004 and 3e-8 (a
        # degenerate zero-amplitude planar orbit) reached the exported CSV.
        if lam_bounds is not None:
            lo, hi = lam_bounds
            if not (lo <= v_new[3] <= hi):
                stopped = (f"{param}={v_new[3]:.6g} left bounds "
                           f"[{lo:g}, {hi:g}]")
                if verbose:
                    print(f"    stopped: {stopped}")
                break

        v, t_prev = v_new, t_new
        rows.append(_record(v))

        # grow the step when Newton found it easy
        if it <= 3:
            step = min(step * 1.3, ds_max)

    if verbose:
        print(f"    {len(rows)} points, {len(folds)} folds, "
              f"{n_failed} retries")

    keys = ('lam', 'delta', 'Az', 'beta', 'x0', 'vy0', 'T', 'C',
            'nu_max', 'lambda_max')
    out = {k: np.array([r[k] for r in rows]) for k in keys}
    out['stable'] = np.array([r['stable'] for r in rows], dtype=bool)
    out['state0'] = [r['state0'] for r in rows]
    out['folds'] = folds
    out['n_failed'] = n_failed
    out['param'] = param
    out['other'] = other
    out['gamma'] = gamma
    out['x_eq'] = x_eq
    out['n_far_field'] = n_far_field
    out['stopped'] = stopped or f"completed {n_steps} steps"
    # Post-condition: the guard must have held for every recorded member.
    dx_g = np.abs(out['x0'] - x_eq) / gamma
    assert np.all(dx_g <= max_dx_over_gamma), (
        f"far-field member leaked into the branch: max |x0-x_eq| = "
        f"{dx_g.max():.3f} gamma")
    out['max_dx_over_gamma'] = float(dx_g.max()) if dx_g.size else 0.0
    if lam_bounds is not None:
        lo, hi = lam_bounds
        assert np.all((out['lam'] >= lo) & (out['lam'] <= hi)), (
            f"out-of-bounds member leaked into the branch: {param} in "
            f"[{out['lam'].min():.6g}, {out['lam'].max():.6g}], "
            f"bounds [{lo:g}, {hi:g}]")
    return out


if __name__ == '__main__':
    from src.jacobi_match import lagrange_point

    from src.orbits import compute_halo_orbit

    MU_EM = 0.01215
    print("\n== Pseudo-arclength continuation, Earth-Moon halo families ==")
    for which in ('L1', 'L2'):
        eq = lagrange_point(which, MU_EM)
        print(f"\n  {which}  (eq x = {eq[0]:.8f})")
        # seed from a VALIDATED halo so we start on the halo branch, not the
        # vertical-Lyapunov branch the Richardson guess prefers at L2
        s0, Th0, Az0 = find_halo_seed(eq, MU_EM, verbose=True)
        br = continue_branch(eq, MU_EM, param='Az',
                             seed_state=(s0, Th0),
                             ds=2e-3, n_steps=90,
                             lam_bounds=(1e-4, 0.10), verbose=True)
        print(f"    Az   [{br['Az'].min():.5f}, {br['Az'].max():.5f}]")
        print(f"    C    [{br['C'].min():.8f}, {br['C'].max():.8f}]")
        print(f"    T    [{br['T'].min():.5f}, {br['T'].max():.5f}]")
        print(f"    nu_max [{np.nanmin(br['nu_max']):.4f}, "
              f"{np.nanmax(br['nu_max']):.4f}]   "
              f"stable points: {int(br['stable'].sum())}/{len(br['Az'])}")
        print(f"    delta  [{br['delta'].min():+.4f}, {br['delta'].max():+.4f}]"
              f"   -> {'HALO' if np.abs(br['delta']).min() > 1e-6 else 'symmetric branch'}")
        dC = np.diff(br['C'])
        print(f"    C monotonic along arclength: "
              f"{bool(np.all(dC < 0) or np.all(dC > 0))}")
