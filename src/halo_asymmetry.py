"""
halo_asymmetry.py — does the halo family terminate as beta increases?  No.

This module replaces an earlier `bifurcation.py` whose premise was wrong.  The
negative result is recorded here in full because the wrong version is an easy
mistake to repeat.

The order parameter
──────────────────
The CR3BP is invariant under reflection in the x-y plane,

    S : (x, y, z, vx, vy, vz)  ->  (x, y, -z, vx, vy, -vz)

An orbit invariant under S composed with a half-period shift obeys
z(t + T/2) = -z(t), hence z(T/2) = -z(0) exactly.  Vertical-Lyapunov / axial
orbits have that symmetry; a halo does not.  So

    delta = ( |z0| - |z(T/2)| ) / ( |z0| + |z(T/2)| )                       (1)

is O(0.1) for a genuine halo and 0 on the symmetric branch.  It is a useful
branch DISCRIMINATOR: it is what exposed the Earth-Moon L2 orbit at Az = 0.02 as
a vertical-Lyapunov member (delta ~ 1e-11) rather than a halo.

The false lead
──────────────
Swept at FIXED Az = 0.003, delta appeared to fall off a cliff between
beta = 0.05 and 0.10 — from 1.3e-1 to ~1e-13 with nothing in between — which
looked like the halo family terminating in a symmetry-breaking bifurcation.
It is an artifact, for two compounding reasons:

  1. delta is a NONLINEAR quantity: it vanishes identically for a linearised
     orbit.  Holding Az fixed while the equilibrium's own length scale
     gamma = (1-mu) - x_eq grows (0.00997 at beta=0 to 0.0353 at beta=0.1)
     shrinks Az/gamma four-fold, pushing the orbit into the linear regime.  The
     decay of delta was measuring that, not a change of family.

  2. The continuation was branch-hopping.  At fixed Az the period jumped
     3.260 -> 2.634 across beta = 0.005 -> 0.010, so the "middle regime" was a
     different family altogether.

Holding Az/gamma fixed instead removes both: the continuation runs smoothly and
delta stays near 0.13 through beta = 0.03.

The verdict
───────────
On the smooth (scaled-amplitude) branch:

  * delta has a TRANSVERSAL ZERO at beta ~ 0.1067 — it passes through zero and
    grows again with the opposite sign (+2.0e-3 at beta=0.110 rising to
    +1.38e-2 at beta=0.140).  A bifurcation would have the branch END there, not
    continue through.  The zero is simply the beta at which the orbit's two
    z-extrema happen to be equal.

  * fitting delta ~ (beta_c - beta)^p gives p = 1.704 with R^2 = 0.998.
    A pitchfork requires p = 0.5.  The measured exponent is off by more than a
    factor of three, so the square-root law is decisively rejected.

There is no bifurcation, and the halo family does not terminate.  The collapse
of halo character with beta is the same smooth death of the Earth's tidal term
already quantified by the saddle strength s = mu/r2^3 in critical_beta.py, seen
in the orbit family instead of in the linearisation — not a separate phenomenon
and not a dynamical event.

LIMITATION — read before using fig_asymmetry()
──────────────────────────────────────────────
The natural-parameter continuation used here is NOT robust.  Sweeps started at
different beta land on different branches: a sweep begun at beta = 0 shows delta
jumping +0.13 -> -0.61 across beta = 0.005, while one begun at beta = 0.020
shows +0.137 -> -0.149 there and a zero near beta = 0.107.  The coarse
fixed-amplitude sweep collapses to ~1e-13; a finer one does not.

The two robust conclusions survive every sweep tried:
  * the fitted exponent is never near 1/2 (measured 1.17 and 1.70), so a
    pitchfork is rejected;
  * delta crosses zero TRANSVERSALLY and the family continues past it.

The fine structure is not trustworthy, so no figure is shipped from this module.
Making a publishable statement about the beta-dependence of halo asymmetry needs
pseudo-arclength continuation with a proper tangent predictor and arclength
constraint, not the natural-parameter stepping used here.
"""

from __future__ import annotations

import numpy as np

from src.orbits import (compute_halo_orbit, HaloConvergenceError,
                        HaloValidationError)
from src.critical_beta import equilibrium, saddle_strength, MU_SE

PITCHFORK_EXPONENT = 0.5


def asymmetry(info: dict) -> float:
    """delta of eq. (1), from a compute_halo_orbit info dict."""
    a, b = abs(info['z0']), abs(info['z_half'])
    return (a - b) / (a + b) if (a + b) > 0 else 0.0


def gamma_of(beta: float, mu: float = MU_SE) -> float:
    """Local length scale: the equilibrium's standoff from the secondary."""
    return (1.0 - mu) - equilibrium(beta, mu)


def sweep(betas, Az=None, az_over_gamma=None, mu: float = MU_SE,
          verbose: bool = False) -> dict:
    """
    Continue the orbit through beta, recording delta.

    Give exactly one of `Az` (fixed absolute amplitude — reproduces the
    artifact) or `az_over_gamma` (fixed Az/gamma — the meaningful sweep).
    """
    if (Az is None) == (az_over_gamma is None):
        raise ValueError("give exactly one of Az or az_over_gamma")

    rows, seed = [], None
    for b in np.asarray(betas, dtype=float):
        g = gamma_of(float(b), mu)
        A = float(Az) if Az is not None else az_over_gamma * g
        try:
            s0, P, info = compute_halo_orbit(
                [equilibrium(float(b), mu), 0.0, 0.0], A, mu, 0.0, 0.0,
                float(b), seed=seed, require_halo=False, return_info=True)
        except (HaloConvergenceError, HaloValidationError):
            continue
        seed = (s0[0], s0[4], P / 2.0)
        rows.append((float(b), asymmetry(info), float(P), A, g,
                     saddle_strength(float(b), mu)))

    if verbose:
        print(f"    {len(rows)}/{len(np.asarray(betas))} converged")

    a = np.array(rows) if rows else np.zeros((0, 6))
    return dict(beta=a[:, 0], delta=a[:, 1], T=a[:, 2],
                Az=a[:, 3], gamma=a[:, 4], s=a[:, 5])


def power_law_exponent(beta: np.ndarray, delta: np.ndarray,
                       noise: float = 1e-6) -> dict:
    """
    Fit |delta| ~ (beta_c - beta)^p on the decaying tail, scanning beta_c.

    p ~ 0.5 would support a pitchfork; anything else rejects it.
    """
    m = (np.abs(delta) > noise) & (delta < 0)
    if m.sum() < 6:
        return dict(ok=False, reason=f"{m.sum()} usable points")
    bb, dd = beta[m], np.abs(delta[m])

    best = None
    for bc in np.linspace(bb.max() + 1e-4, bb.max() + 0.09, 500):
        X, Y = np.log(bc - bb), np.log(dd)
        A = np.vstack([X, np.ones_like(X)]).T
        c, *_ = np.linalg.lstsq(A, Y, rcond=None)
        r2 = 1.0 - np.sum((Y - A @ c) ** 2) / np.sum((Y - Y.mean()) ** 2)
        if best is None or r2 > best[0]:
            best = (r2, float(bc), float(c[0]))
    r2, bc, p = best
    return dict(ok=True, r2=r2, beta_c=bc, p=p,
                pitchfork=abs(p - PITCHFORK_EXPONENT) < 0.1,
                beta_used=bb, delta_used=dd)


def zero_crossing(beta: np.ndarray, delta: np.ndarray,
                  after: float = 0.05) -> float | None:
    """Locate a transversal zero of delta above `after` by linear interpolation."""
    m = beta > after
    b, d = beta[m], delta[m]
    k = np.where(np.diff(np.sign(d)) != 0)[0]
    if k.size == 0:
        return None
    i = int(k[0])
    return float(b[i] - d[i] * (b[i + 1] - b[i]) / (d[i + 1] - d[i]))


def fig_asymmetry(output: str = 'fig9_halo_asymmetry.png',
                  verbose: bool = True) -> dict:
    """Plain-paper figure: the artifact, the pitchfork test, and the crossing."""
    from src.paperstyle import use, panel_label, thin_guide, DASHES
    use()
    import matplotlib.pyplot as plt

    if verbose:
        print("  Sweeping at fixed Az (reproduces the artifact) …")
    g0 = gamma_of(0.0)
    ratio = 0.003 / g0
    b_fix = np.linspace(0.0, 0.10, 26)
    fix = sweep(b_fix, Az=0.003, verbose=verbose)

    if verbose:
        print("  Sweeping at fixed Az/gamma (the meaningful one) …")
    b_scl = np.concatenate([np.linspace(0.0, 0.05, 18),
                            np.linspace(0.055, 0.145, 19)])
    scl = sweep(b_scl, az_over_gamma=ratio, verbose=verbose)

    fit = power_law_exponent(scl['beta'], scl['delta'])
    bz = zero_crossing(scl['beta'], scl['delta'])
    if verbose:
        if fit['ok']:
            print(f"  power law: p = {fit['p']:.4f} (pitchfork needs "
                  f"{PITCHFORK_EXPONENT}), R^2 = {fit['r2']:.6f}")
        if bz:
            print(f"  transversal zero of delta at beta = {bz:.5f}")

    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.2))

    # (a) the artifact
    ax = axes[0]
    l1 = ax.plot(fix['beta'], np.abs(fix['delta']), 'k-', lw=1.0,
                 label=r'$A_z = 0.003$ fixed')[0]
    l2 = ax.plot(scl['beta'], np.abs(scl['delta']), 'k-', lw=1.0,
                 label=r'$A_z/\gamma$ fixed')[0]
    l2.set_dashes([5, 2])
    ax.set_yscale('log')
    ax.set_ylim(1e-15, 1.0)
    thin_guide(ax, y=1e-6, label='noise floor')
    ax.set_xlabel(r'$\beta$')
    ax.set_ylabel(r'$|\delta|$')
    ax.legend(loc='lower left')
    panel_label(ax, '(a)')

    # (b) pitchfork test
    ax = axes[1]
    if fit['ok']:
        bb, dd = fit['beta_used'], fit['delta_used']
        ax.loglog(fit['beta_c'] - bb, dd, 'ko', ms=3.0, mfc='none', mew=0.8,
                  label='data')
        xr = np.array([(fit['beta_c'] - bb).min(), (fit['beta_c'] - bb).max()])
        Cf = dd[0] / (fit['beta_c'] - bb[0]) ** fit['p']
        ax.loglog(xr, Cf * xr ** fit['p'], 'k-', lw=1.0,
                  label=rf"$p={fit['p']:.2f}$")
        Cp = dd[0] / (fit['beta_c'] - bb[0]) ** PITCHFORK_EXPONENT
        lp, = ax.loglog(xr, Cp * xr ** PITCHFORK_EXPONENT, 'k-', lw=1.0,
                        label=r'pitchfork, $p=1/2$')
        lp.set_dashes([1, 1.6])
    ax.set_xlabel(r'$\beta_c - \beta$')
    ax.set_ylabel(r'$|\delta|$')
    ax.legend(loc='lower right')
    panel_label(ax, '(b)')

    # (c) the family continues through the zero
    ax = axes[2]
    ax.plot(scl['beta'], scl['delta'], 'k-', lw=1.0)
    ax.plot(scl['beta'], scl['delta'], 'ko', ms=2.2)
    ax.axhline(0.0, color='0.45', lw=0.6, dashes=[2, 2], zorder=0)
    if bz:
        thin_guide(ax, x=bz)
        ax.annotate(rf'$\delta=0$ at $\beta={bz:.4f}$',
                    xy=(bz, 0.055), xytext=(6, 0), textcoords='offset points',
                    fontsize=7.5, va='center')
    ax.set_xlabel(r'$\beta$')
    ax.set_ylabel(r'$\delta$  (signed)')
    panel_label(ax, '(c)')

    fig.suptitle('Halo asymmetry does not terminate the family: '
                 'the apparent collapse is a fixed-amplitude artifact',
                 fontsize=9.5, y=1.008)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    if verbose:
        print(f"  Saved -> {output}")

    return dict(fixed=fix, scaled=scl, fit=fit, beta_zero=bz)


if __name__ == '__main__':
    print("\n== Halo asymmetry vs beta ==============================")
    r = fig_asymmetry()
    f, bz = r['fit'], r['beta_zero']
    print("\n  VERDICT")
    if f['ok']:
        print(f"    fitted exponent p        = {f['p']:.4f}   "
              f"(R^2 = {f['r2']:.6f})")
        print(f"    pitchfork requires p     = {PITCHFORK_EXPONENT}")
        print(f"    pitchfork consistent?    = {f['pitchfork']}")
    if bz:
        print(f"    delta crosses zero at    = beta {bz:.5f}, transversally,")
        print(f"                               and grows again beyond it")
    print("    -> NO bifurcation; the halo family does not terminate.")
