"""
frequency_ratio.py — the in-plane / out-of-plane frequency ratio at the on-axis
                     sail equilibrium, and the resonance question it settles.

Why this module exists
──────────────────────
critical_beta.py establishes an exact threshold at TIDAL PARITY, s = mu/r2^3 = 1:

    beta_crit = 1 - (1 - mu^(1/3))^2 = 2 mu^(1/3) - mu^(2/3)              (1)

That is a level crossing of a smooth, monotone function.  Nothing bifurcates
there — A = 1 + s with s > 0 always — so a referee is entitled to ask why THAT
level and not another, and "s = 1" is partly a statement about the choice of
non-dimensionalisation.  The standard way to promote such a marker into a
dynamical event is to find a commensurability of the linear frequencies.  This
module looks for one.

The linear rates
────────────────
At the on-axis equilibrium the effective potential gives U_xx = 1 + 2A,
U_yy = 1 - A, U_zz = -A, so the out-of-plane motion is a pure oscillator

    nu = sqrt(A)                                                          (2)

and the planar characteristic polynomial lam^4 + (2-A) lam^2 + (1+A-2A^2) = 0
has roots lam^2 = [ (A-2) +/- D ] / 2 with D = sqrt(A(9A-8)), giving

    omega^2 = [ D - (A-2) ] / 2      (centre)
    lam_u^2 = [ (A-2) + D ] / 2      (saddle)                             (3)

RESULT 1 — there is no resonance, anywhere
──────────────────────────────────────────
Over the entire physically reachable range (A from 4.0608 at beta = 0 down to 1
in the hovering limit), the ratio is confined to

    nu / omega  in  [ 2 sqrt(2) / 3 ,  1 )  =  [0.9428090, 1)             (4)

a band 5.72 % wide.  NOT ONE low-order rational lies inside it — not 1:1, 9:8,
6:5, 5:4, 4:3, 3:2, 2:1, nor their reciprocals.  nu < omega strictly for A > 1,
and 1:1 is approached only asymptotically as the equilibrium degenerates into
the heliocentric hovering point.

So no commensurability justifies the tidal-parity threshold, and none exists
anywhere else in the face-on solar-sail CR3BP either.  Eq. (1) must be presented
as a SCALE MARKER with an exact closed form and a clean geometric reading
(r2 = mu^(1/3) = 3^(1/3) r_H), not as a dynamical transition.  This is a checked
negative and belongs in the paper as one.

CAVEAT on what (4) does NOT imply.  Halo families bifurcate from planar Lyapunov
families at a commensurability of the NONLINEAR frequencies, reached at finite
amplitude.  The classical L1 has nu/omega = 0.9659 and halo orbits exist there
regardless.  Nothing here constrains halo existence; only the linear ratio is
bounded away from unity.

RESULT 2 — an exact extremum, universal in mu
─────────────────────────────────────────────
nu/omega is stationary where d(nu/omega)/dA = 0.  Differentiating (2)-(3), the
condition reduces to D = 2A, i.e. 9A^2 - 8A = 4A^2, i.e.

    A_star = 8/5                                                          (5)

and there the three linear rates are exactly

    lam_u = sqrt(7/5),   nu = sqrt(8/5),   omega = sqrt(9/5)
    ->  lam_u^2 : nu^2 : omega^2  =  7 : 8 : 9                            (6)
    ->  nu / omega = sqrt(8/9) = 2 sqrt(2) / 3   (the minimum in (4))     (7)

A_star, the rate ratios (6) and the minimum value (7) are INDEPENDENT of mu —
they are properties of the collinear linearisation itself, shared by every
three-body system.  Only the beta that realises A = 8/5 depends on mu, and that
one has no closed form (unlike (1), because A mixes both the solar and Earth
terms rather than isolating r2).

Why (5) is the more defensible landmark
───────────────────────────────────────
An extremum is intrinsic: it survives any smooth reparametrisation and any
choice of units.  A level crossing at "s = 1" does not — it is a number crossing
unity in one particular non-dimensionalisation.  So (5) is immune to the "why
that value?" objection that (1) is exposed to, even though (1) is the result
with the closed form in beta.  Report both, and let (5) carry the claim that the
collinear structure has intrinsic internal scales.

Sun-Earth values
────────────────
    beta at tidal parity  s = 1        0.02864646      (exact, eq. 1)
    beta at A = 8/5                    0.04093195      (root-found)
    s at A = 8/5                       0.5896800
    ratio of the two betas             1.4287
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from src.critical_beta import (A_parameter, MU_SE, equilibrium,
                               saddle_strength)

# ── exact constants of the extremum, eqs. (5)-(7) ─────────────────────────────
A_STAR = 8.0 / 5.0                       # where nu/omega is stationary
NU_OMEGA_MIN = 2.0 * np.sqrt(2.0) / 3.0  # = sqrt(8/9), the minimum value
NU_STAR = np.sqrt(8.0 / 5.0)
OMEGA_STAR = np.sqrt(9.0 / 5.0)
LAMBDA_U_STAR = np.sqrt(7.0 / 5.0)

# Low-order rationals tested for attainability.
_RESONANCES = {
    '1:2': 0.5, '2:3': 2.0 / 3.0, '3:4': 0.75, '4:5': 0.8, '5:6': 5.0 / 6.0,
    '8:9': 8.0 / 9.0, '1:1': 1.0, '9:8': 9.0 / 8.0, '6:5': 1.2, '5:4': 1.25,
    '4:3': 4.0 / 3.0, '3:2': 1.5, '5:3': 5.0 / 3.0, '2:1': 2.0,
}


def linear_rates(A: float) -> dict:
    """nu, omega and lam_u from A alone, eqs. (2)-(3).  No mu needed."""
    D = np.sqrt(A * (9.0 * A - 8.0))
    om2 = (D - (A - 2.0)) / 2.0
    lu2 = ((A - 2.0) + D) / 2.0
    return dict(nu=float(np.sqrt(A)),
                omega=float(np.sqrt(om2)) if om2 > 0 else float('nan'),
                lam_u=float(np.sqrt(lu2)) if lu2 > 0 else 0.0,
                D=float(D))


def nu_over_omega(A: float) -> float:
    """The frequency ratio as a function of A alone.  Minimised at A = 8/5."""
    r = linear_rates(A)
    return r['nu'] / r['omega']


def beta_at_A(A_target: float, mu: float = MU_SE,
              b_hi: float = 0.95) -> float:
    """beta realising a given A.  A decreases monotonically in beta."""
    return brentq(lambda b: A_parameter(b, mu) - A_target,
                  1e-12, b_hi, xtol=1e-15)


def beta_at_frequency_extremum(mu: float = MU_SE) -> float:
    """beta where nu/omega is stationary, i.e. where A = 8/5.  Eq. (5)."""
    return beta_at_A(A_STAR, mu)


def verify_extremum(verbose: bool = True) -> dict:
    """Confirm eqs. (5)-(7) analytically and against a dense numeric scan."""
    r = linear_rates(A_STAR)
    dense_A = np.linspace(1.0 + 1e-9, 4.2, 400001)
    dense_r = np.sqrt(dense_A) / np.sqrt(
        (np.sqrt(dense_A * (9.0 * dense_A - 8.0)) - (dense_A - 2.0)) / 2.0)
    i = int(np.argmin(dense_r))

    out = dict(
        A_numeric=float(dense_A[i]), ratio_numeric=float(dense_r[i]),
        A_exact=A_STAR, ratio_exact=NU_OMEGA_MIN,
        dA=abs(float(dense_A[i]) - A_STAR),
        dratio=abs(float(dense_r[i]) - NU_OMEGA_MIN),
        D_equals_2A=abs(r['D'] - 2.0 * A_STAR),
        nu2=r['nu']**2, omega2=r['omega']**2, lam_u2=r['lam_u']**2,
        ratio_min=float(dense_r.min()), ratio_max=float(dense_r.max()),
    )
    if verbose:
        print(f"  extremum condition D = 2A         "
              f"|D - 2A| = {out['D_equals_2A']:.2e}")
        print(f"  A_star  exact 8/5 = {A_STAR}          "
              f"numeric = {out['A_numeric']:.8f}   d = {out['dA']:.2e}")
        print(f"  nu^2 : omega^2 : lam_u^2 = "
              f"{out['nu2']:.10f} : {out['omega2']:.10f} : {out['lam_u2']:.10f}")
        print(f"                           = 8/5 : 9/5 : 7/5   -> 8 : 9 : 7")
        print(f"  min nu/omega = 2*sqrt(2)/3 = {NU_OMEGA_MIN:.12f}   "
              f"numeric = {out['ratio_numeric']:.12f}   "
              f"d = {out['dratio']:.2e}")
        print(f"  attainable band: [{out['ratio_min']:.7f}, "
              f"{out['ratio_max']:.7f})   "
              f"width {100*(1-out['ratio_min']):.2f} % below unity")
    return out


def resonance_scan(verbose: bool = True) -> dict:
    """Which low-order rationals fall inside the attainable band?  None."""
    lo, hi = NU_OMEGA_MIN, 1.0
    hits = {k: (lo <= v < hi) for k, v in _RESONANCES.items()}
    if verbose:
        print(f"  attainable nu/omega: [{lo:.7f}, {hi:.7f})")
        for k, v in sorted(_RESONANCES.items(), key=lambda kv: kv[1]):
            print(f"    {k:>4} = {v:.6f}   "
                  f"{'*** REACHABLE ***' if hits[k] else 'not attainable'}")
        print(f"  -> {sum(hits.values())} of {len(hits)} low-order rationals "
              f"are reachable.")
    return dict(hits=hits, n_reachable=sum(hits.values()), lo=lo, hi=hi)


def summary(mu: float = MU_SE, verbose: bool = True) -> dict:
    """Headline numbers for this module."""
    from src.critical_beta import critical_beta_tidal_exact
    b_star = beta_at_frequency_extremum(mu)
    b_tide = critical_beta_tidal_exact(mu)
    out = dict(beta_star=b_star, beta_tidal=b_tide,
               s_at_star=saddle_strength(b_star, mu),
               A_at_tidal=A_parameter(b_tide, mu),
               ratio_at_tidal=nu_over_omega(A_parameter(b_tide, mu)),
               ratio_of_betas=b_star / b_tide)
    if verbose:
        print(f"  beta at tidal parity (s=1)     = {b_tide:.8f}   (exact)")
        print(f"  beta at A = 8/5 (freq extremum) = {b_star:.8f}")
        print(f"  s at the frequency extremum     = {out['s_at_star']:.7f}"
              f"   <- NOT 1, so the two markers are distinct")
        print(f"  A at tidal parity               = {out['A_at_tidal']:.8f}"
              f"   <- NOT 2, the 1+s split is O(mu^1/3) short")
        print(f"  nu/omega at tidal parity        = "
              f"{out['ratio_at_tidal']:.7f}")
        print(f"  beta_star / beta_tidal          = "
              f"{out['ratio_of_betas']:.4f}")
    return out


def table_across_systems(verbose: bool = True) -> list:
    """A_star and the rate ratios are mu-independent; the betas are not."""
    systems = [('Sun-Mercury', 1.66e-7), ('Sun-Earth', MU_SE),
               ('Sun-Jupiter', 9.537e-4), ('Earth-Moon', 1.215e-2)]
    from src.critical_beta import critical_beta_tidal_exact
    rows = []
    if verbose:
        print(f"  {'system':<13}{'mu':>12}{'beta(A=8/5)':>14}"
              f"{'beta(s=1)':>13}{'s at A=8/5':>12}{'nu/omega':>11}")
    for name, mu in systems:
        b85 = beta_at_frequency_extremum(mu)
        row = dict(system=name, mu=mu, beta_star=b85,
                   beta_tidal=critical_beta_tidal_exact(mu),
                   s_at_star=saddle_strength(b85, mu),
                   ratio=nu_over_omega(A_STAR))
        rows.append(row)
        if verbose:
            print(f"  {name:<13}{mu:>12.4e}{b85:>14.8f}"
                  f"{row['beta_tidal']:>13.8f}{row['s_at_star']:>12.6f}"
                  f"{row['ratio']:>11.7f}")
    if verbose:
        print("  nu/omega at the extremum is identical in every system: "
              "2*sqrt(2)/3.")
    return rows


if __name__ == '__main__':
    print("\n== Frequency ratio at the on-axis sail equilibrium ==\n")
    print("-- exact extremum, eqs. (5)-(7) " + "-" * 38)
    verify_extremum()
    print("\n-- resonance search " + "-" * 50)
    resonance_scan()
    print("\n-- Sun-Earth summary " + "-" * 49)
    summary()
    print("\n-- across systems " + "-" * 52)
    table_across_systems()
    print()
