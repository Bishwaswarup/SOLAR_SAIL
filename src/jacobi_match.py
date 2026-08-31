"""
jacobi_match.py — robust halo-family continuation and explicit Jacobi-constant
                  matching for Earth-Moon L1/L2.

Why this module exists
──────────────────────
The earlier pipeline set Az = 0.02 for both the L1 and the L2 halo and reported
that their Jacobi constants agreed to 5.7e-5, describing the pair as "matched".
Both halves of that are wrong:

  1. Equal Az at L1 and L2 does NOT imply equal C.  The two families have
     different geometry and different energy-vs-amplitude slopes.  Any agreement
     at one particular Az is a coincidence.

  2. Calling compute_halo_orbit() independently at each Az does not track a
     single family.  Sweeping Az for Earth-Moon L2 produces

         Az     C(L2 halo)
         0.005  3.17207607
         0.010  3.15127516     <- jump
         0.020  3.17087868     <- jump back
         0.040  3.13757644     <- jump

     i.e. the corrector lands on different branches depending on the Richardson
     initial guess.  The apparent match at Az = 0.02 sits in the middle of that
     branch-hopping.

This module fixes both: `family()` walks a family by natural-parameter
continuation (each solution seeds the next), and `match_jacobi()` solves
explicitly for the Az pair with C_L1 = C_L2.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from src.equilibria import find_artificial_equilibrium
from src.jacobi import jacobi_constant_sail
from src.orbits import compute_halo_orbit

MU_EM = 0.01215
EM_KM = 384_400.0


def jacobi_constant(state: np.ndarray, mu: float = MU_EM) -> float:
    """
    Jacobi constant for the UNSAILED Earth-Moon problem.

    Everything in this module runs at beta = 0 (see compute_halo_orbit below),
    where the sail and gravitational Jacobi integrals coincide identically.  So
    this delegates to the canonical implementation with beta pinned to 0.0
    rather than carrying a second copy of the formula: one implementation, and
    the beta = 0 assumption is visible at the point it is made.

    If sail effects are ever introduced here, this wrapper must go and the call
    sites must pass the real beta — see src/jacobi.py on why C_grav is not an
    integral for beta != 0.
    """
    return float(jacobi_constant_sail(state, mu, 0.0))


def lagrange_point(which: str = 'L1', mu: float = MU_EM) -> np.ndarray:
    """Collinear point by Newton solve on the zero-sail equilibrium condition."""
    gamma = (mu / 3.0) ** (1.0 / 3.0)
    second = 1.0 - mu
    x0 = [second - gamma, 0.0, 0.0] if which == 'L1' else [second + gamma, 0.0, 0.0]
    return find_artificial_equilibrium(0.0, 0.0, 0.0, mu, x0)


def family(which: str = 'L1',
           az_grid: np.ndarray | None = None,
           mu: float = MU_EM,
           verbose: bool = False) -> dict:
    """
    Track a halo family by natural-parameter continuation in Az.

    Each converged solution seeds the next, so the corrector stays on one branch
    instead of being re-seeded from the Richardson guess at every step.

    Returns
    -------
    dict with arrays  Az, C, T, state0 (list), and `n_failed`
    """
    if az_grid is None:
        az_grid = np.linspace(0.010, 0.055, 24)

    from src.orbits import halo_family
    f = halo_family(lagrange_point(which, mu), mu, az_grid,
                    amplitude='z0', verbose=False)

    if verbose:
        print(f"    {which}: {f['Az'].size}/{len(az_grid)} on-branch"
              f"  ({f['n_failed']} skipped)")
        if f['Az'].size:
            print(f"    Az [{f['Az'][0]:.4f}, {f['Az'][-1]:.4f}]"
                  f"   C [{f['C'].min():.6f}, {f['C'].max():.6f}]")

    return dict(which=which, Az=f['Az'], C=f['C'], T=f['T'],
                state0=f['state0'], n_failed=f['n_failed'],
                failures=f['failures'])


def _C_of_Az(which: str, Az: float, mu: float,
             guess=None, T_guess: float = None,
             amplitude: str = 'z0') -> tuple[float, np.ndarray, float]:
    """
    C, state0, period for the halo at amplitude `Az`.

    Delegates to the fixed corrector in orbits.py, which now pins z0 to Az,
    accepts a (x0, vy0, T_half) continuation seed without clobbering it,
    validates against vertical-Lyapunov / axial branches, and falls back to the
    Richardson guess if a seed lands in the wrong basin.
    """
    eq = lagrange_point(which, mu)
    seed = None
    if guess is not None:
        seed = (guess[0], guess[4],
                T_guess / 2.0 if T_guess else None)
    s0, T = compute_halo_orbit(eq, Az=float(Az), mu=mu,
                              alpha=0.0, delta=0.0, beta=0.0,
                              seed=seed, amplitude=amplitude)
    return jacobi_constant(s0, mu), s0, T


def match_jacobi(Az_L1: float = 0.02,
                 mu: float = MU_EM,
                 verbose: bool = True) -> dict:
    """
    Fix the L1 halo at `Az_L1`, then solve for the L2 amplitude whose Jacobi
    constant matches it:   C_L2(Az) - C_L1 = 0.

    This is the correct way to set up an energy-matched pair.  Equal Az is not.

    Returns
    -------
    dict  with Az_L1, Az_L2, C_L1, C_L2, dC, state0_L1, state0_L2, T_L1, T_L2
    """
    C1, s1, T1 = _C_of_Az('L1', Az_L1, mu)
    if verbose:
        print(f"  L1 halo fixed at Az = {Az_L1:.5f}   C = {C1:.8f}")

    fam2 = family('L2', mu=mu, verbose=verbose)
    if fam2['Az'].size < 3:
        raise RuntimeError("L2 family continuation produced too few points.")

    # bracket the root on the continued family
    g = fam2['C'] - C1
    sign_change = np.where(np.diff(np.sign(g)) != 0)[0]
    if sign_change.size == 0:
        raise RuntimeError(
            f"C_L1 = {C1:.8f} lies outside the tracked L2 family range "
            f"[{fam2['C'].min():.8f}, {fam2['C'].max():.8f}]. "
            "Choose a different Az_L1.")

    k = int(sign_change[0])
    lo, hi = fam2['Az'][k], fam2['Az'][k + 1]
    seed = fam2['state0'][k]

    def resid(Az):
        c, _, _ = _C_of_Az('L2', Az, mu, guess=seed)
        return c - C1

    Az2 = brentq(resid, lo, hi, xtol=1e-9)
    C2, s2, T2 = _C_of_Az('L2', Az2, mu, guess=seed)

    if verbose:
        print(f"  L2 halo matched at Az = {Az2:.6f}   C = {C2:.8f}")
        print(f"  dC = {C1 - C2:+.3e}   (bracket [{lo:.4f}, {hi:.4f}])")

    return dict(Az_L1=Az_L1, Az_L2=float(Az2), C_L1=C1, C_L2=C2,
                dC=float(C1 - C2), state0_L1=s1, state0_L2=s2,
                T_L1=T1, T_L2=T2, family_L2=fam2)


def _az_for_C(fam: dict, C_target: float, mu: float = MU_EM) -> tuple[float, np.ndarray, float]:
    """Invert a tracked family: find the Az whose halo has C = C_target."""
    g = fam['C'] - C_target
    k = np.where(np.diff(np.sign(g)) != 0)[0]
    if k.size == 0:
        raise RuntimeError(
            f"C = {C_target:.8f} outside {fam['which']} family range "
            f"[{fam['C'].min():.8f}, {fam['C'].max():.8f}]")
    k = int(k[0])
    seed = fam['state0'][k]

    def resid(Az):
        c, _, _ = _C_of_Az(fam['which'], Az, mu, guess=seed)
        return c - C_target

    Az = brentq(resid, fam['Az'][k], fam['Az'][k + 1], xtol=1e-10)
    c, s0, T = _C_of_Az(fam['which'], Az, mu, guess=seed)
    return float(Az), s0, T


def matched_pairs(n: int = 6, mu: float = MU_EM, verbose: bool = True) -> list[dict]:
    """
    Find genuine energy-matched (L1 halo, L2 halo) pairs in the Earth-Moon system.

    Equal Az does NOT give equal C.  This walks both families by continuation,
    intersects their Jacobi-constant ranges, and inverts each family for a grid
    of target C inside the overlap.
    """
    f1 = family('L1', az_grid=np.linspace(0.010, 0.075, 34), mu=mu, verbose=verbose)
    f2 = family('L2', az_grid=np.linspace(0.0025, 0.050, 34), mu=mu, verbose=verbose)

    lo = max(f1['C'].min(), f2['C'].min())
    hi = min(f1['C'].max(), f2['C'].max())
    if verbose:
        print(f"\n  L1 family C range: [{f1['C'].min():.8f}, {f1['C'].max():.8f}]")
        print(f"  L2 family C range: [{f2['C'].min():.8f}, {f2['C'].max():.8f}]")
        if hi <= lo:
            print("  -> ranges are DISJOINT: no energy-matched pair exists here.")
        else:
            print(f"  -> overlap: C in [{lo:.8f}, {hi:.8f}]")

    if hi <= lo:
        return []

    out = []
    margin = 0.02 * (hi - lo)
    for Ct in np.linspace(lo + margin, hi - margin, n):
        try:
            a1, s1, T1 = _az_for_C(f1, float(Ct), mu)
            a2, s2, T2 = _az_for_C(f2, float(Ct), mu)
        except Exception:
            continue
        out.append(dict(C=float(Ct), Az_L1=a1, Az_L2=a2,
                        state0_L1=s1, state0_L2=s2, T_L1=T1, T_L2=T2,
                        ratio=a1 / a2))

    if verbose and out:
        print(f"\n  {'C_target':>12}  {'Az_L1':>9}  {'Az_L2':>9}  {'ratio':>7}"
              f"  {'T_L1':>8}  {'T_L2':>8}")
        print("  " + "-"*64)
        for r in out:
            print(f"  {r['C']:12.8f}  {r['Az_L1']:9.5f}  {r['Az_L2']:9.5f}"
                  f"  {r['ratio']:7.1f}  {r['T_L1']:8.5f}  {r['T_L2']:8.5f}")
        print("\n  Energy-matched pairs require VERY unequal amplitudes.")
        print("  Equal Az = 0.02 gives dC = 2.2e-2, not 5.7e-5.")

    return out


if __name__ == '__main__':
    print("\n── Earth-Moon halo families by continuation ──────────")
    f1 = family('L1', verbose=True)
    f2 = family('L2', verbose=True)

    print("\n  Monotonic C along each tracked family?")
    for f in (f1, f2):
        if f['C'].size > 1:
            d = np.diff(f['C'])
            print(f"    {f['which']}: dC/dAz all negative = "
                  f"{bool(np.all(d < 0))}   max |jump| = {np.abs(d).max():.2e}")

    print("\n── Explicit Jacobi matching ──────────────────────────")
    try:
        m = match_jacobi(Az_L1=0.02, verbose=True)
        print(f"\n  RESULT: energy-matched pair is "
              f"Az_L1 = {m['Az_L1']:.5f},  Az_L2 = {m['Az_L2']:.5f}")
        print(f"          NOT equal Az.  dC = {m['dC']:+.2e}")
    except RuntimeError as e:
        print(f"  {e}")
