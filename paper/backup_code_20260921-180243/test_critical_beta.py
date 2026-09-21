"""
test_critical_beta.py — regression test for the closed-form tidal-parity
                        threshold, eq. (6) of src/critical_beta.py.

The claim under test
────────────────────
Imposing tidal parity  s(beta) = mu/r2^3 = 1,  i.e.  r2 = mu^(1/3),  on the
on-axis force balance

    x - (1-beta)(1-mu)/r1^2 + mu/r2^2 = 0,     r1 = x + mu,  r2 = (1-mu) - x

fixes x = (1-mu) - mu^(1/3), hence r1 = 1 - mu^(1/3) and mu/r2^2 = mu^(1/3).
The balance collapses to (1-mu) = (1-beta)(1-mu)/(1-mu^(1/3))^2, the (1-mu)
factor cancels identically, and

    beta_crit = 1 - (1 - mu^(1/3))^2 = mu^(1/3) (2 - mu^(1/3))            (6)

Why the numerical check is not redundant
────────────────────────────────────────
`critical_beta_tidal_exact` is pure algebra.  `critical_beta_tidal` brentq-solves
s(beta) - 1 = 0, and every evaluation of s goes through `equilibrium()` — itself a
brentq solve on the full three-body balance.  Agreement between the two therefore
tests the whole force-balance path, not just the closed form.  A regression in
`_f_axis`, `equilibrium`, or `saddle_strength` breaks this test.

Tolerance
─────────
TOL = 1e-14 absolute on beta_crit, as specified.  Achieved margin over
mu in [1e-7, 1e-2] is ~3e-16 — roughly 30x inside tolerance, and at the level of
the nested brentq's own resolution, so this is a machine-precision agreement
rather than a loose one.

Run
───
    python test_critical_beta.py        # asserts, then prints the paper table
    pytest test_critical_beta.py        # if pytest is installed
"""

from __future__ import annotations

import sys

import numpy as np

from src.critical_beta import (MU_SE, AU_KM, critical_beta_tidal,
                               critical_beta_tidal_exact, equilibrium,
                               hill_radius, saddle_strength)

# Absolute tolerance on beta_crit between the closed form and the brentq solve.
TOL = 1e-14

# The specified sweep: 61 points, log-spaced over five decades.
MU_GRID = np.logspace(-7.0, -2.0, 61)

# Real systems, for the paper table.  mu = m2 / (m1 + m2).
SYSTEMS = [
    ('Sun-Jupiter',   9.5388e-4),
    ('Sun-Earth',     MU_SE),
    ('Sun-Mars',      3.2271e-7),
    ('Earth-Moon',    1.215e-2),
    ('Mars-Phobos',   1.6604e-8),
    ('Jupiter-Io',    4.7045e-5),
    ('Saturn-Titan',  2.3659e-4),
    ('Pluto-Charon',  1.0480e-1),
]

# Mass ratios whose cube root is an exact binary-representable decimal, so the
# closed form must return an exactly representable threshold.  These anchor the
# algebra independently of any solver.
EXACT_ANCHORS = [
    (1e-6,  0.0199),        # mu^(1/3) = 0.01  -> 2(0.01) - 0.01^2
    (1e-3,  0.19),          # mu^(1/3) = 0.10  -> 2(0.10) - 0.10^2
    (1.0,   1.0),           # mu^(1/3) = 1.00  -> the sail cancels solar gravity
]


# ── the tests ─────────────────────────────────────────────────────────────────

def test_closed_form_matches_brentq_over_grid():
    """Eq. (6) agrees with the brentq solve to < TOL across mu in [1e-7, 1e-2]."""
    worst_mu, worst = None, 0.0
    for mu in MU_GRID:
        b_exact = critical_beta_tidal_exact(mu)
        b_num = critical_beta_tidal(mu)
        d = abs(b_exact - b_num)
        if d > worst:
            worst, worst_mu = d, mu
        assert d < TOL, (
            f"closed form and brentq disagree at mu={mu:.6e}: "
            f"exact={b_exact:.17f}  brentq={b_num:.17f}  |d|={d:.3e} >= {TOL:.0e}")
    return worst, worst_mu


def test_closed_form_matches_brentq_for_real_systems():
    """Same check at the mass ratios that actually appear in the paper table."""
    for name, mu in SYSTEMS:
        b_exact = critical_beta_tidal_exact(mu)
        b_num = critical_beta_tidal(mu)
        d = abs(b_exact - b_num)
        assert d < TOL, (
            f"{name} (mu={mu:.6e}): exact={b_exact:.17f} "
            f"brentq={b_num:.17f} |d|={d:.3e} >= {TOL:.0e}")


def test_tidal_parity_actually_holds():
    """
    The threshold must do what it claims: s(beta_crit) = 1 and r2 = mu^(1/3).

    This is the physical assertion, independent of brentq agreement — it would
    catch an algebra slip that happened to be self-consistent.
    """
    for mu in MU_GRID[::6]:
        b = critical_beta_tidal_exact(mu)
        assert abs(saddle_strength(b, mu) - 1.0) < 1e-12, (
            f"s(beta_crit) != 1 at mu={mu:.3e}: s={saddle_strength(b, mu):.17f}")
        r2 = (1.0 - mu) - equilibrium(b, mu)
        assert abs(r2 - mu ** (1.0 / 3.0)) < 1e-14, (
            f"r2 != mu^(1/3) at mu={mu:.3e}: r2={r2:.17e} "
            f"mu^(1/3)={mu ** (1.0 / 3.0):.17e}")


def test_r2_is_cuberoot_three_hill_radii():
    """r2_crit / r_H = 3^(1/3), independently of mu — eq. (5)."""
    for mu in MU_GRID[::6]:
        ratio = mu ** (1.0 / 3.0) / hill_radius(mu)
        assert abs(ratio - 3.0 ** (1.0 / 3.0)) < 1e-13, (
            f"r2/r_H = {ratio:.17f} at mu={mu:.3e}, expected {3.0 ** (1/3):.17f}")


def test_exact_anchors():
    """Cube roots that are exact decimals give exactly representable thresholds."""
    for mu, expected in EXACT_ANCHORS:
        b = critical_beta_tidal_exact(mu)
        assert abs(b - expected) < 1e-15, (
            f"mu={mu:g}: got {b:.17f}, expected exactly {expected}")


def test_mu_independence_of_the_cancelled_factor():
    """
    The (1-mu) cancellation is the structural claim.  Verify it by construction:
    beta_crit must be reproduced by a form that never references (1-mu) at all,
    and must be strictly monotone increasing in mu.
    """
    b = [critical_beta_tidal_exact(mu) for mu in MU_GRID]
    assert np.all(np.diff(b) > 0), "beta_crit is not monotone increasing in mu"
    for mu in MU_GRID[::6]:
        m = mu ** (1.0 / 3.0)
        assert abs(critical_beta_tidal_exact(mu) - (1.0 - (1.0 - m) ** 2)) < 1e-15


def test_brentq_bracket_covers_large_mu():
    """
    The bracket must adapt.  A hard-coded upper bound of 0.5 fails for every
    mu above (1 - 1/sqrt(2))^3 = 0.02513, which excludes Pluto-Charon.
    """
    for mu in (0.03, 0.104, 0.3):
        b_num = critical_beta_tidal(mu)      # must not raise
        assert abs(b_num - critical_beta_tidal_exact(mu)) < TOL


# ── the paper table ───────────────────────────────────────────────────────────

def paper_table() -> str:
    """Emit the table in a form that can go straight into the manuscript."""
    L = []
    L.append("Table N.  Tidal-parity threshold beta_crit = 1 - (1 - mu^(1/3))^2.")
    L.append("Closed form against an independent brentq solve of s(beta) = 1.")
    L.append("")
    L.append(f"  {'system':<14} {'mu':>12} {'mu^(1/3)':>11} "
             f"{'beta_crit (exact)':>19} {'beta_crit (brentq)':>20} {'|diff|':>10}")
    L.append("  " + "-" * 92)
    for name, mu in sorted(SYSTEMS, key=lambda r: r[1]):
        be = critical_beta_tidal_exact(mu)
        bn = critical_beta_tidal(mu)
        L.append(f"  {name:<14} {mu:12.6e} {mu ** (1 / 3):11.7f} "
                 f"{be:19.16f} {bn:20.16f} {abs(be - bn):10.2e}")
    L.append("")
    L.append("  Sun-Earth sits at beta_crit = 0.0286, inside the flown sail band")
    L.append("  (beta ~ 0.01-0.05).  Standoff at parity: r2 = mu^(1/3) = "
             f"{MU_SE ** (1 / 3) * AU_KM:,.0f} km = 3^(1/3) r_H.")
    return "\n".join(L)


def sweep_table() -> str:
    """The tolerance sweep, condensed to one line per decade."""
    L = []
    L.append(f"Agreement over mu in [1e-7, 1e-2]  ({MU_GRID.size} log-spaced "
             f"points, tolerance {TOL:.0e}):")
    L.append("")
    L.append(f"  {'mu':>10} {'beta_crit (exact)':>19} "
             f"{'beta_crit (brentq)':>20} {'|diff|':>10}")
    L.append("  " + "-" * 63)
    for mu in np.logspace(-7, -2, 6):
        be = critical_beta_tidal_exact(mu)
        bn = critical_beta_tidal(mu)
        L.append(f"  {mu:10.1e} {be:19.16f} {bn:20.16f} {abs(be - bn):10.2e}")
    return "\n".join(L)


# ── standalone runner ─────────────────────────────────────────────────────────

if __name__ == '__main__':
    TESTS = [
        test_closed_form_matches_brentq_over_grid,
        test_closed_form_matches_brentq_for_real_systems,
        test_tidal_parity_actually_holds,
        test_r2_is_cuberoot_three_hill_radii,
        test_exact_anchors,
        test_mu_independence_of_the_cancelled_factor,
        test_brentq_bracket_covers_large_mu,
    ]

    print("=" * 94)
    print("  Closed-form tidal-parity threshold — regression test")
    print("=" * 94)
    print()

    n_fail, worst = 0, None
    for t in TESTS:
        try:
            r = t()
            if r is not None:
                worst = r
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            n_fail += 1
            print(f"  FAIL  {t.__name__}")
            print(f"        {e}")

    print()
    if worst is not None:
        print(f"  Worst disagreement over the sweep: {worst[0]:.3e} "
              f"at mu = {worst[1]:.6e}   (tolerance {TOL:.0e}, "
              f"margin {TOL / worst[0]:.0f}x)")
        print()

    print(sweep_table())
    print()
    print(paper_table())
    print()
    print("=" * 94)
    print(f"  {len(TESTS) - n_fail}/{len(TESTS)} passed"
          + ("" if n_fail == 0 else f"   {n_fail} FAILED"))
    print("=" * 94)
    sys.exit(1 if n_fail else 0)
