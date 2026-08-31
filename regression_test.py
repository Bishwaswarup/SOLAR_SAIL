"""
regression_test.py — does the rewritten corrector still handle every case
                     the existing pipeline depends on?

Run from the SOLAR_SAIL folder:   python regression_test.py
Paste the whole output back.
"""
import numpy as np
from src.orbits import (compute_halo_orbit, halo_family,
                        HaloConvergenceError, HaloValidationError)
from src.equilibria import find_artificial_equilibrium
from src.jacobi import (jacobi_constant_sail,
                        jacobi_constant_gravitational,
                        jacobi_offset)

MU_SE = 3.003e-6
MU_EM = 0.01215
D = 365.25 / (2 * np.pi)


def show(tag, eq, mu, Az, beta, expect=None, **kw):
    try:
        s, P, i = compute_halo_orbit(eq, Az, mu, 0.0, 0.0, beta,
                                     return_info=True, **kw)
        print(f"  {tag}")
        print(f"      T={P:.6f}  x0={s[0]:.8f}  vy0={s[4]:.8f}")
        print(f"      resid={i['residual']:.1e}  iters={i['n_iter']}"
              f"  z_asym={i['z_asymmetry']:.4f}  max|z|={i['amplitude_achieved']:.6f}")
        if expect:
            print(f"      expected: {expect}")
        return True
    except (HaloConvergenceError, HaloValidationError, ValueError) as e:
        print(f"  {tag}")
        print(f"      *** {type(e).__name__}: {str(e)[:130]}")
        if expect:
            print(f"      expected: {expect}")
        return False


print("=" * 74)
print("1.  SUN-EARTH cases that main.py depends on")
print("=" * 74)
eqc = find_artificial_equilibrium(0., 0., 0.,  MU_SE, [0.99, 0., 0.])
eqs = find_artificial_equilibrium(0., 0., 0.5, MU_SE, [eqc[0] - 0.05, 0., 0.])
print(f"  eq_class x = {eqc[0]:.8f}     eq_sail x = {eqs[0]:.8f}")
print()
show("classical  beta=0.0   Az=0.003", eqc, MU_SE, 0.003, 0.0,
     "T=3.053122  x0=0.98897724  vy0=0.01006625")
show("sail       beta=0.5   Az=0.003", eqs, MU_SE, 0.003, 0.5,
     "T=6.281845  x0=0.79366856")

print()
print("=" * 74)
print("2.  BETA SWEEP  (fig1 / fig2 / fig3 / animations)")
print("=" * 74)
ok = 0
for b in [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]:
    xg = 0.99 - 0.42 * b
    try:
        eq = find_artificial_equilibrium(0., 0., b, MU_SE, [xg, 0., 0.])
        s, P, i = compute_halo_orbit(eq, 0.003, MU_SE, 0., 0., b,
                                     return_info=True)
        print(f"  beta={b:.2f}  x_eq={eq[0]:.8f}  T={P:.6f}"
              f"  ({P*D:7.1f} d)  it={i['n_iter']:2d}  asym={i['z_asymmetry']:.4f}")
        ok += 1
    except Exception as e:
        print(f"  beta={b:.2f}  *** {type(e).__name__}: {str(e)[:78]}")
print(f"  -> {ok}/7 converged")

print()
print("=" * 74)
print("3.  STATION-KEEPING nominal  (fig5sk uses beta=0.05)")
print("=" * 74)
eq5 = find_artificial_equilibrium(0., 0., 0.05, MU_SE, [0.98, 0., 0.])
show("beta=0.05  Az=0.008", eq5, MU_SE, 0.008, 0.05)

print()
print("=" * 74)
print("4.  EARTH-MOON: the spurious L2 orbit must be REJECTED")
print("=" * 74)
g = (MU_EM / 3.0) ** (1.0 / 3.0)
L1 = find_artificial_equilibrium(0., 0., 0., MU_EM, [1 - MU_EM - g, 0., 0.])
L2 = find_artificial_equilibrium(0., 0., 0., MU_EM, [1 - MU_EM + g, 0., 0.])
print(f"  L1 x = {L1[0]:.8f}     L2 x = {L2[0]:.8f}")
print()
show("EM-L1  Az=0.020  (should PASS, genuine halo)", L1, MU_EM, 0.020, 0.0)
show("EM-L2  Az=0.020  (should FAIL: vertical-Lyapunov)", L2, MU_EM, 0.020, 0.0)
show("EM-L2  Az=0.020  require_halo=False (should PASS)", L2, MU_EM, 0.020, 0.0,
     require_halo=False)

print()
print("=" * 74)
print("5.  FAMILY CONTINUATION  (must be monotonic, no branch jumps)")
print("=" * 74)
for nm, eq in (("L1", L1), ("L2", L2)):
    f = halo_family(eq, MU_EM, np.linspace(0.010, 0.055, 19), verbose=True)
    if f['Az'].size > 1:
        d = np.diff(f['C'])
        print(f"  {nm}: n={f['Az'].size}  C=[{f['C'].min():.8f}, {f['C'].max():.8f}]")
        print(f"      monotonic={bool(np.all(d < 0))}  max|jump|={np.abs(d).max():.2e}")
    else:
        print(f"  {nm}: EMPTY  ({f['n_failed']} failures)")

print()
print("=" * 74)
print("6.  z0  vs  true max|z|  per family")
print("=" * 74)
for nm, eq in (("EM-L1", L1), ("EM-L2", L2)):
    print(f"  {nm}:")
    for Az in (0.015, 0.030, 0.045):
        try:
            s, P, i = compute_halo_orbit(eq, Az, MU_EM, 0., 0., 0.,
                                         return_info=True)
            print(f"      z0={Az:.3f} -> max|z|={i['amplitude_achieved']:.6f}"
                  f"  ratio={i['amplitude_achieved']/Az:.3f}"
                  f"  z(T/2)={i['z_half']:+.6f}")
        except Exception as e:
            print(f"      z0={Az:.3f} -> {type(e).__name__}")

print()
print("=" * 74)
print("7.  amplitude='max' mode  (true-amplitude targeting)")
print("=" * 74)
for nm, eq in (("EM-L1", L1), ("EM-L2", L2)):
    for Az in (0.020, 0.040):
        try:
            s, P, i = compute_halo_orbit(eq, Az, MU_EM, 0., 0., 0.,
                                         amplitude='max', return_info=True)
            print(f"  {nm} max|z|={Az:.3f}: z0={s[2]:.6f}"
                  f"  achieved={i['amplitude_achieved']:.6f}"
                  f"  T={P:.5f}  it={i['n_iter']}")
        except Exception as e:
            print(f"  {nm} max|z|={Az:.3f}: {type(e).__name__}: {str(e)[:60]}")

print()
print("=" * 74)
print("8.  CLOSED-FORM TIDAL-PARITY THRESHOLD  (headline result)")
print("=" * 74)
# Full assertions and the paper table live in test_critical_beta.py; this is the
# smoke check so the single `python regression_test.py` entry point covers it.
try:
    from src.critical_beta import (critical_beta_tidal,
                                   critical_beta_tidal_exact)
    worst, worst_mu = 0.0, None
    for _mu in np.logspace(-7.0, -2.0, 61):
        _d = abs(critical_beta_tidal_exact(_mu) - critical_beta_tidal(_mu))
        if _d > worst:
            worst, worst_mu = _d, _mu
    print(f"  beta_crit = 1 - (1 - mu^(1/3))^2        (exact, no root-finding)")
    print(f"  Sun-Earth                              = "
          f"{critical_beta_tidal_exact(MU_SE):.17f}")
    print(f"  vs brentq on s(beta)-1                 = "
          f"{critical_beta_tidal(MU_SE):.17f}")
    print(f"  worst |diff| over mu in [1e-7, 1e-2]   = {worst:.3e} "
          f"at mu = {worst_mu:.4e}")
    print(f"  -> {'PASS' if worst < 1e-14 else 'FAIL'} "
          f"(tolerance 1e-14, margin {1e-14 / worst:.0f}x)")
    print("  full table + 7 assertions: python test_critical_beta.py")
except Exception as e:
    print(f"  *** {type(e).__name__}: {str(e)[:110]}")

print()
print("=" * 74)
print("9.  JACOBI INTEGRAL: the reported C must be conserved on its own orbit")
print("=" * 74)
# Bug A2: continuation.py and orbits.py labelled sail orbits with the
# GRAVITATIONAL Jacobi constant, which is not an integral for beta != 0.  The
# values looked sensible, were smooth in Az and reproducible -- because they
# were always sampled at the same crossing.  This is the test that catches it:
# propagate a full period and assert the reported quantity does not move.
from scipy.integrate import solve_ivp
from src.dynamics import cr3bp_sail_eom

_TOL_SAIL = 1e-10        # absolute floor: C_sail must hold to integrator noise
_MIN_RATIO = 1e3         # and must beat C_grav by >=3 decades when beta != 0
#   Scale-free is the point: the bug signal (C_grav spread) is 1e-5 at
#   beta=0.001 and 1e-2 at beta=0.05, so a 3-decade separation catches it with
#   several decades to spare while tolerating integrator noise in C_sail.
_worst = 0.0
_fails = []
print(f"  {'beta':>7} {'Az':>10} {'C_sail spread':>15} {'C_grav spread':>15}"
      f" {'ratio':>10}")
for _b in (0.0, 0.001, 0.01, 0.05):
    try:
        _eq = find_artificial_equilibrium(0., 0., _b, MU_SE,
                                          [0.99 - 0.42 * _b, 0., 0.])
        _gam = abs((1.0 - MU_SE) - _eq[0])
        _Az = 0.25 * _gam
        _s0, _P = compute_halo_orbit(_eq, float(_Az), MU_SE, 0., 0., _b)
        _sol = solve_ivp(cr3bp_sail_eom, [0.0, _P], _s0,
                         args=(0.0, 0.0, _b, MU_SE),   # alpha, delta, beta, mu
                         rtol=1e-13, atol=1e-14, dense_output=True)
        _ts = np.linspace(0.0, _P, 240)
        _cs = np.array([jacobi_constant_sail(_sol.sol(t), MU_SE, _b)
                        for t in _ts])
        _cg = np.array([jacobi_constant_gravitational(_sol.sol(t), MU_SE)
                        for t in _ts])
        _ss = float(_cs.max() - _cs.min())
        _sg = float(_cg.max() - _cg.min())
        _worst = max(_worst, _ss)
        if _ss > _TOL_SAIL:
            _fails.append((_b, f'C_sail spread {_ss:.2e} > {_TOL_SAIL:g}'))
        if _b > 0.0 and _ss > 0.0 and (_sg / _ss) < _MIN_RATIO:
            _fails.append((_b, f'C_grav/C_sail ratio {_sg/_ss:.1e} '
                               f'< {_MIN_RATIO:g}'))
        print(f"  {_b:>7.3f} {_Az:>10.6f} {_ss:>15.3e} {_sg:>15.3e}"
              f" {(_sg/_ss if _ss > 0 else float('inf')):>10.1e}")
    except Exception as _e:
        print(f"  {_b:>7.3f}  *** {type(_e).__name__}: {str(_e)[:60]}")
        _fails.append((_b, None))

print(f"\n  C_sail conserved to < {_TOL_SAIL:g} AND >= {_MIN_RATIO:g}x tighter"
      f" than C_grav: {'PASS' if not _fails else 'FAIL'}")
for _f in _fails:
    print(f"      beta={_f[0]}: {_f[1]}")
print(f"  worst C_sail spread = {_worst:.3e}")

# The offset must account for the entire discrepancy, eq. (3).
_st = np.array([0.9887, 0.0, 5e-4, 0.0, 0.0089, 0.0])
for _b in (0.001, 0.05):
    _lhs = (jacobi_constant_gravitational(_st, MU_SE)
            - jacobi_constant_sail(_st, MU_SE, _b))
    _rhs = jacobi_offset(_st, MU_SE, _b)
    print(f"  beta={_b:<6} C_grav - C_sail = {_lhs:+.12e}   "
          f"2*beta*(1-mu)/r1 = {_rhs:+.12e}   "
          f"{'PASS' if abs(_lhs - _rhs) < 1e-14 else 'FAIL'}")

# beta = 0 must make the two identical.
_d0 = abs(jacobi_constant_gravitational(_st, MU_SE)
          - jacobi_constant_sail(_st, MU_SE, 0.0))
print(f"  beta=0: C_grav == C_sail to {_d0:.3e}  "
      f"{'PASS' if _d0 == 0.0 else 'FAIL'}")

# A legacy two-argument call must fail loudly, not return a wrong number.
try:
    jacobi_constant_sail(_st, MU_SE)
    print("  legacy 2-arg call: *** FAIL - it was accepted")
except TypeError:
    print("  legacy 2-arg call raises TypeError: PASS")


print()
print("done.")
