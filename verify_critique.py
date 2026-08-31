"""
verify_critique.py — independent check of every claim in the referee critique.
Nothing here imports the project's own equilibrium solver except for cross-check.
"""
import numpy as np
from scipy.optimize import brentq

MU  = 3.003e-6
AU  = 1.495978707e8   # km

# ══════════════════════════════════════════════════════════════════════
# 1.  Is the sail "L1" a heliocentric hovering point?
# ══════════════════════════════════════════════════════════════════════
print("="*72)
print("1.  HOVERING POINT vs DISPLACED LAGRANGE POINT")
print("="*72)

def hover_radius(beta, mu=MU):
    """Equilibrium radius with the Earth DELETED entirely."""
    return ((1.0 - beta) * (1.0 - mu))**(1.0/3.0)

def f_axis(x, beta, mu=MU):
    """
    Full on-axis force balance, spacecraft sunward of Earth.
      centrifugal(+x) - reduced solar gravity + Earth pull(+x, outward)
    """
    r1 = x + mu
    r2 = (1.0 - mu) - x
    return x - (1.0 - beta)*(1.0 - mu)/r1**2 + mu/r2**2

def eq_axis(beta, mu=MU):
    """Solve the full balance for the sail equilibrium on the x-axis."""
    lo, hi = 1e-3, (1.0 - mu) - 1e-9
    return brentq(f_axis, lo, hi, args=(beta, mu), xtol=1e-15, rtol=1e-15)

beta = 0.5
x_full  = eq_axis(beta)
x_hover = hover_radius(beta)
x_pure  = 0.5**(1.0/3.0)

print(f"  Project value (results.txt)      x = 0.79367422")
print(f"  Recomputed full balance          x = {x_full:.8f}")
print(f"  No-Earth hovering  [(1-b)(1-mu)]^1/3 = {x_hover:.8f}")
print(f"  Naive 0.5^(1/3)                      = {x_pure:.8f}")
print()
print(f"  Earth's ENTIRE contribution   dx = {x_full - x_hover:+.3e} nd"
      f"  = {abs(x_full-x_hover)*AU:,.0f} km")

# Hill radius
r_hill = (MU/3.0)**(1.0/3.0)
d_earth = (1.0 - MU) - x_full
print(f"  Earth Hill radius             r_H = {r_hill:.6e} nd = {r_hill*AU:,.0f} km")
print(f"  Standoff from Earth             d = {d_earth:.6f} nd = {d_earth*AU:,.0f} km")
print(f"  Standoff in Hill radii          d/r_H = {d_earth/r_hill:.2f}")
print()
x_cl = eq_axis(0.0)
print(f"  For reference, classical L1      x = {x_cl:.8f}"
      f"   d/r_H = {((1-MU)-x_cl)/r_hill:.4f}")
print("  -> classical L1 sits just INSIDE the Hill sphere (d/r_H < 1)")

# ══════════════════════════════════════════════════════════════════════
# 2.  Epicyclic frequency: is the 1-year period forced?
# ══════════════════════════════════════════════════════════════════════
print()
print("="*72)
print("2.  IS THE ONE-YEAR PERIOD A RESONANCE OR FORCED?")
print("="*72)

def A_param(beta, mu=MU, x=None):
    """A = (1-b)(1-mu)/r1^3 + mu/r2^3.  Saddle exists iff A > 1."""
    if x is None:
        x = eq_axis(beta, mu)
    r1 = x + mu
    r2 = (1.0 - mu) - x
    return (1.0-beta)*(1.0-mu)/r1**3 + mu/r2**3

def linear_freqs(beta, mu=MU):
    """
    Eigenvalues of the linearised system about the on-axis equilibrium.
    Uxx = 1+2A, Uyy = 1-A, Uzz = -A   (sail is conservative & radial at alpha=0)
    Planar:  lam^4 + (2-A) lam^2 + (1+A-2A^2) = 0
    Vertical: lam^2 = -A  ->  freq sqrt(A)
    """
    A = A_param(beta, mu)
    b = 2.0 - A
    c = 1.0 + A - 2.0*A**2
    disc = b*b - 4.0*c
    lam2 = np.roots([1.0, b, c])
    return A, lam2, np.sqrt(A)

for b in (0.0, 0.5):
    A, lam2, nu = linear_freqs(b)
    x = eq_axis(b)
    print(f"  beta = {b}:   x = {x:.8f}   A = {A:.8f}")
    print(f"     lambda^2 roots = {lam2}")
    for L2 in lam2:
        L2 = complex(L2)
        if L2.real > 0 and abs(L2.imag) < 1e-12:
            lam = np.sqrt(L2.real)
            print(f"     REAL saddle  lambda = +/-{lam:.6f}   "
                  f"e-fold = {1.0/lam:.4f} nd = {1.0/lam*365.25/(2*np.pi):.1f} d")
        elif L2.real < 0 and abs(L2.imag) < 1e-12:
            w = np.sqrt(-L2.real)
            print(f"     CENTRE  omega = {w:.8f}   "
                  f"period = {2*np.pi/w:.6f} nd = {2*np.pi/w*365.25/(2*np.pi):.2f} d")
    print(f"     vertical  nu = {nu:.8f}   period = {2*np.pi/nu:.6f} nd")
    print()

print("  Kepler check: at the no-Earth hovering point r1^3 = (1-b)(1-mu) exactly,")
print("  so (1-b)(1-mu)/r1^3 = 1 identically  =>  A = 1 + mu/r2^3.")
print("  A -> 1 gives Uyy -> 0, Uzz -> -1: epicyclic freq = mean motion = 1.")
print("  The 2*pi period is FORCED by the 1/r^2 field, not a resonance.")

# ══════════════════════════════════════════════════════════════════════
# 3.  Saddle strength and the critical beta
# ══════════════════════════════════════════════════════════════════════
print()
print("="*72)
print("3.  SADDLE STRENGTH s(beta) = A - 1 = mu/r2^3   AND CRITICAL BETA")
print("="*72)

def saddle_strength(beta, mu=MU):
    x = eq_axis(beta, mu)
    r2 = (1.0 - mu) - x
    return mu/r2**3

print("   beta      x_eq        d/r_H     A          s = A-1     lam_u")
print("  " + "-"*68)
for b in [0.0, 1e-4, 4.2e-4, 1e-3, 0.005, 0.01, 0.0287, 0.05, 0.1, 0.2, 0.3, 0.5]:
    x = eq_axis(b)
    d = (1.0-MU) - x
    A = A_param(b)
    s = saddle_strength(b)
    lam2 = np.roots([1.0, 2.0-A, 1.0+A-2.0*A**2])
    lam_u = 0.0
    for L2 in lam2:
        L2 = complex(L2)
        if L2.real > 0 and abs(L2.imag) < 1e-12:
            lam_u = np.sqrt(L2.real)
    print(f"  {b:7.5f}  {x:.8f}  {d/r_hill:8.3f}  {A:9.5f}  {s:10.3e}  {lam_u:8.5f}")

# critical beta: standoff = Hill radius
g1 = lambda b: ((1.0-MU) - eq_axis(b)) - r_hill
b_hill = brentq(g1, 1e-8, 0.4, xtol=1e-14)
# critical beta: saddle strength drops to unity  (Earth term == solar term)
g2 = lambda b: saddle_strength(b) - 1.0
b_sad = brentq(g2, 1e-6, 0.4, xtol=1e-14)

print()
print(f"  CRITERION A - equilibrium exits Earth's Hill sphere (d = r_H):")
print(f"      beta_crit = {b_hill:.6e}   x = {eq_axis(b_hill):.8f}")
print(f"      -> essentially ANY sail pushes the point out; L1 already sits at d/r_H = 0.997")
print()
print(f"  CRITERION B - saddle strength falls to unity (mu/r2^3 = 1,")
print(f"                i.e. Earth's tidal term equals the reduced solar term):")
print(f"      beta_crit = {b_sad:.6f}   x = {eq_axis(b_sad):.8f}")
print(f"      standoff  = {((1-MU)-eq_axis(b_sad))*AU:,.0f} km"
      f"  = {((1-MU)-eq_axis(b_sad))/r_hill:.3f} r_H")
print(f"      r2 = mu^(1/3) = {MU**(1/3):.6f}  (exact analytic value)")
print()
print(f"  NOTE: beta_crit(B) = {b_sad:.4f} lies INSIDE current sail technology")
print(f"        (beta ~ 0.01-0.05).  That is the publishable statement.")
