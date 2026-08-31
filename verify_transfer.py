"""
verify_transfer.py — (a) is the 0.05 m/s "transfer" self-matching?
                     (b) do the Earth-Moon L1/L2 halos at Az=0.02 really share C?
"""
import numpy as np
from src.equilibria import find_artificial_equilibrium
from src.orbits     import compute_halo_orbit
from src.manifolds  import compute_manifold
from src.transfer   import poincare_section, match_manifolds, transfer_dv
from scipy.integrate import solve_ivp
from src.dynamics   import cr3bp_sail_eom

MU_SE = 3.003e-6
AU    = 1.495978707e8
VEL_SE = 29.7847          # km/s per non-dim velocity
MU_EM = 0.01215
EM_KM = 384_400.0
EM_VEL = 1.023

# ══════════════════════════════════════════════════════════════════════
print("="*72)
print("(a)  IS THE SUN-EARTH 'TRANSFER' SELF-MATCHING?")
print("="*72)

eq_c = find_artificial_equilibrium(0., 0., 0., MU_SE, [0.99, 0., 0.])
s0, T = compute_halo_orbit(eq_c, Az=0.003, mu=MU_SE, alpha=0., delta=0., beta=0.)
print(f"  Halo: x0 = {s0[0]:.8f}  T = {T:.6f}")
print(f"  main.py builds BOTH manifolds from this SAME state0.\n")

su_ = compute_manifold(s0, T, MU_SE, 0.,0.,0., 'unstable','+', 30, t_max=2.5*np.pi)
ss_ = compute_manifold(s0, T, MU_SE, 0.,0.,0., 'stable',  '+', 30, t_max=2.5*np.pi)

cu = poincare_section(su_, section='y', value=0.0, direction=0)
cs = poincare_section(ss_, section='y', value=0.0, direction=0)
(ib, jb), sU, sS, dv_vec = match_manifolds(cu, cs)
dv, _, pres = transfer_dv(sU, sS)
print(f"  Reproduced best match: |dV| = {dv*VEL_SE*1000:.3f} m/s"
      f"   pos_res = {pres*AU:.3f} km")

# reference: the halo orbit's own y=0 crossings
ref = solve_ivp(cr3bp_sail_eom, [0, T], s0, args=(0.,0.,0.,MU_SE),
                rtol=1e-12, atol=1e-12, dense_output=True)
ts = np.linspace(0, T, 20000)
Y  = ref.sol(ts)
sign = np.sign(Y[1])
idx  = np.where(np.diff(sign) != 0)[0]
orbit_cross = [Y[:, i] for i in idx]
print(f"  The halo itself crosses y=0 at {len(orbit_cross)} points:")
for k, oc in enumerate(orbit_cross):
    print(f"      x = {oc[0]:.8f}  z = {oc[2]:+.6f}")

# how far is the matched pair from the orbit itself?
def dist_to_orbit(st):
    return min(np.linalg.norm(st[:3] - oc[:3]) for oc in orbit_cross)

d_u = dist_to_orbit(sU)
d_s = dist_to_orbit(sS)
print(f"\n  Matched unstable point: distance to the halo = {d_u:.3e} nd"
      f" = {d_u*AU:.1f} km")
print(f"  Matched stable   point: distance to the halo = {d_s:.3e} nd"
      f" = {d_s*AU:.1f} km")
print(f"  Separation between the two matched points   = "
      f"{np.linalg.norm(sU[:3]-sS[:3])*AU:.3f} km")

# how far do the strands ever get from the orbit?
maxsep_u = max(max(np.linalg.norm(st[:3, k] - s0[:3]) for k in range(st.shape[1]))
               for st in su_)
print(f"\n  Max excursion of ANY unstable strand from x0 = "
       f"{maxsep_u:.3e} nd = {maxsep_u*AU:,.0f} km")
print(f"  (eps = 1e-6 nd = {1e-6*AU:.0f} km;  lambda_u = 1.25 over one period)")
print("\n  VERDICT: both manifolds emanate from the SAME periodic orbit, so the")
print("  y=0 section necessarily contains the orbit itself.  The matcher finds")
print("  that self-intersection.  This is not a transfer between two orbits.")

# ══════════════════════════════════════════════════════════════════════
print()
print("="*72)
print("(b)  EARTH-MOON L1/L2 HALOS AT Az=0.02 - DO THEY SHARE C?")
print("="*72)

def jacobi(state, mu):
    x, y, z, vx, vy, vz = state
    r1 = np.sqrt((x+mu)**2 + y**2 + z**2)
    r2 = np.sqrt((x-(1-mu))**2 + y**2 + z**2)
    U  = 0.5*(x*x + y*y) + (1-mu)/r1 + mu/r2
    return 2*U - (vx*vx + vy*vy + vz*vz)

def em_lag(which):
    g = (MU_EM/3.0)**(1.0/3.0)
    moon = 1.0 - MU_EM
    x0 = [moon-g, 0., 0.] if which == 'L1' else [moon+g, 0., 0.]
    return find_artificial_equilibrium(0., 0., 0., MU_EM, x0)

L1 = em_lag('L1'); L2 = em_lag('L2')
print(f"  L1 = {L1[0]:.8f}     L2 = {L2[0]:.8f}")
print(f"  C(L1 point) = {jacobi([L1[0],0,0,0,0,0], MU_EM):.8f}")
print(f"  C(L2 point) = {jacobi([L2[0],0,0,0,0,0], MU_EM):.8f}")
print()
print("   Az      C(L1 halo)    C(L2 halo)     dC          |dC| km-equiv")
print("  " + "-"*66)
rows = []
for Az in [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04]:
    try:
        s1, T1 = compute_halo_orbit(L1, Az=Az, mu=MU_EM, alpha=0., delta=0., beta=0.)
        s2, T2 = compute_halo_orbit(L2, Az=Az, mu=MU_EM, alpha=0., delta=0., beta=0.)
        c1 = jacobi(s1, MU_EM); c2 = jacobi(s2, MU_EM)
        rows.append((Az, c1, c2))
        print(f"  {Az:5.3f}   {c1:.8f}   {c2:.8f}   {c1-c2:+.3e}")
    except Exception as e:
        print(f"  {Az:5.3f}   FAILED: {str(e)[:44]}")

print()
print("  The claim in the README was C_L1=3.170935, C_L2=3.170879 at Az=0.02.")
print("  Compare against the table above.")
