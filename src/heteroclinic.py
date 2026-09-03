"""
heteroclinic.py — Earth-Moon L1/L2 heteroclinic connection figures.

Computes halo orbits around the Earth-Moon L1 and L2 points, propagates
their unstable/stable manifold tubes, and finds the near-heteroclinic
connection at the Moon Poincaré section (x = 1 − μ).

What this module claims, and what it does not
────────────────────────────────────────────
An earlier version of this file asserted that equal amplitudes Az = 0.02 give
the L1 and L2 halos "essentially the same Jacobi constant (ΔC < 10⁻⁴)", and
reported a ΔV ≈ 500 m/s "heteroclinic" transfer.  Every part of that was wrong:

  1. Equal Az does NOT imply equal C.  The two families have different geometry
     and different energy-vs-amplitude slopes.  At Az = 0.02 the true gap is
     ΔC = 2.2e-2, three orders of magnitude larger than claimed.

  2. The Az = 0.02 "L2 halo" was not a halo.  The corrector had landed on the
     vertical-Lyapunov branch (T = 3.5195 against the true family's 3.4091,
     z-asymmetry ~ 1e-11).  orbits.compute_halo_orbit(require_halo=True) now
     rejects it outright.

  3. A GENUINE heteroclinic connection costs ZERO ΔV by construction — it is a
     single orbit asymptotic to both ends.  Any nonzero ΔV means the manifolds
     did not intersect, so the result is a manifold-guided two-impulse transfer
     and must be reported as one.  It is not a heteroclinic connection.

What is computed here
─────────────────────
A genuine energy-matched pair, obtained by walking BOTH halo families with
pseudo-arclength continuation (continuation.py) and solving for the amplitudes
that share a Jacobi constant.  Energy matching is a NECESSARY condition for a
heteroclinic connection: W^u(L1 halo) and W^s(L2 halo) can only intersect if
both orbits lie on the same energy surface.  It is not sufficient — the tubes
must also intersect in the remaining phase-space directions.

Matched pair used by default (see matched_pair()):

    C      = 3.1112724126        (identical for both, to 8.9e-16)
    L1:  Az = 0.09315184   T = 2.783249 nd = 12.10 d   z-asymmetry 0.2575
    L2:  Az = 0.06193811   T = 3.325038 nd = 14.46 d   z-asymmetry 0.6344

The amplitude ratio is 1.504 — energy matching requires markedly UNEQUAL
amplitudes, which is the whole point.  Both orbits pass require_halo=True, so
neither is a vertical-Lyapunov member.

The reported ΔV is a SAMPLING UPPER BOUND, not a converged cost
───────────────────────────────────────────────────────────────
Both manifolds are sampled at a finite number of strands, so the "best pair" on
the section is whichever two SAMPLED points happen to be closest.  That is a
property of the sampling, not of the dynamics, and it does not converge as the
grid is refined — it decreases:

    n_strands   t_max     ΔV [m/s]   residual [km]
        60      5π          1891         383
       120      5π          1891         383
       240      5π          1891         383
       120      8π          2637         876
       240      8π          1929         739
       360      8π           388          24

Refining the grid keeps finding closer pairs, and the position residual falls
with it — the signature of tubes that genuinely (or very nearly) intersect, with
the discrete sampling as the binding constraint.  So:

  * quote the ΔV as an UPPER BOUND at a stated sampling, never as "the" cost;
  * do not read a physical difference into 1891 vs 388 m/s — that is grid, not
    dynamics;
  * establishing a genuine heteroclinic connection needs a TARGETING root-solve
    on the tube intersection (drive the section-to-section residual to zero with
    the strand phase and the manifold-crossing time as unknowns), not a denser
    scan.  That is the remaining piece of work, and it is not done here.

The earlier "544 m/s" and "≈ 500 m/s" figures were quoted with none of these
qualifications, from a spurious orbit pair, and should not be reused.

Self-intersection guard
───────────────────────
W^u and W^s of the same orbit both contain that orbit, so matching crossings
drawn from one orbit returns the orbit's own self-intersection: dr -> 0,
dv -> 0, and an apparently free "transfer" that is not a transfer.  That is the
defect still present in main.py.  Here the two manifolds come from two DISTINCT
orbits, and match_manifolds() is additionally given exclude_states/min_sep so
no crossing still sitting on its originating halo can be selected.

Poincaré section: x = 1 − μ  (the Moon's x-position)

Figures
───────
  fig_poincare_map()      → fig6_poincare_map.png   (y–ẏ phase portrait, white)
  fig_manifold_transfer() → fig7_manifold_transfer.png  (x–y rotating frame, dark)
"""

import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from src.equilibria import find_artificial_equilibrium
from src.orbits     import compute_halo_orbit
# Earth-Moon here is UNSAILED (beta = 0), so the sail and gravitational Jacobi
# integrals coincide; beta = 0.0 is passed explicitly because jacobi.py now
# requires it rather than letting a sail orbit be mislabelled by default.
from src.jacobi     import jacobi_constant_sail
from src.manifolds  import compute_manifold
from src.dynamics   import cr3bp_sail_eom
from src.transfer   import poincare_section, match_manifolds, transfer_dv

# ── Earth-Moon constants ───────────────────────────────────────────────────────
MU_EM   = 0.01215       # Moon / (Earth + Moon)
EM_KM   = 384_400.0     # 1 non-dim length [km]
EM_VEL  = 1.023         # characteristic velocity [km/s] = 2π·EM_KM / T_moon

# Genuine energy-matched pair (see matched_pair()).  These are the amplitudes at
# which C_L1 == C_L2 to machine precision; they are deliberately unequal.
# The former single _AZ_DEFAULT = 0.02 for BOTH orbits gave dC = 2.2e-2 and an
# L2 orbit that was not a halo at all.
_AZ_L1_DEFAULT = 0.09315183816518526
_AZ_L2_DEFAULT = 0.06193811315139242
_C_MATCHED     = 3.1112724126481726

# Cached corrector seeds (x0, vy0, T_half) for the pair above.  Needed because
# the Richardson guess cannot reach these amplitudes on its own (0.62 and 0.37
# of the local scale gamma) — it is exactly why pseudo-arclength continuation is
# used to find them.  Without a seed, compute_halo_orbit raises
# HaloConvergenceError, so the amplitudes alone would be unusable.
_SEED_L1 = (np.float64(0.8271415053607784), np.float64(0.20821125695792386), 1.3916243743698842)
_SEED_L2 = (np.float64(1.086503462503645), np.float64(0.2706816932045701), 1.6625189686963568)

# Minimum separation a section crossing must keep from either originating halo
# before it may be selected as a transfer endpoint, in non-dimensional units.
# 2.6e-3 nd = 1,000 km: comfortably beyond the eps = 1e-6 nd (0.38 km) manifold
# seed displacement, so a strand that has not actually departed is excluded.
_MIN_SEP_DEFAULT = 1000.0 / 384_400.0

# Dark palette — retained for fig_manifold_transfer (figure 7) only.
_DARK   = '#0d1117'
_ORANGE = '#FF6B35'
_TEAL   = '#4ECDC4'
_YELLOW = '#FFE66D'
_WHITE  = '#e8e8e8'
_GREY   = '#555555'

# Light palette — used by fig_poincare_map (figure 6).  Okabe-Ito hues, which
# stay distinguishable in greyscale print and under the common CVD types.
_LIGHT       = 'white'
_ORANGE_L    = '#D55E00'
_TEAL_L      = '#0072B2'
_INK         = '#1a1a1a'
_GREY_L      = '#555555'
_GREY_FAINT  = '#999999'


# ── helpers ────────────────────────────────────────────────────────────────────

def find_em_lagrange(which: str = 'L1') -> np.ndarray:
    """Return the exact Earth-Moon L1 or L2 collinear equilibrium (no sail)."""
    gamma = (MU_EM / 3.0) ** (1.0 / 3.0)
    moon  = 1.0 - MU_EM
    x0    = [moon - gamma, 0.0, 0.0] if which == 'L1' else [moon + gamma, 0.0, 0.0]
    return find_artificial_equilibrium(0.0, 0.0, 0.0, MU_EM, x0)


def matched_pair(mu: float = MU_EM, n_probe: int = 60,
                 verbose: bool = True) -> dict:
    """
    Derive a genuine energy-matched (L1 halo, L2 halo) pair from scratch.

    Method
    ──────
    1. Walk each family by PSEUDO-ARCLENGTH continuation, seeded from a
       validated halo.  Natural-parameter stepping is not adequate: it loses the
       branch at folds, which is how the earlier pipeline ended up comparing an
       L1 halo against an L2 vertical-Lyapunov orbit.
    2. Intersect the two families' Jacobi-constant ranges, and over that overlap
       pick the target C whose amplitude ratio Az_L1/Az_L2 is closest to 1 — the
       most balanced pair available.
    3. Interpolate each branch to that C for a seed, Newton-correct with
       require_halo=True, then brentq the L2 amplitude so C_L2 == C_L1 exactly.

    Returns
    -------
    dict with C, dC, ratio, and per-point sub-dicts 'L1'/'L2' carrying
    Az, state0, T, C, z_asym, gamma, eq.
    """
    from src.continuation import continue_branch, find_halo_seed, local_scale
    from src.jacobi import jacobi_constant_sail
    _C = lambda st, m: jacobi_constant_sail(st, m, 0.0)

    branches = {}
    for which in ('L1', 'L2'):
        eq = find_em_lagrange(which) if mu == MU_EM else None
        if eq is None:
            raise ValueError("matched_pair() is specific to the Earth-Moon mu")
        g = local_scale(eq, mu)
        s0, Th, _ = find_halo_seed(eq, mu, verbose=False)
        branches[which] = continue_branch(
            eq, mu, param='Az', seed_state=(s0, Th),
            ds=0.02 * g, ds_min=1e-4 * g, ds_max=0.2 * g,
            n_steps=140, lam_bounds=(1e-3 * g, 0.75 * g),
            with_stability=False, verbose=False)
        if verbose:
            br = branches[which]
            print(f"    {which}: {len(br['Az'])} members  "
                  f"Az [{br['Az'].min():.5f}, {br['Az'].max():.5f}]  "
                  f"C [{br['C'].min():.8f}, {br['C'].max():.8f}]  "
                  f"{len(br['folds'])} folds")

    def _at_C(br, C_t):
        """Interpolate a branch to C = C_t.  Returns (Az, x0, vy0, T_half)."""
        C, Az, x0, vy0, T = br['C'], br['Az'], br['x0'], br['vy0'], br['T']
        for i in range(len(C) - 1):
            if (C[i] - C_t) * (C[i + 1] - C_t) <= 0 and C[i] != C[i + 1]:
                w = (C_t - C[i]) / (C[i + 1] - C[i])
                return (Az[i] + w * (Az[i + 1] - Az[i]),
                        x0[i] + w * (x0[i + 1] - x0[i]),
                        vy0[i] + w * (vy0[i + 1] - vy0[i]),
                        (T[i] + w * (T[i + 1] - T[i])) / 2.0)
        return None

    lo = max(branches['L1']['C'].min(), branches['L2']['C'].min())
    hi = min(branches['L1']['C'].max(), branches['L2']['C'].max())
    if hi <= lo:
        raise RuntimeError(
            f"the two families' Jacobi ranges are DISJOINT "
            f"(L1 [{branches['L1']['C'].min():.8f}, {branches['L1']['C'].max():.8f}], "
            f"L2 [{branches['L2']['C'].min():.8f}, {branches['L2']['C'].max():.8f}]): "
            f"no energy-matched pair exists, so no heteroclinic connection can.")
    if verbose:
        print(f"    C overlap: [{lo:.8f}, {hi:.8f}]  width {hi - lo:.6f}")

    best = None
    for C_t in np.linspace(lo + 1e-4, hi - 1e-4, n_probe):
        p1, p2 = _at_C(branches['L1'], C_t), _at_C(branches['L2'], C_t)
        if not p1 or not p2:
            continue
        r = p1[0] / p2[0]
        if best is None or abs(np.log(r)) < abs(np.log(best[0])):
            best = (r, C_t, p1, p2)
    if best is None:
        raise RuntimeError("no C in the overlap inverted on both families")
    _, C_t, p1, p2 = best

    out = {}
    for which, pp in (('L1', p1), ('L2', p2)):
        eq = find_em_lagrange(which)
        s0, P, info = compute_halo_orbit(
            eq, float(pp[0]), mu, 0., 0., 0.,
            seed=(pp[1], pp[2], pp[3]), retry_unseeded=False,
            require_halo=True, return_info=True)
        out[which] = dict(Az=float(pp[0]), state0=s0, T=float(P),
                          C=float(_C(s0, mu)),
                          z_asym=float(info['z_asymmetry']),
                          gamma=local_scale(eq, mu), eq=eq)

    # Drive C_L2 onto C_L1 exactly.
    C1 = out['L1']['C']
    eq2, s2_0, T2_0 = out['L2']['eq'], out['L2']['state0'], out['L2']['T']

    def _resid(Az):
        st, _ = compute_halo_orbit(eq2, float(Az), mu, 0., 0., 0.,
                                   seed=(s2_0[0], s2_0[4], T2_0 / 2.0),
                                   retry_unseeded=False, require_halo=True)
        return _C(st, mu) - C1

    Az2_0 = out['L2']['Az']
    Az2 = brentq(_resid, Az2_0 * 0.998, Az2_0 * 1.002, xtol=1e-12)
    s2, P2, i2 = compute_halo_orbit(eq2, float(Az2), mu, 0., 0., 0.,
                                    seed=(s2_0[0], s2_0[4], T2_0 / 2.0),
                                    retry_unseeded=False, require_halo=True,
                                    return_info=True)
    out['L2'].update(Az=float(Az2), state0=s2, T=float(P2),
                     C=float(_C(s2, mu)), z_asym=float(i2['z_asymmetry']))

    out['C'] = out['L1']['C']
    out['dC'] = out['L1']['C'] - out['L2']['C']
    out['ratio'] = out['L1']['Az'] / out['L2']['Az']
    if verbose:
        print(f"    MATCHED  C = {out['C']:.10f}   dC = {out['dC']:+.3e}")
        for w in ('L1', 'L2'):
            d = out[w]
            print(f"      {w}: Az = {d['Az']:.8f} ({d['Az']/d['gamma']:.3f} gamma)"
                  f"  T = {d['T']:.6f} nd = {d['T']*27.32/(2*np.pi):.2f} d"
                  f"  z_asym = {d['z_asym']:.4f}")
        print(f"    amplitude ratio = {out['ratio']:.4f}  "
              f"(equal amplitudes would give dC = 2.2e-2)")
    return out


def _full_orbit(state0: np.ndarray, period: float, n: int = 600) -> np.ndarray:
    """Integrate one halo period; return (6, n) array."""
    res = solve_ivp(
        cr3bp_sail_eom, [0.0, period], state0,
        args=(0.0, 0.0, 0.0, MU_EM),
        t_eval=np.linspace(0.0, period, n),
        rtol=1e-11, atol=1e-11,
    )
    return res.y


def _compute_manifolds(Az_L1=_AZ_L1_DEFAULT, Az_L2=_AZ_L2_DEFAULT,
                       pair=None, n_strands=60, t_max=5.0*np.pi,
                       min_sep=_MIN_SEP_DEFAULT, verbose=True):
    """
    Compute all manifold strands and Poincaré crossings at x = 1-mu (Moon).

    Takes TWO amplitudes, one per point.  The single shared Az of the earlier
    version is what produced the dC = 2.2e-2 mismatch and the spurious
    vertical-Lyapunov L2 orbit; energy matching requires unequal amplitudes.

    Pass `pair=matched_pair()` to derive the amplitudes from scratch, or rely on
    the cached defaults.  Either way both orbits are re-solved here with
    require_halo=True, so a vertical-Lyapunov branch cannot slip through.

    Returns a dict with all intermediate quantities needed by both figure
    functions so they can share computation when called together.

    Notes
    -----
    * L1 unstable '+' branch travels toward the Moon.  The '-' branch goes
      toward Earth and is not useful for an L1->L2 transfer, so we skip it.
    * L2 stable '+' and '-' branches both cross the Moon section.
    * Filter: keep crossings with |y| < 0.3 and |z| < 0.2 to exclude strands
      that escape the Earth-Moon system.
    * `exclude` carries points sampled along BOTH originating halos; any
      crossing within `min_sep` of one of them is refused as a transfer
      endpoint, so a strand that has not actually departed cannot be selected.
    """
    if pair is not None:
        Az_L1, Az_L2 = pair['L1']['Az'], pair['L2']['Az']

    if verbose:
        print("  Finding Earth-Moon L1 and L2 ...")
    L1 = find_em_lagrange('L1')
    L2 = find_em_lagrange('L2')
    if verbose:
        print(f"    L1 = ({L1[0]:.5f}, 0)    L2 = ({L2[0]:.5f}, 0)")

    # Seeds: from the supplied pair, or the cached ones when the default
    # amplitudes are in use.  A seed is mandatory at these amplitudes.
    if pair is not None:
        seeds = {w: (pair[w]['state0'][0], pair[w]['state0'][4],
                     pair[w]['T'] / 2.0) for w in ('L1', 'L2')}
    elif (Az_L1, Az_L2) == (_AZ_L1_DEFAULT, _AZ_L2_DEFAULT):
        seeds = {'L1': _SEED_L1, 'L2': _SEED_L2}
    else:
        seeds = {}

    orbits = {}
    for w, eq, Az in (('L1', L1, Az_L1), ('L2', L2, Az_L2)):
        if verbose:
            print(f"  Computing {w} halo  (Az = {Az:.8f}) ...")
        st, P, info = compute_halo_orbit(
            eq, float(Az), MU_EM, 0., 0., 0.,
            seed=seeds.get(w), retry_unseeded=seeds.get(w) is None,
            require_halo=True, return_info=True)
        orbits[w] = (st, P, info)
        if verbose:
            print(f"    period = {P:.6f} nd  ({P * 27.32 / (2*np.pi):.2f} days)"
                  f"   C = {jacobi_constant_sail(st, MU_EM, 0.0):.10f}"
                  f"   z-asymmetry = {info['z_asymmetry']:.4f}")

    s1, T1, _ = orbits['L1']
    s2, T2, _ = orbits['L2']
    C1 = jacobi_constant_sail(s1, MU_EM, 0.0)
    C2 = jacobi_constant_sail(s2, MU_EM, 0.0)
    if verbose:
        print(f"  Energy match: dC = {C1 - C2:+.3e}"
              f"   (equal amplitudes give 2.2e-2)")

    if verbose:
        print("  Propagating L1 unstable manifold (Moon-bound '+' branch) ...")
    u_p = compute_manifold(s1, T1, MU_EM, 0., 0., 0.,
                           'unstable', '+', n_strands, t_max=t_max)

    if verbose:
        print("  Propagating L2 stable manifold (both branches) ...")
    s_p = compute_manifold(s2, T2, MU_EM, 0., 0., 0.,
                           'stable', '+', n_strands, t_max=t_max)
    s_m = compute_manifold(s2, T2, MU_EM, 0., 0., 0.,
                           'stable', '-', n_strands, t_max=t_max)

    # ── Poincare section at the Moon's x-position ────────────────────────────
    moon_x = 1.0 - MU_EM
    if verbose:
        print(f"  Finding crossings at x = {moon_x:.5f} (Moon) ...")

    cross_u  = poincare_section(u_p, 'x', moon_x, direction=0)
    cross_sp = poincare_section(s_p, 'x', moon_x, direction=0)
    cross_sm = poincare_section(s_m, 'x', moon_x, direction=0)
    cross_s  = cross_sp + cross_sm

    # Filter: discard strands that leave the Earth-Moon system
    cross_u = [c for c in cross_u if abs(c[1]) < 0.3 and abs(c[2]) < 0.2]
    cross_s = [c for c in cross_s if abs(c[1]) < 0.3 and abs(c[2]) < 0.2]

    # Self-intersection guard: sample both originating halos densely and refuse
    # any crossing that is still within min_sep of one of them.
    orb1 = _full_orbit(s1, T1, n=400)
    orb2 = _full_orbit(s2, T2, n=400)
    exclude = ([orb1[:, k] for k in range(orb1.shape[1])]
               + [orb2[:, k] for k in range(orb2.shape[1])])

    if verbose:
        print(f"    L1 unstable: {len(cross_u)} crossings, "
              f"L2 stable: {len(cross_s)} crossings")
        print(f"    self-intersection guard: min_sep = {min_sep:.3e} nd "
              f"= {min_sep * EM_KM:,.0f} km from either halo")

    return dict(L1=L1, L2=L2, s1=s1, T1=T1, s2=s2, T2=T2,
                Az_L1=float(Az_L1), Az_L2=float(Az_L2),
                C_L1=float(C1), C_L2=float(C2), dC=float(C1 - C2),
                u_p=u_p, s_p=s_p, s_m=s_m,
                cross_u=cross_u, cross_s=cross_s, moon_x=moon_x,
                exclude=exclude, min_sep=float(min_sep),
                orb1=orb1, orb2=orb2)


# ── figure 6 — Poincaré map ────────────────────────────────────────────────────

def fig_poincare_map(
    output: str    = 'fig6_poincare_map.png',
    Az_L1: float   = _AZ_L1_DEFAULT,
    Az_L2: float   = _AZ_L2_DEFAULT,
    pair: dict     = None,
    n_strands: int = 60,
    t_max: float   = 5.0 * np.pi,
    cache: dict    = None,
) -> dict:
    """
    Poincaré portrait (y, ẏ) at the Moon's x-position, showing the L1
    unstable and L2 stable manifold families of an ENERGY-MATCHED pair.

    Both branches lie on the same energy surface C, which is the necessary
    condition for the tubes to intersect at all.  Whether they actually overlap
    in (y, ẏ) is what this figure shows.

    Returns the data dict so fig_manifold_transfer() can reuse it.
    """
    data = cache or _compute_manifolds(Az_L1=Az_L1, Az_L2=Az_L2, pair=pair,
                                       n_strands=n_strands,
                                       t_max=t_max, verbose=True)
    cross_u = data['cross_u']
    cross_s = data['cross_s']
    L1 = data['L1'];  L2 = data['L2']

    # ── plot (y, ẏ) at the Moon section ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor(_LIGHT)
    ax.set_facecolor(_LIGHT)

    if cross_u:
        yu  = [c[1] for c in cross_u]
        dyu = [c[4] for c in cross_u]
        ax.scatter(yu, dyu, marker='o', s=26, facecolors='none',
                   edgecolors=_ORANGE_L, linewidths=1.0, zorder=5,
                   label='L₁ unstable manifold')

    if cross_s:
        ys  = [c[1] for c in cross_s]
        dys = [c[4] for c in cross_s]
        ax.scatter(ys, dys, marker='s', s=22, facecolors='none',
                   edgecolors=_TEAL_L, linewidths=1.0, zorder=5,
                   label='L₂ stable manifold')

    # Overlap region (intersection → heteroclinic candidates)
    ax.axvline(0, color=_GREY_FAINT, lw=0.6, ls=':', zorder=0)
    ax.axhline(0, color=_GREY_FAINT, lw=0.6, ls=':', zorder=0)

    ax.set_xlabel('y  [Earth–Moon non-dim]', color=_INK, fontsize=12)
    ax.set_ylabel('ẏ  [non-dim]',             color=_INK, fontsize=12)
    ax.set_title('Poincaré Section at the Moon  (x = 1 − μ)\n'
                 'Earth–Moon  L₁ Unstable ∩ L₂ Stable Manifolds, '
                 'energy-matched pair',
                 color=_INK, fontsize=13, pad=10)

    ax.tick_params(colors=_INK, direction='in', top=True, right=True)
    for sp in ax.spines.values():
        sp.set_color(_INK)
        sp.set_linewidth(0.8)
    ax.legend(facecolor=_LIGHT, edgecolor=_GREY_L, labelcolor=_INK,
              framealpha=1.0, fontsize=11, loc='upper left')

    # Robust limits: a handful of strands cross the section at large |ydot| and
    # would otherwise stretch the axis by 4x, flattening the tube structure that
    # is the point of the figure.  Clip to the 1-99 percentile, padded.
    _yd = [c[4] for c in cross_u] + [c[4] for c in cross_s]
    if _yd:
        _q1, _q99 = np.percentile(_yd, [1, 99])
        _pad = 0.18 * (_q99 - _q1)
        ax.set_ylim(_q1 - _pad, _q99 + _pad)
        _n_out = sum(1 for v in _yd if not (_q1 - _pad <= v <= _q99 + _pad))
        if _n_out:
            ax.text(0.015, 0.015, f'{_n_out} of {len(_yd)} crossings outside '
                                  f'the plotted $\\dot{{y}}$ range',
                    transform=ax.transAxes, ha='left', va='bottom',
                    color=_GREY_FAINT, fontsize=7)

    # Annotate equilibrium x positions for context
    moon_x = data['moon_x']
    ax.text(0.98, 0.97,
            f'Section: x = {moon_x:.5f}  (Moon)\n'
            f'L₁ @ x = {L1[0]:.4f},  $A_z$ = {data["Az_L1"]:.5f}\n'
            f'L₂ @ x = {L2[0]:.4f},  $A_z$ = {data["Az_L2"]:.5f}\n'
            f'matched  C = {data["C_L1"]:.7f},  '
            f'$\\Delta C$ = {data["dC"]:+.1e}',
            transform=ax.transAxes, ha='right', va='top',
            color=_GREY_L, fontsize=8, zorder=12,
            bbox=dict(boxstyle='round,pad=0.4', facecolor=_LIGHT,
                      edgecolor='#dddddd', alpha=0.94))

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches='tight', facecolor=_LIGHT)
    plt.close(fig)
    print(f"  ✓ Saved → {output}")
    return data


# ── figure 7 — heteroclinic transfer ──────────────────────────────────────────

def fig_manifold_transfer(
    output: str    = 'fig7_manifold_transfer.png',
    Az_L1: float   = _AZ_L1_DEFAULT,
    Az_L2: float   = _AZ_L2_DEFAULT,
    pair: dict     = None,
    n_strands: int = 60,
    t_max: float   = 5.0 * np.pi,
    cache: dict    = None,
) -> dict:
    """
    Plot the best manifold-guided L1 -> L2 transfer in the Earth-Moon rotating
    frame (x-y projection), with the manifold tubes as context.

    Reported as a TWO-IMPULSE TRANSFER, not a heteroclinic connection.  A true
    heteroclinic connection costs zero delta-V by construction; the nonzero
    delta-V here measures the residual gap between W^u(L1) and W^s(L2) on the
    Moon section, and the position residual states how far from an actual
    intersection the best pair is.  Both numbers are printed and annotated.

    Returns a dict of the transfer quantities.
    """
    data = cache or _compute_manifolds(Az_L1=Az_L1, Az_L2=Az_L2, pair=pair,
                                       n_strands=n_strands,
                                       t_max=t_max, verbose=True)

    L1 = data['L1'];  L2 = data['L2']
    s1 = data['s1'];  T1 = data['T1']
    s2 = data['s2'];  T2 = data['T2']
    u_p = data['u_p']
    s_p = data['s_p'];  s_m = data['s_m']
    cross_u = data['cross_u']
    cross_s = data['cross_s']

    if not cross_u or not cross_s:
        print("  WARNING: no crossings found — cannot plot transfer.")
        return

    # ── best manifold-guided transfer, with the self-intersection guard ──────
    # High w_pos: minimise position residual first, then velocity.
    # exclude_states/min_sep refuse any crossing still sitting on its
    # originating halo, so an undeparted strand cannot be selected.
    (i_u, j_s), state_u, state_s, dv_vec = match_manifolds(
        cross_u, cross_s, w_pos=1e6,
        exclude_states=data['exclude'], min_sep=data['min_sep'])
    dv_mag, _, pos_res = transfer_dv(state_u, state_s)
    dv_ms = dv_mag * EM_VEL * 1000          # m/s
    dr_km = pos_res * EM_KM                 # km
    print(f"  Best manifold-guided pair (self-intersection guard active):")
    print(f"    |ΔV|              = {dv_ms:,.1f} m/s")
    print(f"    position residual = {dr_km:,.0f} km")
    print(f"    -> TWO-IMPULSE TRANSFER, not a heteroclinic connection: a true")
    print(f"       connection would show |ΔV| = 0 with a vanishing residual.")

    # ── full halo orbit trajectories ──────────────────────────────────────────
    orb1 = data['orb1']
    orb2 = data['orb2']

    # ── plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 9))
    fig.patch.set_facecolor(_DARK)
    ax.set_facecolor(_DARK)

    # Manifold tubes — faint.  Alpha scales with strand count: at the 360
    # strands the convergence study calls for, a fixed 0.13 saturates the canvas
    # and buries the halos, the Moon and the caption.
    _n = max(len(u_p), len(s_p) + len(s_m), 1)
    _a = float(np.clip(8.0 / _n, 0.02, 0.15))
    for traj in u_p:
        ax.plot(traj[0], traj[1], color=_ORANGE, alpha=_a, lw=0.5)
    for traj in s_p:
        ax.plot(traj[0], traj[1], color=_TEAL,   alpha=_a, lw=0.5)
    for traj in s_m:
        ax.plot(traj[0], traj[1], color=_TEAL,   alpha=_a, lw=0.5)

    # Halo orbits — bright
    ax.plot(orb1[0], orb1[1], color=_ORANGE, lw=2.2, zorder=7,
            label=f'L₁ halo,  $A_z$ = {data["Az_L1"]:.5f}')
    ax.plot(orb2[0], orb2[1], color=_TEAL,   lw=2.2, zorder=7,
            label=f'L₂ halo,  $A_z$ = {data["Az_L2"]:.5f}')

    # Moon's orbit (dotted circle, radius = 1−μ in dimensionless coords,
    # but in the rotating frame the Moon is a fixed point)
    # Just show its position:
    moon_x = 1.0 - MU_EM
    ax.axvline(moon_x, color='#888', lw=0.6, ls=':', alpha=0.5)

    # Primaries
    ax.scatter([-MU_EM],    [0], s=140, color='#4488ff', zorder=10, label='Earth')
    ax.scatter([moon_x],    [0], s=70,  color='#cccccc', zorder=10, label='Moon')
    ax.text(-MU_EM, -0.055, 'Earth', color='#88aaff', fontsize=9,
            ha='center', va='top')
    ax.text(moon_x, -0.055, 'Moon',  color='#cccccc', fontsize=9,
            ha='center', va='top')

    # Equilibrium points
    ax.scatter([L1[0], L2[0]], [0, 0], s=40, color=_WHITE,
               marker='x', zorder=9, linewidths=1.5)
    for _x, _lab in ((L1[0], 'L₁'), (L2[0], 'L₂')):
        ax.text(_x, 0.055, _lab, color=_WHITE, fontsize=9.5, ha='center',
                zorder=14,
                bbox=dict(boxstyle='round,pad=0.18', facecolor=_DARK,
                          edgecolor='none', alpha=0.75))

    # ΔV connection marker
    ax.scatter([state_u[0]], [state_u[1]], s=200, color=_YELLOW,
               marker='*', zorder=12, linewidths=1,
               label=f'Two-impulse patch point,  ΔV = {dv_ms:,.0f} m/s')
    ax.annotate(f'ΔV = {dv_ms:,.0f} m/s\n{dr_km:,.0f} km residual',
                xy=(state_u[0], state_u[1]),
                xytext=(state_u[0] + 0.10, state_u[1] + 0.20),
                color=_YELLOW, fontsize=9, ha='left', arrowprops=dict(
                    arrowstyle='->', color=_YELLOW, lw=1.2))
    # State the framing on the figure itself, so it cannot be misread.
    ax.text(0.015, 0.015,
            f'Energy-matched pair:  C = {data["C_L1"]:.7f},  '
            f'$\\Delta C$ = {data["dC"]:+.1e}\n'
            f'Manifold-guided TWO-IMPULSE transfer — not a heteroclinic\n'
            f'connection, which would cost ΔV = 0 by construction.\n'
            f'Self-intersection guard: min. {data["min_sep"]*EM_KM:,.0f} km '
            f'from either halo.',
            transform=ax.transAxes, ha='left', va='bottom',
            color='#c8c8c8', fontsize=7.8, linespacing=1.5, zorder=15,
            bbox=dict(boxstyle='round,pad=0.45', facecolor=_DARK,
                      edgecolor='#333844', alpha=0.92))

    ax.set_xlabel('x  [Earth–Moon non-dim]', color=_WHITE, fontsize=12)
    ax.set_ylabel('y  [Earth–Moon non-dim]', color=_WHITE, fontsize=12)
    ax.set_title('Heteroclinic Transfer Trajectory  L₁ → L₂\n'
                 'Earth–Moon Rotating Frame',
                 color=_WHITE, fontsize=13, pad=10)
    # Focus on the Earth-Moon neighbourhood.  At these amplitudes many L2
    # stable strands escape to |x| ~ 5, which without a window compresses the
    # halos, the Moon and the patch point into an unreadable blob at the centre.
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-0.28, 1.38)
    ax.set_ylim(-0.52, 0.52)
    ax.tick_params(colors=_WHITE)
    for sp in ax.spines.values():
        sp.set_color(_GREY)
    ax.legend(facecolor='#1a1a2e', labelcolor=_WHITE,
              framealpha=0.85, fontsize=10, loc='upper left')

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches='tight', facecolor=_DARK)
    plt.close(fig)
    print(f"  ✓ Saved → {output}")
    return dict(dv_nd=float(dv_mag), dv_ms=float(dv_ms),
                pos_residual_nd=float(pos_res), pos_residual_km=float(dr_km),
                state_u=state_u, state_s=state_s, idx=(i_u, j_s),
                C=data['C_L1'], dC=data['dC'],
                Az_L1=data['Az_L1'], Az_L2=data['Az_L2'])


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    _args = [a for a in sys.argv[1:] if not a.startswith('#')]
    which = _args[0] if _args else 'both'

    # Best sampling reached in the convergence study above.  The ΔV it
    # reports is an upper bound at this grid, not a converged value.
    n     = 360
    t_max = 8.0 * np.pi

    print("\n── Energy-matched halo pair by pseudo-arclength continuation ──")
    _pair = matched_pair(verbose=True)

    if which in ('poincare', 'fig6', 'both'):
        print("\n── Figure 6: Poincaré map ───────────────────────────")
        cache = fig_poincare_map('fig6_poincare_map.png',
                                  pair=_pair, n_strands=n, t_max=t_max)
    else:
        cache = None

    if which in ('transfer', 'fig7', 'both'):
        print("\n── Figure 7: manifold-guided L₁ → L₂ transfer ───────")
        fig_manifold_transfer('fig7_manifold_transfer.png',
                               pair=_pair, n_strands=n, t_max=t_max,
                               cache=cache)
