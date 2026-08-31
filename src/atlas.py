"""
atlas.py — halo family atlas over the technologically relevant sail band,
           beta in [0.001, 0.05], Sun-Earth.

Why this band
─────────────
Flown and near-term solar sails sit at beta of order 1e-3 to 5e-2.  Below it the
sail is irrelevant; above it, critical_beta.py shows the collinear structure has
already dissolved (tidal parity at beta = 0.0286, and by beta = 0.5 the
equilibrium is 20.6 Hill radii out and the dynamics are Keplerian).  So this band
is where a sail both matters and still lives in a genuine three-body structure —
and it is exactly the band the earlier version of this project skipped, jumping
from beta = 0 straight to beta = 0.5.

What the atlas contains
───────────────────────
For each beta, the halo family is walked in Az by pseudo-arclength continuation
(continuation.py), recording at every family member:

    Az        out-of-plane amplitude (the crossing value z0)
    C         Jacobi constant, with the sail folded into the potential as
              (1-beta)(1-mu)/r1  — valid because a face-on sail is conservative
    T         full period
    nu_max    max |(lam + 1/lam)/2| over the non-trivial Floquet pairs;
              <= 1 is linear stability
    delta     z-extrema asymmetry, the branch discriminator

Caveats stated up front
───────────────────────
  * alpha = 0 throughout.  A steered sail is NOT conservative, so C is not an
    integral and none of the energy bookkeeping here carries over.  That case is
    open and is the natural sequel.
  * The Jacobi constant reported uses the face-on sail potential, so it is
    comparable ACROSS Az at fixed beta, and across beta only in the sense that
    the potential is the beta-dependent one.  Do not compare raw C at different
    beta as if it were the same function.
  * Families are seeded by scanning for a validated halo (find_halo_seed) because
    the Richardson guess does not land on the halo branch everywhere.
"""

from __future__ import annotations

import numpy as np

from src.jacobi import jacobi_constant_sail
from src.continuation import (MAX_DX_OVER_GAMMA, continue_branch,
                             find_halo_seed, local_scale)
from src.critical_beta import equilibrium, critical_beta_tidal, MU_SE

AU_KM = 1.495978707e8
DAYS_PER_ND = 365.25 / (2.0 * np.pi)

DEFAULT_BETAS = np.array([0.001, 0.002, 0.003, 0.005, 0.007, 0.010,
                          0.014, 0.019, 0.025, 0.032, 0.040, 0.050])


def build(betas=None, mu: float = MU_SE,
          ds_over_gamma: float = 0.02, n_steps: int = 70,
          az_max_over_gamma: float = 1.0,
          max_dx_over_gamma: float = MAX_DX_OVER_GAMMA,
          verbose: bool = True) -> dict:
    """
    Walk the halo family at each beta.  Returns {beta: branch_dict} plus meta.

    Every length is expressed in units of the local scale
    gamma = |(1-mu) - x_eq|, which grows by a factor of two across this beta
    band (0.0101 at beta=0.001 to 0.0196 at beta=0.050).  Absolute step sizes
    and amplitude bounds therefore mean different things at each beta, and fixed
    ones are what let far-field orbits into the previous atlas:

      * `ds_over_gamma` — arclength step.  The former absolute ds = 1.5e-3 is
        15 % of gamma for Sun-Earth, so a single step overshot the entire seed
        amplitude (0.05 gamma = 5e-4) threefold.
      * `az_max_over_gamma` — outer bound on Az.  The former absolute 0.05 is
        5 gamma, far outside the local family.
      * `max_dx_over_gamma` — the hard far-field guard in continue_branch,
        which is normally what terminates each branch.
    """
    if betas is None:
        betas = DEFAULT_BETAS

    families, failed = {}, []
    for b in np.asarray(betas, dtype=float):
        eq = [equilibrium(float(b), mu), 0.0, 0.0]
        gamma = local_scale(eq, mu)
        if verbose:
            print(f"  beta = {b:.4f}   x_eq = {eq[0]:.8f}   "
                  f"gamma = {gamma:.6f}")
        try:
            s0, Th0, Az0 = find_halo_seed(eq, mu, beta=float(b),
                                          max_dx_over_gamma=max_dx_over_gamma,
                                          verbose=verbose)
        except RuntimeError as e:
            if verbose:
                print(f"      no halo seed: {str(e)[:70]}")
            failed.append((float(b), 'no seed'))
            continue
        try:
            br = continue_branch(eq, mu, param='Az',
                                 seed_state=(s0, Th0), other=float(b),
                                 ds=ds_over_gamma * gamma,
                                 ds_min=1e-4 * gamma,
                                 ds_max=0.2 * gamma,
                                 n_steps=n_steps,
                                 lam_bounds=(1e-3 * gamma,
                                             az_max_over_gamma * gamma),
                                 max_dx_over_gamma=max_dx_over_gamma,
                                 with_stability=True, verbose=False)
        except RuntimeError as e:
            if verbose:
                print(f"      continuation failed: {str(e)[:70]}")
            failed.append((float(b), 'continuation'))
            continue

        br['Az_seed'] = float(Az0)
        families[float(b)] = br
        if verbose:
            ns = int(br['stable'].sum())
            print(f"      {len(br['Az'])} members | "
                  f"Az/gamma [{br['Az'].min()/gamma:.3f}, "
                  f"{br['Az'].max()/gamma:.3f}] | "
                  f"max|x0-x_eq| = {br['max_dx_over_gamma']:.2f} gamma | "
                  f"nu_max [{np.nanmin(br['nu_max']):.2f}, "
                  f"{np.nanmax(br['nu_max']):.2f}] | "
                  f"{len(br['folds'])} folds | {ns} stable")
            print(f"      stop: {br['stopped']}")

    return dict(families=families, failed=failed, mu=mu,
                beta_tidal=critical_beta_tidal(mu),
                max_dx_over_gamma=max_dx_over_gamma)


def summarise(atlas: dict) -> None:
    """Print a compact table of the atlas."""
    print()
    print("   beta     x_eq        gamma   members  Az/gamma range   "
          "T range [d]     nu_max min  max|dx|/g  folds  stable")
    print("  " + "-" * 116)
    for b in sorted(atlas['families']):
        br = atlas['families'][b]
        g = br['gamma']
        print(f"  {b:6.4f}  {br['x_eq']:.7f}  {g:.6f}  {len(br['Az']):7d}  "
              f"[{br['Az'].min()/g:.3f},{br['Az'].max()/g:.3f}]  "
              f"[{br['T'].min()*DAYS_PER_ND:6.1f},{br['T'].max()*DAYS_PER_ND:6.1f}]  "
              f"{np.nanmin(br['nu_max']):10.3f}  "
              f"{br['max_dx_over_gamma']:9.3f}  {len(br['folds']):5d}  "
              f"{int(br['stable'].sum()):6d}")
    print(f"\n  Far-field guard: |x0 - x_eq| <= "
          f"{atlas['max_dx_over_gamma']:g} gamma, enforced before any member is "
          f"recorded.")
    n_far = sum(v['n_far_field'] for v in atlas['families'].values())
    print(f"  Branches terminated by the guard: "
          f"{sum(1 for v in atlas['families'].values() if v['n_far_field'])}"
          f"/{len(atlas['families'])}  ({n_far} states rejected).")
    if atlas['failed']:
        print(f"\n  failed: {atlas['failed']}")


def fig_atlas(output: str = 'fig10_halo_atlas.png',
              atlas: dict = None,
              verbose: bool = True) -> dict:
    """
    Four-panel plain-paper atlas.

    (a) family curves C(Az) at each beta
    (b) period T(Az) at each beta
    (c) stability index nu_max(Az), with the |nu| = 1 stability boundary
    (d) the (beta, Az) plane: where families exist, folds, stable segments
    """
    from src.paperstyle import use, panel_label, thin_guide
    use()
    import matplotlib.pyplot as plt

    if atlas is None:
        atlas = build(verbose=verbose)
    fams = atlas['families']
    if not fams:
        raise RuntimeError("atlas is empty")

    betas = sorted(fams)
    # greyscale ramp: low beta light, high beta dark (monotone, print-safe)
    greys = np.linspace(0.72, 0.0, len(betas))

    fig, axes = plt.subplots(2, 2, figsize=(7.6, 6.0))

    # (a) family depth below the local equilibrium, in the SAIL potential.
    #     Raw C_sail is not comparable across beta because the potential itself
    #     is beta-dependent; the difference from the equilibrium value is.
    ax = axes[0, 0]
    for g, b in zip(greys, betas):
        br = fams[b]
        C_eq = jacobi_constant_sail(
            np.array([br['x_eq'], 0.0, 0.0, 0.0, 0.0, 0.0]), MU_SE, b)
        ax.plot(br['Az'] / br['gamma'], br['C'] - C_eq, '-',
                color=str(g), lw=0.9)
    ax.set_xlabel(r'$A_z / \gamma$')
    ax.set_ylabel(r'$C_{\rm sail} - C_{\rm sail}(x_{\rm eq})$')
    panel_label(ax, '(a)')

    # (b) T(Az) in days
    ax = axes[0, 1]
    for g, b in zip(greys, betas):
        br = fams[b]
        ax.plot(br['Az'], br['T'] * DAYS_PER_ND, '-', color=str(g), lw=0.9)
    ax.set_xlabel(r'$A_z$  [nd]')
    ax.set_ylabel('period  [days]')
    panel_label(ax, '(b)')

    # (c) stability index
    ax = axes[1, 0]
    for g, b in zip(greys, betas):
        br = fams[b]
        ax.semilogy(br['Az'], np.maximum(br['nu_max'], 1e-2), '-',
                    color=str(g), lw=0.9)
    thin_guide(ax, y=1.0, label=r'$|\nu|=1$ (stability boundary)')
    ax.set_xlabel(r'$A_z$  [nd]')
    ax.set_ylabel(r'$\nu_{\max}$')
    panel_label(ax, '(c)')

    # (d) the (beta, Az) plane
    ax = axes[1, 1]
    for b in betas:
        br = fams[b]
        ax.plot(np.full_like(br['Az'], b), br['Az'], '-',
                color='0.62', lw=0.7)
        st = br['stable']
        if st.any():
            ax.plot(np.full(st.sum(), b), br['Az'][st], 'ko', ms=2.4)
        for f in br['folds']:
            ax.plot([b], [f], 'k^', ms=4.0, mfc='white', mew=0.8)
    ax.set_xscale('log')
    thin_guide(ax, x=atlas['beta_tidal'])
    ax.annotate(rf"tidal parity $\beta={atlas['beta_tidal']:.4f}$",
                xy=(atlas['beta_tidal'] * 0.92, ax.get_ylim()[1] * 0.55),
                rotation=90, fontsize=7.0, ha='right', va='center')
    ax.plot([], [], '-', color='0.62', lw=0.7, label='family')
    ax.plot([], [], 'ko', ms=2.4, label=r'linearly stable')
    ax.plot([], [], 'k^', ms=4.0, mfc='white', mew=0.8, label='fold in $A_z$')
    ax.legend(loc='upper left', fontsize=7.0)
    ax.set_xlabel(r'$\beta$')
    ax.set_ylabel(r'$A_z$  [nd]')
    panel_label(ax, '(d)')

    fig.suptitle(r'Sun--Earth halo families across the flown sail band, '
                 r'$\beta \in [0.001,\,0.05]$,  $\alpha = 0$',
                 fontsize=9.5, y=0.999)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(output)
    plt.close(fig)
    if verbose:
        print(f"\n  Saved -> {output}")

    return atlas


def export_csv(atlas: dict, path: str = 'halo_atlas.csv') -> str:
    """Write the atlas as a flat CSV so others can use it without running this."""
    import csv
    with open(path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['beta', 'gamma', 'Az', 'Az_over_gamma',
                    'C_sail', 'C_sail_eq', 'dC_from_eq',
                    'T_nd', 'T_days', 'x0', 'x_eq', 'dx_over_gamma', 'vy0',
                    'nu_max', 'lambda_max', 'delta', 'stable'])
        for b in sorted(atlas['families']):
            br = atlas['families'][b]
            g = br['gamma']
            C_eq = jacobi_constant_sail(
                np.array([br['x_eq'], 0.0, 0.0, 0.0, 0.0, 0.0]), MU_SE, b)
            for i in range(len(br['Az'])):
                w.writerow([f"{b:.6f}", f"{g:.10f}", f"{br['Az'][i]:.10f}",
                            f"{br['Az'][i]/g:.8f}",
                            f"{br['C'][i]:.10f}", f"{C_eq:.10f}",
                            f"{br['C'][i]-C_eq:.10e}", f"{br['T'][i]:.10f}",
                            f"{br['T'][i]*DAYS_PER_ND:.6f}",
                            f"{br['x0'][i]:.12f}", f"{br['x_eq']:.12f}",
                            f"{abs(br['x0'][i]-br['x_eq'])/g:.8f}",
                            f"{br['vy0'][i]:.12f}",
                            f"{br['nu_max'][i]:.6f}",
                            f"{br['lambda_max'][i]:.6f}",
                            f"{br['delta'][i]:.8f}",
                            int(br['stable'][i])])
    return path


if __name__ == '__main__':
    print("\n== Halo family atlas, Sun-Earth, beta in [0.001, 0.05] ==")
    a = build()
    summarise(a)
    fig_atlas(atlas=a)
    p = export_csv(a)
    n = sum(len(v['Az']) for v in a['families'].values())
    print(f"  Exported {n} family members -> {p}")
