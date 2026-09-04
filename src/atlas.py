"""
atlas.py — halo family atlas over the technologically relevant sail band,
           beta in [0.001, 0.05], Sun-Earth.

Why this band
─────────────
Flown solar sails sit at beta of 6e-4 to 6e-3 (sail_technology.py, reduced from
primary specifications); designed but unflown ones reach 2e-2.  This band spans
both and extends a little beyond, to 5e-2.  Below it the
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
          chain_beta: bool = True,
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

    Branch continuity ACROSS beta (`chain_beta`)
    ────────────────────────────────────────────
    Pseudo-arclength keeps a single family on its branch as Az varies, but an
    earlier version called find_halo_seed independently at every beta, and that
    function returns the FIRST amplitude passing require_halo.  Nothing then tied
    consecutive beta to the same branch, and at beta = 0.040 the scan landed on a
    different one: x0 sat Earthward of x_eq instead of sunward (and crossed
    x = 1), the z-asymmetry delta was negative where every neighbour was
    positive, and the branch died after 43 members against 69 either side.  One
    of twelve atlas curves was a different family.

    With chain_beta the first beta is bootstrapped by find_halo_seed and every
    later beta is seeded from its predecessor's converged state, with z0 rescaled
    by gamma_new/gamma_old so Az/gamma is preserved across the step.  The atlas is
    then one connected sheet rather than twelve independent walks.  Two guards
    verify it afterwards: sign(delta) must match the reference beta, and the
    family must sit sunward of the equilibrium (x0 < x_eq).  A beta that fails
    either is re-seeded from scratch and, if it still fails, recorded in
    `suspect` rather than silently shipped.
    """
    if betas is None:
        betas = DEFAULT_BETAS

    families, failed, suspect = {}, [], []
    prev = None          # (state0, T_half, gamma) of the last accepted beta
    ref_delta_sign = 0   # branch reference, set by the first accepted beta

    def _walk(eq, gamma, b, seed):
        return continue_branch(eq, mu, param='Az',
                               seed_state=seed, other=float(b),
                               ds=ds_over_gamma * gamma,
                               ds_min=1e-4 * gamma,
                               ds_max=0.2 * gamma,
                               n_steps=n_steps,
                               lam_bounds=(1e-3 * gamma,
                                           az_max_over_gamma * gamma),
                               max_dx_over_gamma=max_dx_over_gamma,
                               with_stability=True, verbose=False)

    def _bootstrap(eq, b, gamma):
        s0, Th0, Az0 = find_halo_seed(eq, mu, beta=float(b),
                                      max_dx_over_gamma=max_dx_over_gamma,
                                      verbose=False)
        return (s0, Th0), float(Az0)

    def _branch_ok(br, eq):
        """(ok, reason).  Checks branch identity against the reference."""
        med = float(np.median(br['delta']))
        if ref_delta_sign and np.sign(med) != ref_delta_sign:
            return False, f'delta sign {np.sign(med):+.0f} vs ref {ref_delta_sign:+.0f}'
        if np.any(br['x0'] >= eq[0]):
            return False, 'x0 reaches or exceeds x_eq (not sunward)'
        return True, ''

    for b in np.asarray(betas, dtype=float):
        eq = [equilibrium(float(b), mu), 0.0, 0.0]
        gamma = local_scale(eq, mu)
        if verbose:
            print(f"  beta = {b:.4f}   x_eq = {eq[0]:.8f}   "
                  f"gamma = {gamma:.6f}")

        # -- choose a seed: chained from the previous beta, else bootstrap ----
        seed, Az0, how = None, float('nan'), ''
        if chain_beta and prev is not None:
            s_prev, Th_prev, g_prev = prev
            s_try = np.array(s_prev, dtype=float).copy()
            s_try[2] *= gamma / g_prev          # preserve Az/gamma
            seed, Az0, how = (s_try, Th_prev), float(abs(s_try[2])), 'chained'
        if seed is None:
            try:
                seed, Az0 = _bootstrap(eq, b, gamma)
                how = 'bootstrap'
            except RuntimeError as e:
                if verbose:
                    print(f"      no halo seed: {str(e)[:70]}")
                failed.append((float(b), 'no seed'))
                continue

        # -- walk, with one re-seed if the chained seed fails or drifts -------
        br = None
        for attempt in range(2):
            try:
                cand = _walk(eq, gamma, b, seed)
            except RuntimeError as e:
                cand, why = None, f'continuation failed: {str(e)[:50]}'
            else:
                ok, why = _branch_ok(cand, eq)
                if ok:
                    br = cand
                    break
                cand = None
            if attempt == 0 and how == 'chained':
                if verbose:
                    print(f"      chained seed rejected ({why}); re-seeding")
                try:
                    seed, Az0 = _bootstrap(eq, b, gamma)
                    how = 'bootstrap(after chain)'
                except RuntimeError:
                    break
            else:
                if verbose:
                    print(f"      branch guard: {why}")
                suspect.append((float(b), why))
                br = cand
                break

        if br is None:
            failed.append((float(b), 'continuation/guard'))
            continue

        if not ref_delta_sign:
            ref_delta_sign = int(np.sign(np.median(br['delta'])))
        br['seeded_by'] = how
        prev = (br['state0'][0], br['T'][0] / 2.0, gamma)

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
            print(f"      stop: {br['stopped']}   seed: {br['seeded_by']}")

    return dict(families=families, failed=failed, suspect=suspect, mu=mu,
                beta_tidal=critical_beta_tidal(mu),
                max_dx_over_gamma=max_dx_over_gamma,
                chain_beta=chain_beta,
                ref_delta_sign=ref_delta_sign)


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


def load_csv(path: str = 'halo_atlas.csv', mu: float = MU_SE) -> dict:
    """
    Reconstruct an atlas dict from halo_atlas.csv.

    Why this exists.  fig_atlas() used to call build() whenever it was not
    handed an atlas, so plotting the figure silently re-walked every family --
    tens of minutes, with no output, indistinguishable from a hang.  Plotting
    should never recompute: build() writes the CSV, and everything the figure
    needs is in it.  The only field not stored per row was the fold locations,
    which is why export_csv now writes an `is_fold` flag.

    A CSV written before that column existed loads fine; its families simply
    carry no fold markers.
    """
    import csv as _csv
    import os

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found -- run `python main.py atlas` first "
            f"(it walks the families and writes the CSV)")

    rows = list(_csv.DictReader(open(path)))
    if not rows:
        raise RuntimeError(f"{path} is empty")

    families = {}
    for r in rows:
        b = float(r['beta'])
        f = families.setdefault(b, {k: [] for k in
                                    ('Az', 'C', 'T', 'x0', 'vy0', 'nu_max',
                                     'lambda_max', 'delta', 'stable',
                                     'folds')})
        f['Az'].append(float(r['Az']))
        f['C'].append(float(r.get('C_sail', r.get('C', 'nan'))))
        f['T'].append(float(r['T_nd']))
        f['x0'].append(float(r['x0']))
        f['vy0'].append(float(r['vy0']))
        f['nu_max'].append(float(r['nu_max']))
        f['lambda_max'].append(float(r['lambda_max']))
        f['delta'].append(float(r['delta']))
        f['stable'].append(bool(int(r['stable'])))
        if int(r.get('is_fold', 0)):
            f['folds'].append(float(r['Az']))
        f['gamma'] = float(r['gamma'])
        f['x_eq'] = float(r['x_eq'])
        f['seeded_by'] = r.get('seeded_by', '')

    for b, f in families.items():
        for k in ('Az', 'C', 'T', 'x0', 'vy0', 'nu_max', 'lambda_max',
                  'delta'):
            f[k] = np.asarray(f[k], dtype=float)
        f['stable'] = np.asarray(f['stable'], dtype=bool)
        f['max_dx_over_gamma'] = float(
            np.max(np.abs(f['x0'] - f['x_eq'])) / f['gamma'])
        f['n_far_field'] = 0
        f['stopped'] = 'loaded from CSV'

    return dict(families=families, failed=[], suspect=[], mu=mu,
                beta_tidal=critical_beta_tidal(mu),
                max_dx_over_gamma=MAX_DX_OVER_GAMMA,
                chain_beta=None, ref_delta_sign=0, source=path)


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
        # Never rebuild implicitly: plotting must be cheap.  Load the cached
        # CSV, and say plainly what to run if it is absent.
        atlas = load_csv()
        if verbose:
            print(f"  loaded {len(atlas['families'])} families from "
                  f"{atlas['source']}")
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

    fig.suptitle(r'Sun--Earth halo families across the flown and near-term '
                 r'sail band, $\beta \in [0.001,\,0.05]$,  $\alpha = 0$',
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
                    'nu_max', 'lambda_max', 'delta', 'stable', 'is_fold',
                    'seeded_by'])
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
                            int(br['stable'][i]),
                            int(any(abs(br['Az'][i] - f) <= 1e-12
                                    for f in br.get('folds', ()))),
                            br.get('seeded_by', '')])
    return path


if __name__ == '__main__':
    print("\n== Halo family atlas, Sun-Earth, beta in [0.001, 0.05] ==")
    a = build()
    summarise(a)
    fig_atlas(atlas=a)
    p = export_csv(a)
    n = sum(len(v['Az']) for v in a['families'].values())
    print(f"  Exported {n} family members -> {p}")
