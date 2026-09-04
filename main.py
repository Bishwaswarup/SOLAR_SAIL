#!/usr/bin/env python3
"""
main.py — single entry point for the Sun-Earth solar-sail study.

    python main.py                # paper set: verify + results + figures
    python main.py verify         # regression suite + closed-form assertions
    python main.py results        # recompute every headline number -> results.txt
    python main.py figures        # regenerate the paper figures
    python main.py atlas          # re-walk the halo families (SLOW, ~minutes)
    python main.py earthmoon      # Earth-Moon material: NOT part of the paper
    python main.py all            # everything above except earthmoon
    python main.py list           # show the stages and exit

What this replaces
──────────────────
The previous main.py was a beta = 0.5 demonstration pipeline whose "L1 -> L2
transfer" built BOTH manifolds from the same orbit and therefore matched that
orbit against itself (bug A5): the two "matched" states were the same point,
separated by 0.000 km, each sitting the seed perturbation away from the halo.
There was no second orbit and no transfer.  That demonstration is deleted rather
than shipped with a caveat.  beta = 0.5 is in any case past both thresholds --
the equilibrium is 20.6 Hill radii out and the dynamics are Keplerian -- so it
was never the interesting case.

Scope of the paper set
──────────────────────
Face-on (alpha = 0), Sun-Earth, beta in [0.001, 0.05].  The figures are

    fig1   halo family vs beta
    fig2   eigenvalue map
    fig3   Floquet exponents
    fig5   sail control authority        (rebuilt on the true 6x2 Jacobian)
    fig8   dissolution of the collinear structure    <- headline
    fig10  halo atlas across the flown sail band

Deliberately NOT in the paper set:

    Figure 4 (reachable acceleration set)  -- deleted.  An off-axis calculation
        with no role in a face-on paper.
    Earth-Moon (fig6, fig7, matched pairs) -- reachable via `earthmoon`, but out
        of scope: the transfer Delta-V does not converge with manifold sampling,
        and the family tables have been superseded three times.  It belongs in a
        second paper, not this one.
    fig5_minimum_beta, fig5_simulation     -- deleted.  Both were LQR results
        computed on a 6x3 unconstrained thruster (bug A3), so they described a
        spacecraft a sail cannot be.
"""

from __future__ import annotations

import io
import os
import subprocess
import sys
import time
import traceback
from contextlib import redirect_stdout

RESULTS = 'results.txt'

PAPER_FIGURES = ('fig1_schematic.png',
                 'fig4_frequency_ratio.png',
                 'fig1_beta_family.png',
                 'fig2_stability.png',
                 'fig3_floquet.png',
                 'fig5_control_authority.png',
                 'fig5_station_keeping.png',
                 'fig8_structure_dissolution.png',
                 'fig10_halo_atlas.png')


# ── small helpers ─────────────────────────────────────────────────────────────

def _rule(title: str = '', ch: str = '=', width: int = 78) -> None:
    if title:
        print('\n' + ch * width)
        print(title)
        print(ch * width)
    else:
        print(ch * width)


def _step(label: str, fn, *args, **kwargs):
    """Run one step, timing it, and never let a failure kill the pipeline."""
    t0 = time.time()
    print(f"  {label} ...", end=' ', flush=True)
    try:
        out = fn(*args, **kwargs)
        print(f"ok  [{time.time() - t0:.1f}s]")
        return True, out
    except Exception as exc:                       # noqa: BLE001
        print(f"FAILED  [{time.time() - t0:.1f}s]")
        print(f"      {type(exc).__name__}: {str(exc)[:150]}")
        _step.failures.append((label, f"{type(exc).__name__}: {exc}"))
        if os.environ.get('SOLARSAIL_TRACE'):
            traceback.print_exc()
        return False, None


_step.failures = []


# ── stages ────────────────────────────────────────────────────────────────────

def stage_verify() -> bool:
    """Run the regression suite and the closed-form assertions as subprocesses."""
    _rule('VERIFY')
    ok = True
    for script in ('regression_test.py', 'test_critical_beta.py',
                   'paper/verify_numbers.py'):
        if not os.path.exists(script):
            print(f"  {script}: not present, skipped")
            continue
        t0 = time.time()
        print(f"  {script} ...", end=' ', flush=True)
        r = subprocess.run([sys.executable, script],
                           capture_output=True, text=True)
        dt = time.time() - t0
        tail = (r.stdout or '').strip().splitlines()
        # 'should FAIL' / 'expected:' label lines describe cases that are
        # SUPPOSED to fail (the corrector rejecting a spurious orbit), so they
        # are not failures of the suite.
        fails = [l for l in tail
                 if 'FAIL' in l
                 and 'should FAIL' not in l
                 and 'expected' not in l.lower()]
        if r.returncode != 0 or fails:
            ok = False
            print(f"FAILED  [{dt:.1f}s]")
            for l in (fails or tail[-6:]):
                print(f"      {l}")
            if r.stderr:
                print(f"      {r.stderr.strip().splitlines()[-1][:150]}")
        else:
            passes = sum('PASS' in l for l in tail)
            print(f"ok  [{dt:.1f}s]  ({passes} PASS lines)")
    return ok


def stage_results() -> bool:
    """Recompute every headline number and write results.txt."""
    _rule('RESULTS')
    from src import critical_beta, frequency_ratio, sail_authority

    buf = io.StringIO()
    with redirect_stdout(buf):
        print("Solar Sail CR3BP — computed results")
        print("Regenerated by main.py; do not edit by hand.")
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\n")

        print("=" * 74)
        print("1.  DISSOLUTION OF THE COLLINEAR STRUCTURE  (headline)")
        print("=" * 74)
        critical_beta.summary()
        b_exact = critical_beta.critical_beta_tidal_exact()
        b_root = critical_beta.critical_beta_tidal()
        print(f"\n  closed form   beta_crit = 1 - (1 - mu^(1/3))^2")
        print(f"                          = {b_exact:.17f}")
        print(f"  brentq on s(beta)-1     = {b_root:.17f}")
        print(f"  |difference|            = {abs(b_exact - b_root):.3e}")

        print("\n" + "=" * 74)
        print("2.  FREQUENCY RATIO AND THE RESONANCE QUESTION")
        print("=" * 74)
        frequency_ratio.verify_extremum()
        print()
        frequency_ratio.resonance_scan()
        print()
        frequency_ratio.summary()
        print()
        frequency_ratio.table_across_systems()

        print("\n" + "=" * 74)
        print("3.  SAIL CONTROL AUTHORITY  (bug A3 characterised)")
        print("=" * 74)
        sail_authority.report()

        print("\n" + "=" * 74)
        print("4.  SAIL TECHNOLOGY: WHERE beta_crit SITS AGAINST REAL HARDWARE")
        print("=" * 74)
        try:
            from src import sail_technology
            # table() and compare_to_threshold() RETURN strings; they do not print.
            for name in ('table', 'compare_to_threshold'):
                fn = getattr(sail_technology, name, None)
                if callable(fn):
                    out = fn()
                    print(out if isinstance(out, str) else '')
                    print()
            lo, hi = sail_technology.flown_beta_range()
            b_crit = critical_beta.critical_beta_tidal_exact()
            print(f"  flown beta range      : {lo:.5f} .. {hi:.5f}")
            print(f"  beta_crit             : {b_crit:.5f}")
            print(f"  factor beyond best flown: {b_crit / hi:.2f}x")
        except Exception as exc:                   # noqa: BLE001
            print(f"  sail_technology unavailable: {type(exc).__name__}: {exc}")

    text = buf.getvalue()
    with open(RESULTS, 'w') as fh:
        fh.write(text)
    print(f"  wrote {RESULTS}  ({len(text.splitlines())} lines)")
    print(f"  headline: beta_crit = {critical_beta.critical_beta_tidal_exact():.8f}")
    return True


def stage_figures() -> bool:
    """Regenerate the paper figures.  Earth-Moon is not included."""
    _rule('FIGURES')
    from src import (atlas, critical_beta, frequency_ratio, paper_extras,
                     sail_authority, schematic)

    _step('fig1  setup schematic', schematic.fig_schematic, verbose=False)
    _step('fig4  frequency ratio',
          frequency_ratio.fig_frequency_ratio, verbose=False)

    # The beta-sweep (halo corrector + monodromy at each beta) costs ~20 s and
    # is shared by figures 1-3.  Compute it once and pass it in; calling each
    # figure with sweep=None recomputes it three times over.
    ok, sweep = _step('beta-sweep (shared by fig1-3)',
                      paper_extras._compute_sweep, verbose=False)
    if not ok:
        sweep = None
    _step('fig1  halo family vs beta', paper_extras.fig_beta_family,
          sweep=sweep, verbose=False)
    _step('fig2  eigenvalue map', paper_extras.fig_stability_sweep,
          sweep=sweep, verbose=False)
    _step('fig3  Floquet exponents', paper_extras.fig_floquet,
          sweep=sweep, verbose=False)
    _step('fig5  sail control authority',
          sail_authority.fig_control_authority, verbose=False)
    _step('fig5sk sensitivity-matrix corrector',
          paper_extras.fig_stationkeeping_halo)
    _step('fig8  structure dissolution',
          critical_beta.fig_structure_dissolution, verbose=False)
    # fig10 plots the CACHED atlas.  It must never rebuild here: the family
    # walk takes tens of minutes and produced no output, which looked exactly
    # like a hang.  `python main.py atlas` is the stage that computes it.
    import os
    if os.path.exists('halo_atlas.csv'):
        _step('fig10 halo atlas (from cache)', atlas.fig_atlas, verbose=False)
    else:
        print("  fig10 halo atlas ... SKIPPED  (no halo_atlas.csv)")
        print("      run `python main.py atlas` first -- it walks the "
              "families, which takes minutes, then writes the cache.")
    return not _step.failures


def stage_atlas() -> bool:
    """Re-walk the halo families and rewrite halo_atlas.csv.  Slow."""
    _rule('ATLAS  (slow: tens of minutes)')
    from src import atlas
    print("  walking 12 halo families by pseudo-arclength.")
    print("  This is the expensive stage; progress is printed per beta.\n")
    ok, a = _step('walking families', atlas.build, verbose=True)
    if ok:
        _step('export halo_atlas.csv', atlas.export_csv, a)
        _step('fig10 halo atlas', atlas.fig_atlas, atlas=a, verbose=False)
        if a.get('suspect'):
            print(f"\n  BRANCH GUARD flagged: {a['suspect']}")
        seeds = {b: br.get('seeded_by', '?') for b, br in
                 sorted(a['families'].items())}
        print(f"  seeding: {seeds}")
    return ok


def stage_earthmoon() -> bool:
    """Earth-Moon material.  NOT part of the paper; see the module docstring."""
    _rule('EARTH-MOON  (out of scope for the paper)')
    print("  These results are excluded from the paper set:")
    print("    * the transfer Delta-V does not converge with manifold sampling")
    print("    * the matched-pair tables have been superseded three times")
    print("  Running them anyway because you asked for this stage.\n")
    from src import heteroclinic
    _step('matched pair', heteroclinic.matched_pair)
    _step('fig6  Poincare map', heteroclinic.fig_poincare_map)
    _step('fig7  manifold transfer', heteroclinic.fig_manifold_transfer)
    return True


def stage_status() -> bool:
    """Report which figures exist and whether any is older than its code."""
    _rule('STATUS')
    srcs = [os.path.join('src', f) for f in os.listdir('src')
            if f.endswith('.py')]
    newest_src = max((os.path.getmtime(f) for f in srcs), default=0.0)
    newest_name = max(srcs, key=os.path.getmtime) if srcs else '-'
    print(f"  newest source file: {newest_name} "
          f"({time.strftime('%m-%d %H:%M', time.localtime(newest_src))})")
    print()
    stale = []
    for f in PAPER_FIGURES:
        if not os.path.exists(f):
            print(f"  {f:<34} MISSING")
            stale.append(f)
            continue
        t = os.path.getmtime(f)
        flag = 'stale' if t < newest_src else 'ok'
        if flag == 'stale':
            stale.append(f)
        print(f"  {f:<34} {time.strftime('%m-%d %H:%M', time.localtime(t))}"
              f"   {flag}")
    print()
    if stale:
        print(f"  {len(stale)} figure(s) older than the newest source — "
              f"run:  python main.py figures")
    else:
        print("  all paper figures are current.")
    return not stale


STAGES = {
    'verify': stage_verify,
    'results': stage_results,
    'figures': stage_figures,
    'atlas': stage_atlas,
    'earthmoon': stage_earthmoon,
    'status': stage_status,
}

PAPER_SET = ('verify', 'results', 'figures')
ALL_SET = ('verify', 'results', 'atlas', 'figures', 'status')


def main(argv) -> int:
    args = [a for a in argv[1:] if not a.startswith('-')]

    if args and args[0] == 'list':
        print(__doc__)
        return 0

    if not args:
        chosen = PAPER_SET
    elif args[0] == 'all':
        chosen = ALL_SET
    else:
        chosen = []
        for a in args:
            if a not in STAGES:
                print(f"unknown stage: {a}\n"
                      f"available: {', '.join(STAGES)}, all, list")
                return 2
            chosen.append(a)

    _rule(f"SOLAR SAIL CR3BP — running: {', '.join(chosen)}", ch='#')
    t0 = time.time()
    _step.failures = []
    results = {}
    for name in chosen:
        try:
            results[name] = STAGES[name]()
        except Exception as exc:                   # noqa: BLE001
            results[name] = False
            print(f"\n  stage '{name}' raised "
                  f"{type(exc).__name__}: {str(exc)[:160]}")
            if os.environ.get('SOLARSAIL_TRACE'):
                traceback.print_exc()

    _rule('SUMMARY')
    for name in chosen:
        print(f"  {name:<12} {'ok' if results.get(name) else 'FAILED'}")
    if _step.failures:
        print(f"\n  {len(_step.failures)} step failure(s):")
        for lbl, err in _step.failures:
            print(f"    {lbl}: {err[:120]}")
    print(f"\n  total {time.time() - t0:.1f}s")
    print("  (set SOLARSAIL_TRACE=1 for full tracebacks)")
    return 0 if all(results.values()) and not _step.failures else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv))
