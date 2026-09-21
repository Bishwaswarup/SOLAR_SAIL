"""
verify_numbers.py — machine-check every numeric claim in main.tex.

Run from the repository root:   python paper/verify_numbers.py

Each check recomputes a quantity from the source modules and compares it with
the literal typed into the manuscript.  A manuscript number that drifts from the
code fails here rather than in review.
"""
from __future__ import annotations

import re
import sys

import numpy as np

sys.path.insert(0, '.')

from src.critical_beta import (MU_SE, A_parameter, critical_beta_hill,
                               critical_beta_tidal, critical_beta_tidal_exact,
                               equilibrium, hill_radius, linear_modes,
                               saddle_strength)
from src.frequency_ratio import (A_STAR, LAMBDA_U_STAR, NU_OMEGA_MIN, NU_STAR,
                                 OMEGA_STAR, attainability,
                                 beta_at_frequency_extremum, classical_A,
                                 nu_over_omega, resonance_scan)
from src.sail_authority import (ALPHA_STAR, COND_AT_STAR, COS2_ALPHA_STAR,
                                controllability_rank, linearised_cr3bp,
                                sail_control_jacobian, thruster_jacobian)

TEX = 'paper/main.tex'
AU_KM = 1.495978707e8

checks, failures = [], []


def chk(label, computed, claimed, tol=None, rel=False):
    if tol is None:
        tol = abs(claimed) * 1e-4 if rel else 5e-6
    ok = abs(computed - claimed) <= tol
    checks.append((label, computed, claimed, ok))
    if not ok:
        failures.append(label)
    return ok


# ── Section 3-4: the closed form ─────────────────────────────────────────────
b_exact = critical_beta_tidal_exact(MU_SE)
chk('beta_crit closed form (Sun-Earth)', b_exact, 0.028646456169, tol=1e-11)
chk('beta_crit root-find agreement', abs(b_exact - critical_beta_tidal(MU_SE)),
    0.0, tol=1e-14)
chk('r2 at parity = mu^(1/3)', MU_SE ** (1 / 3), 0.014427, tol=1e-6)
chk('r2/r_H at parity = 3^(1/3)', MU_SE ** (1 / 3) / hill_radius(MU_SE),
    1.4422, tol=1e-4)
chk('standoff at parity [km]', MU_SE ** (1 / 3) * AU_KM, 2158294, tol=2.0)
chk('classical L1 standoff / r_H',
    ((1 - MU_SE) - equilibrium(0.0, MU_SE)) / hill_radius(MU_SE),
    0.9967, tol=5e-4)
chk('Hill-exit beta', critical_beta_hill(MU_SE), 2.9814e-4, tol=1e-8)

for name, mu, claimed in [('Sun-Mercury', 1.66e-7, 0.010961524792),
                          ('Sun-Earth', MU_SE, 0.028646456169),
                          ('Sun-Jupiter', 9.5388e-4, 0.187186695655),
                          ('Earth-Moon', 1.215e-2, 0.406934946280)]:
    chk(f'beta_crit {name}', critical_beta_tidal_exact(mu), claimed, tol=1e-11)

# ── Section 3: A and lambda_u at parity ─────────────────────────────────────
A_par = A_parameter(b_exact, MU_SE)
chk('A at parity', A_par, 2.014635, tol=1e-5)
chk('A_parity closed form', 1 + (1 - MU_SE) / (1 - MU_SE ** (1 / 3)),
    A_par, tol=1e-9)
chk('lambda_u at parity', linear_modes(b_exact, MU_SE)['lam_u'], 1.505418,
    tol=1e-5)
chk('S at parity', (1 - MU_SE) / (1 - MU_SE ** (1 / 3)), 1.014635, tol=1e-5)

for b, s_claim, lu_claim in [(0.0, 3.0303, 2.53256), (0.010, 2.123, 2.13831),
                             (0.050, 0.4012, 0.99449), (0.100, 0.06820, 0.44400),
                             (0.500, 3.419e-4, 0.03578)]:
    chk(f's(beta={b})', saddle_strength(b, MU_SE), s_claim, rel=True)
    chk(f'lambda_u(beta={b})', linear_modes(b, MU_SE)['lam_u'], lu_claim,
        tol=1e-4)

x5 = equilibrium(0.5, MU_SE)
chk('Earth contribution at beta=0.5 [km]',
    abs(x5 - ((1 - 0.5) * (1 - MU_SE)) ** (1 / 3)) * AU_KM, 3817, tol=3.0)
chk('standoff at beta=0.5 [km]', ((1 - MU_SE) - x5) * AU_KM, 30865449, tol=60.0)
chk('standoff at beta=0.5 [r_H]',
    ((1 - MU_SE) - x5) / hill_radius(MU_SE), 20.6, tol=0.05)

# ── Section 3.2: A is Richardson's c2 ───────────────────────────────────────
def richardson_c2(mu):
    """c2 = (1/g^3)[mu + (1-mu) g^3/(1-g)^3] with g the distance to m2."""
    g = (1 - mu) - equilibrium(0.0, mu)
    return mu / g ** 3 + (1 - mu) / (1 - g) ** 3


for name, mu in [('Sun-Earth', MU_SE), ('Earth-Moon', 1.215e-2),
                 ('Sun-Jupiter', 9.5388e-4)]:
    chk(f'A == Richardson c2 ({name})',
        abs(A_parameter(0.0, mu) - richardson_c2(mu)), 0.0, tol=1e-13)

# ── Section 5: frequency ratio ──────────────────────────────────────────────
chk('A_star', A_STAR, 8 / 5, tol=1e-15)
chk('nu/omega bound', NU_OMEGA_MIN, 0.942809041582, tol=1e-12)
chk('nu^2 at A_star', NU_STAR ** 2, 8 / 5, tol=1e-12)
chk('omega^2 at A_star', OMEGA_STAR ** 2, 9 / 5, tol=1e-12)
chk('lambda_u^2 at A_star', LAMBDA_U_STAR ** 2, 7 / 5, tol=1e-12)
chk('band width [%]', 100 * (1 - NU_OMEGA_MIN), 5.72, tol=5e-3)
chk('nu/omega at classical L1', nu_over_omega(A_parameter(0.0, MU_SE)),
    0.9659, tol=1e-4)

rs = resonance_scan(verbose=False)
chk('low-order rationals reachable', rs['n_reachable'], 0, tol=0)
chk('rationals tested', len(rs['hits']), 14, tol=0)

b_star = beta_at_frequency_extremum(MU_SE)
chk('beta at A=8/5', b_star, 0.040932, tol=1e-5)
chk('s at A=8/5', saddle_strength(b_star, MU_SE), 0.5897, tol=1e-4)
chk('beta_star/beta_tidal', b_star / b_exact, 1.4289, tol=1e-3)

att = attainability(verbose=False)
for w, lo_c, hi_c in [('L1', 4.042, 8.000), ('L2', 1.570, 3.959),
                      ('L3', 1.000, 1.570)]:
    chk(f'A({w}) min', att['ranges'][w][0], lo_c, tol=1e-3)
    chk(f'A({w}) max', att['ranges'][w][1], hi_c, tol=1e-3)
chk('mu at classical A=8/5', att['mu_at_A_star'], 0.480187660, tol=1e-8)
chk('L3 max A', att['A_L3_max'], 1.569787, tol=1e-6)
chk('L3 miss of the bound', att['ratio_L3_max'] - NU_OMEGA_MIN, 3.07e-5,
    tol=1e-7)
for w, claimed in [('L1', 4.060819), ('L2', 3.940764), ('L3', 1.000003)]:
    chk(f'Sun-Earth classical A({w})', classical_A(w, MU_SE), claimed, tol=1e-5)

# ── Section 7: control authority ────────────────────────────────────────────
chk('alpha_star [deg]', np.degrees(ALPHA_STAR), 35.264, tol=1e-3)
chk('cos^2 alpha_star', COS2_ALPHA_STAR, 2 / 3, tol=1e-12)
chk('sigma2/sigma1 at alpha_star', COND_AT_STAR, 1 / 3, tol=1e-12)

A_ctrl = A_parameter(0.05, MU_SE)
chk('A at beta=0.05', A_ctrl, 1.409194, tol=1e-5)
M = linearised_cr3bp(A_ctrl)
xeq = equilibrium(0.05, MU_SE)
chk('rank, ideal thruster', controllability_rank(M, thruster_jacobian()), 6,
    tol=0)
for a0, d0, claimed in [(0.0, 0.0, 4), (0.0, 90.0, 2), (0.5, 0.0, 6)]:
    B = sail_control_jacobian((xeq, 0, 0), np.radians(a0), np.radians(d0),
                              0.05, MU_SE)
    chk(f'rank, sail alpha0={a0} delta0={d0}', controllability_rank(M, B),
        claimed, tol=0)

B0 = sail_control_jacobian((xeq, 0, 0), 0.0, 0.0, 0.05, MU_SE)
chk('|da/ddelta| at alpha0=0', float(np.linalg.norm(B0[3:, 1])), 0.0, tol=1e-14)
sv = np.linalg.svd(sail_control_jacobian(
    (xeq, 0, 0), np.radians(0.5), 0.0, 0.05, MU_SE)[3:, :], compute_uv=False)
chk('conditioning at alpha0=0.5deg', sv[0] / sv[1], 115.0, tol=2.0)

# ── Section 8: sail technology ──────────────────────────────────────────────
try:
    from src.sail_technology import flown_beta_range
    lo, hi = flown_beta_range()
    chk('best flown beta', hi, 0.00613, tol=1e-5)
    chk('lowest flown beta', lo, 0.00062, tol=1e-5)
    chk('shortfall vs best flown', b_exact / hi, 4.7, tol=0.05)
except Exception as exc:                                      # noqa: BLE001
    checks.append(('sail_technology import', float('nan'), float('nan'), False))
    failures.append(f'sail_technology: {exc}')

# ── Section 6: the atlas CSV must match what the manuscript claims ──────────
# These were NOT checked before, and that is how the atlas silently lost its
# three lowest beta (an over-strict branch guard) while the manuscript went on
# claiming twelve families over [0.001, 0.05].  Scalars alone are not enough.
import csv as _csv
import os as _os

if _os.path.exists('halo_atlas.csv'):
    _rows = list(_csv.DictReader(open('halo_atlas.csv')))
    _b = sorted({float(r['beta']) for r in _rows})
    chk('atlas: number of beta values', len(_b), 12, tol=0)
    chk('atlas: lowest beta', min(_b), 0.001, tol=1e-9)
    chk('atlas: highest beta', max(_b), 0.050, tol=1e-9)

    _d = [float(r['delta']) for r in _rows]
    chk('atlas: all delta same sign as reference',
        float(min(_d) * max(_d) > 0), 1.0, tol=0)

    _past = sum(float(r['x0']) >= 1.0 - MU_SE for r in _rows)
    chk('atlas: members past the smaller primary', _past, 0, tol=0)

    _bad_med = 0
    for _bb in _b:
        _g = [r for r in _rows if float(r['beta']) == _bb]
        _med = float(np.median([float(r['x0']) for r in _g]))
        if _med >= float(_g[0]['x_eq']):
            _bad_med += 1
    chk('atlas: families not sunward (median x0)', _bad_med, 0, tol=0)

    if 'seeded_by' in _rows[0]:
        _seeds = {}
        for r in _rows:
            _seeds.setdefault(float(r['beta']), r['seeded_by'])
        _boot = [b for b, v in _seeds.items() if v != 'chained']
        # only the first beta should bootstrap; more means the chain is failing
        chk('atlas: beta values not chained (expect 1, the seed)',
            len(_boot), 1, tol=0)
else:
    checks.append(('atlas: halo_atlas.csv present', 0.0, 1.0, False))
    failures.append('halo_atlas.csv missing -- run `python main.py atlas`')

# ── Table 2 (tab:closed): EVERY cell, parsed out of the manuscript ──────────
# These eight cells were the only numbers in the paper that nothing checked,
# and every one of them had silently drifted: the |Delta| column was stale
# (Sun-Earth read 6.9e-18 while results.txt printed 2.082e-16 for the same
# quantity -- the brentq bracket in critical_beta_tidal() changed underneath
# it), and three r2 cells disagreed with the mu printed beside them.  Parsing
# the row out of main.tex rather than hard-coding it means the check cannot
# itself go stale: edit the table and this either passes or tells you why.
tex = open(TEX).read()

_row_re = re.compile(
    r'^\s*(Sun--Mercury|Sun--Earth|Sun--Jupiter|Earth--Moon)\s*&\s*'
    r'\\num\{([^}]*)\}\s*&\s*'          # mu
    r'\\num\{([^}]*)\}\s*&\s*'          # beta_crit, closed form
    r'\\num\{([^}]*)\}\s*&\s*'          # beta_crit, root-find
    r'\\num\{([^}]*)\}\s*&\s*'          # |Delta|
    r'\\num\{([^}]*)\}\s*\\\\',         # r2
    re.M)

_rows = _row_re.findall(tex)
chk('tab:closed: rows parsed from main.tex', len(_rows), 4, tol=0)

for _sys, _mu_s, _cf_s, _rf_s, _d_s, _r2_s in _rows:
    _mu = float(_mu_s)
    _cf, _rf = critical_beta_tidal_exact(_mu), critical_beta_tidal(_mu)

    # closed form and root-find, as typeset
    chk(f'tab:closed {_sys}: closed form', _cf, float(_cf_s), tol=5e-13)
    chk(f'tab:closed {_sys}: root-find', _rf, float(_rf_s), tol=5e-13)

    # the two beta columns are printed to 12 dp and must agree at that width
    chk(f'tab:closed {_sys}: two beta columns agree as printed',
        float(_cf_s), float(_rf_s), tol=5e-13)

    # |Delta| is floating-point noise, so check the ORDER, not the digits:
    # the typeset value must be within a factor of 2 of the true residual and
    # must not be quoted smaller than it really is.
    _d_true, _d_tex = abs(_cf - _rf), float(_d_s)
    chk(f'tab:closed {_sys}: |Delta| not understated',
        1.0 if _d_tex >= _d_true * 0.5 else 0.0, 1.0, tol=0)
    chk(f'tab:closed {_sys}: |Delta| right order of magnitude',
        1.0 if _d_tex <= max(_d_true * 2.0, 1e-17) else 0.0, 1.0, tol=0)

    # r2 at parity is mu^(1/3) EXACTLY, evaluated at the mu printed beside it.
    # This is what caught Sun-Mercury (0.005500 vs 0.005496) and the
    # Sun-Jupiter mu/r2 mismatch.
    chk(f'tab:closed {_sys}: r2 = mu^(1/3) at the printed mu',
        _mu ** (1 / 3), float(_r2_s), tol=5e-7)

# The manuscript's sweep claim: worst |Delta| over 61 log-spaced mu.  The
# abstract and Sec. 4 both quote a number for this; it must be the sweep's
# actual maximum, not one system's residual.
_sweep = np.logspace(-7, -2, 61)
_worst = max(abs(critical_beta_tidal_exact(m) - critical_beta_tidal(m))
             for m in _sweep)
_m = re.search(r'never differ by more than \\num\{([^}]*)\}', tex)
if _m:
    chk('Sec 4: quoted 61-point sweep bound is not understated',
        1.0 if float(_m.group(1)) >= _worst else 0.0, 1.0, tol=0)
    chk('Sec 4: quoted sweep bound is not wildly loose',
        1.0 if float(_m.group(1)) <= _worst * 10 else 0.0, 1.0, tol=0)
else:
    checks.append(('Sec 4: sweep bound sentence found in main.tex', 0.0, 1.0, False))
    failures.append('Sec 4 sweep-bound sentence not found -- did the wording change?')

# ── cross-check: do these literals actually appear in main.tex? ─────────────
literals = ['0.028646456169', '7:8:9', '0.480187660',
            '35.264', '1.569787', '0.00613', '4.060819', '2.014635',
            '1.505418', '0.040932',
            'c_2', 'ceccaroni2016', 'richardson1980', 'gomez2001',
            'almost equal values of the frequencies']
missing = [t for t in literals if t.replace('\\\\', '\\') not in tex
           and t not in tex]

# ── report ──────────────────────────────────────────────────────────────────
print('=' * 78)
print('MANUSCRIPT NUMBER VERIFICATION')
print('=' * 78)
w = max(len(c[0]) for c in checks)
for label, comp, claim, ok in checks:
    flag = 'ok  ' if ok else 'FAIL'
    print(f'  {flag}  {label:<{w}}  computed {comp:<18.10g} claimed {claim:.10g}')
print()
print(f'  {len(checks) - len(failures)}/{len(checks)} checks passed')
if failures:
    print(f'  FAILURES: {failures}')
if missing:
    print(f'  literals not found in {TEX}: {missing}')
print()
sys.exit(1 if (failures or missing) else 0)
