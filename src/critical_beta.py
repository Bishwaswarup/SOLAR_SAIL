"""
critical_beta.py — Dissolution of the Sun-Earth collinear structure with
                   increasing sail lightness number.

This module replaces the earlier (incorrect) framing in which the beta = 0.5
equilibrium was presented as a "displaced L1" with a "stabilised" halo orbit.

What is actually true
─────────────────────
For a face-on sail (alpha = 0) the radiation-pressure force is radial and
conservative, so it simply rescales solar gravity:

    (1 - mu)  ->  (1 - beta)(1 - mu)

The on-axis equilibrium therefore satisfies, with the Earth DELETED,

    x_hover(beta) = [ (1 - beta)(1 - mu) ]^(1/3)                          (1)

which is a *heliocentric hovering point*: the radius at which reduced solar
gravity balances centrifugal acceleration at the synchronous rate.  It is not a
perturbation of L1.  Including the Earth shifts (1) by only

    dx = -mu / (3 r2^2)  + O(mu^2)

Linear structure.  With the sail conservative, the standard CR3BP linearisation
applies to the effective potential

    U = (x^2 + y^2)/2 + (1-beta)(1-mu)/r1 + mu/r2

giving, on the axis,   U_xx = 1 + 2A,   U_yy = 1 - A,   U_zz = -A,   where

    A(beta) = (1-beta)(1-mu)/r1^3 + mu/r2^3                               (2)

The planar characteristic polynomial is

    lam^4 + (2 - A) lam^2 + (1 + A - 2 A^2) = 0                           (3)

and a real saddle pair exists iff  U_xx U_yy < 0, i.e. iff  A > 1.

Because r1^3 -> (1-beta)(1-mu) as the Earth's influence vanishes, the first term
of (2) tends to exactly 1, so

    A - 1  ->  mu / r2^3   ==  s(beta)                                    (4)

s(beta) is the *saddle strength*: the entire hyperbolic character of the point is
of Earth origin.  s > 0 always, so the saddle never strictly disappears — it
degenerates.  The physically meaningful statement is the rate of that collapse.

Two thresholds
──────────────
  (A) Hill-sphere exit:      r2(beta) = r_H = (mu/3)^(1/3)
      Classical L1 already sits at r2 = 0.9967 r_H, so this triggers almost
      immediately:  beta_crit ~ 3e-4.

  (B) Tidal parity:          s(beta) = 1,  i.e.  r2 = mu^(1/3)
      The Earth's tidal term equals the reduced solar term.  Exactly

          r2_crit = mu^(1/3)  =  3^(1/3) r_H  ~ 1.442 r_H                 (5)

      For Sun-Earth this gives beta_crit ~ 0.0286.  On what that means for
      hardware, see sail_technology.py: reduced from primary specifications,
      EVERY solar sail flown to date sits at beta <= 0.0061 (IKAROS 0.00062 from
      its measured 1.12 mN thrust; LightSail-2 0.0061; ACS3 0.0048), and the most
      ambitious funded design, the cancelled Solar Cruiser, reached 0.0202.  So
      tidal parity is a factor of ~3 beyond the best FLOWN sail and within 40 %
      of a designed one — near-term, NOT current, capability.  An earlier version
      of this docstring claimed "inside the range of *current* sail technology
      (beta ~ 0.01-0.05)"; that band was unsourced and is not supported.

      Threshold (B) has a CLOSED FORM.  Impose r2 = mu^(1/3) on the on-axis
      balance  x - (1-beta)(1-mu)/r1^2 + mu/r2^2 = 0,  where r1 = x + mu and
      r2 = (1-mu) - x.  The substitution fixes every geometric quantity:

          x    = (1-mu) - mu^(1/3)
          r1   = 1 - mu^(1/3)
          mu/r2^2 = mu / mu^(2/3) = mu^(1/3)

      so the balance collapses to

          (1-mu) - mu^(1/3) - (1-beta)(1-mu)/(1-mu^(1/3))^2 + mu^(1/3) = 0
          (1-mu)                = (1-beta)(1-mu)/(1-mu^(1/3))^2

      and the factor (1-mu) CANCELS IDENTICALLY, leaving

          1 - beta = (1 - mu^(1/3))^2

          beta_crit = 1 - (1 - mu^(1/3))^2 = mu^(1/3) (2 - mu^(1/3))       (6)

      Eq. (6) is exact — no expansion in mu, no root-finding.  Two consequences
      worth stating in the manuscript:

        * beta_crit depends on the system ONLY through mu^(1/3).  The (1-mu)
          cancellation means the reduced solar term drops out entirely, so the
          threshold is a pure function of the mass ratio and carries no separate
          dependence on the primary's mass or on the sail's distance from it.

        * beta_crit is monotone increasing on mu in (0,1), reaching 1 at mu = 1.
          Cube-root mass ratios that are exact decimals give exact thresholds:
          mu = 1e-6 -> beta_crit = 0.0199 exactly;  mu = 1e-3 -> 0.19 exactly.

      `critical_beta_tidal_exact` implements (6); `critical_beta_tidal` retains
      the brentq solve as an independent numerical check.  They agree to
      < 3e-16 over mu in [1e-7, 1e-2] (see test_critical_beta.py).

Reference frequencies.  As A -> 1 the roots of (3) become lam^2 = -1 and 0:
the in-plane centre frequency and the out-of-plane frequency both tend to the
mean motion, and the period is forced to exactly 2*pi non-dimensional (one year
for Sun-Earth).  This is the Keplerian epicyclic degeneracy of a 1/r^2 field,
not a resonance with the Earth.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

MU_SE = 3.003e-6           # Sun-Earth mass parameter
AU_KM = 1.495978707e8      # km per non-dimensional length
DAYS_PER_ND = 365.25 / (2.0 * np.pi)

# CVD-validated categorical palette (Okabe-Ito derived; passes all six checks)
C_BLUE   = '#0072B2'
C_GREEN  = '#009E73'
C_ORANGE = '#D55E00'
C_PURPLE = '#8C4799'
C_INK    = '#2b2b2b'
C_MUTED  = '#8a8a8a'
C_GRID   = '#dcdcdc'


# ── core scalar relations ─────────────────────────────────────────────────────

def hill_radius(mu: float = MU_SE) -> float:
    """Hill radius of the smaller primary, (mu/3)^(1/3)."""
    return (mu / 3.0) ** (1.0 / 3.0)


def hover_radius(beta: float, mu: float = MU_SE) -> float:
    """Heliocentric hovering radius, eq. (1) — the Earth deleted entirely."""
    return ((1.0 - beta) * (1.0 - mu)) ** (1.0 / 3.0)


def _f_axis(x: float, beta: float, mu: float) -> float:
    """
    On-axis force balance for a spacecraft sunward of the smaller primary:
        centrifugal(+)  -  reduced solar gravity  +  Earth's pull(+, outward)
    """
    r1 = x + mu
    r2 = (1.0 - mu) - x
    return x - (1.0 - beta) * (1.0 - mu) / r1**2 + mu / r2**2


def equilibrium(beta: float, mu: float = MU_SE) -> float:
    """
    Sail equilibrium on the x-axis between the primaries (full three-body).

    d(_f_axis)/dx = 1 + 2(1-beta)(1-mu)/r1^3 + 2 mu/r2^3 > 0, so _f_axis is
    strictly increasing on (0, 1-mu) and the root is unique — widening the
    bracket cannot change which root is returned.  The lower end is 1e-9 rather
    than the former 1e-3 so that large-mu / large-beta cases still bracket:
    _f_axis(0+) -> -(1-beta)(1-mu)/mu^2, which is only strongly negative while
    beta stays below the existence limit 1 - (mu/(1-mu))^3 of eq. (7).
    """
    return brentq(_f_axis, 1e-9, (1.0 - mu) - 1e-12,
                  args=(beta, mu), xtol=1e-15, rtol=8.9e-16)


def A_parameter(beta: float, mu: float = MU_SE, x: float | None = None) -> float:
    """A(beta) of eq. (2).  A real saddle pair exists iff A > 1."""
    if x is None:
        x = equilibrium(beta, mu)
    r1 = x + mu
    r2 = (1.0 - mu) - x
    return (1.0 - beta) * (1.0 - mu) / r1**3 + mu / r2**3


def saddle_strength(beta: float, mu: float = MU_SE, x: float | None = None) -> float:
    """s(beta) = mu / r2^3, eq. (4) — the Earth-origin hyperbolicity."""
    if x is None:
        x = equilibrium(beta, mu)
    r2 = (1.0 - mu) - x
    return mu / r2**3


def linear_modes(beta: float, mu: float = MU_SE) -> dict:
    """
    Linear structure at the on-axis equilibrium.

    Returns
    -------
    dict with keys
        x        equilibrium abscissa
        A        the A parameter
        lam_u    real saddle exponent (0.0 if none)
        omega    in-plane centre frequency
        nu       out-of-plane frequency, sqrt(A)
        T_plane  in-plane period, 2*pi/omega
    """
    x = equilibrium(beta, mu)
    A = A_parameter(beta, mu, x)
    roots = np.roots([1.0, 2.0 - A, 1.0 + A - 2.0 * A**2])

    lam_u = 0.0
    omega = np.nan
    for r in roots:
        r = complex(r)
        if abs(r.imag) > 1e-11:
            continue
        if r.real > 0:
            lam_u = float(np.sqrt(r.real))
        elif r.real < 0:
            omega = float(np.sqrt(-r.real))

    return dict(x=x, A=A, lam_u=lam_u, omega=omega,
                nu=float(np.sqrt(A)),
                T_plane=(2.0 * np.pi / omega if omega == omega else np.nan))


# ── the two thresholds ────────────────────────────────────────────────────────

def critical_beta_hill(mu: float = MU_SE) -> float:
    """beta at which the equilibrium's standoff equals the Hill radius."""
    rH = hill_radius(mu)
    g = lambda b: ((1.0 - mu) - equilibrium(b, mu)) - rH
    return brentq(g, 1e-9, 0.5, xtol=1e-15)


def critical_beta_tidal_exact(mu: float = MU_SE) -> float:
    """
    beta at tidal parity, in CLOSED FORM — eq. (6) of the module docstring:

        beta_crit = 1 - (1 - mu^(1/3))^2 = mu^(1/3) (2 - mu^(1/3))

    Exact: imposing r2 = mu^(1/3) on the on-axis balance cancels the (1-mu)
    factor identically, so no expansion in mu and no root-finding is involved.

    Implementation note.  The two brackets above are the same expression, but
    `1 - (1 - m)^2` subtracts two nearly equal quantities and loses precision as
    m -> 0, whereas `m (2 - m)` does not.  Measured relative error against a
    50-digit reference:

        mu       1 - (1-m)^2      m (2 - m)
        1e-2       9.5e-17         4.9e-17
        1e-7       2.0e-15         1.6e-16
        1e-10      2.6e-14         4.7e-16
        1e-15      5.0e-12         5.5e-16

    The cancelling form would therefore break a 1e-14 tolerance below about
    mu ~ 1e-10.  The stable rearrangement is used here; the manuscript may quote
    either, since they are algebraically identical.
    """
    m = mu ** (1.0 / 3.0)
    return float(m * (2.0 - m))


def critical_beta_tidal(mu: float = MU_SE) -> float:
    """
    beta at tidal parity, s(beta) = 1  <=>  r2 = mu^(1/3), eq. (5).

    Solved numerically by brentq.  Kept as an INDEPENDENT check on
    `critical_beta_tidal_exact` — it goes through `equilibrium()` and
    `saddle_strength()`, so agreement between the two exercises the whole
    on-axis force-balance path rather than just the algebra.

    The bracket is derived rather than hard-coded.  Two limits constrain it:

      * the previous fixed upper bound of 0.5 silently raised "f(a) and f(b)
        must have different signs" for every mu above (1 - 1/sqrt(2))^3 = 0.0251,
        which excluded e.g. Pluto-Charon (mu = 0.104, beta_crit = 0.719);

      * an on-axis equilibrium between the primaries exists only while reduced
        solar gravity can still overcome the secondary's outward pull at small x.
        Requiring _f_axis(0+) < 0 gives the EXISTENCE LIMIT

            beta < 1 - (mu / (1-mu))^3                                    (7)

        beyond which `equilibrium()` has no root to find.  For Sun-Earth (7) is
        1 - 3e-17, i.e. no practical constraint, but at mu = 0.3 it is 0.921 —
        only marginally above beta_crit = 0.891.  The upper bracket is therefore
        placed at the midpoint of beta_crit and (7), which is guaranteed to lie
        in (beta_crit, limit) because s(beta) decreases monotonically through 1.

    Note s(beta) is monotone DECREASING in beta: a larger beta pushes the
    equilibrium sunward, growing r2 and shrinking s = mu/r2^3.  So g(0) > 0 and
    g(b_hi) < 0.
    """
    b_exact = critical_beta_tidal_exact(mu)
    b_limit = 1.0 - (mu / (1.0 - mu)) ** 3
    if not (b_exact < b_limit):
        raise ValueError(
            f"tidal parity at beta={b_exact:.6f} lies beyond the existence "
            f"limit {b_limit:.6f} for mu={mu:g}: no on-axis equilibrium exists "
            f"there, so the threshold is formal only.")
    b_hi = min(b_exact * 1.5 + 1e-6, 0.5 * (b_exact + b_limit))
    g = lambda b: saddle_strength(b, mu) - 1.0
    return brentq(g, 1e-12, b_hi, xtol=1e-15, rtol=8.9e-16)


def summary(mu: float = MU_SE, verbose: bool = True) -> dict:
    """Compute and optionally print the headline numbers."""
    rH = hill_radius(mu)
    b_hill = critical_beta_hill(mu)
    b_tide_exact = critical_beta_tidal_exact(mu)
    b_tide = critical_beta_tidal(mu)

    out = dict(
        r_hill=rH,
        beta_hill=b_hill,
        beta_tidal=b_tide_exact,          # the closed form is the reference
        beta_tidal_brentq=b_tide,         # independent numerical check
        beta_tidal_residual=abs(b_tide - b_tide_exact),
        r2_tidal=mu ** (1.0 / 3.0),
        r2_over_rH_tidal=3.0 ** (1.0 / 3.0),
        x_classical=equilibrium(0.0, mu),
        x_tidal=equilibrium(b_tide_exact, mu),
    )

    if verbose:
        print(f"  Hill radius                r_H       = {rH:.6e} nd"
              f"  ({rH*AU_KM:,.0f} km)")
        print(f"  Classical L1 standoff      r2/r_H    = "
              f"{((1-mu)-out['x_classical'])/rH:.4f}   (inside the Hill sphere)")
        print(f"  (A) Hill-sphere exit       beta_crit = {b_hill:.4e}")
        print(f"  (B) Tidal parity s=1       beta_crit = {b_tide_exact:.6f}")
        print(f"      closed form  1-(1-mu^(1/3))^2         = {b_tide_exact:.17f}")
        print(f"      brentq on s(beta)-1 (independent)     = {b_tide:.17f}")
        print(f"      residual                             = "
              f"{out['beta_tidal_residual']:.3e}")
        print(f"      exact:  r2 = mu^(1/3)  = {out['r2_tidal']:.6f} nd"
              f"  = 3^(1/3) r_H = {out['r2_over_rH_tidal']:.4f} r_H")
        print(f"      standoff = {((1-mu)-out['x_tidal'])*AU_KM:,.0f} km")

    return out


# ── the figure ────────────────────────────────────────────────────────────────

def fig_structure_dissolution(output: str = 'fig8_structure_dissolution.png',
                              n: int = 400,
                              mu: float = MU_SE,
                              verbose: bool = True) -> dict:
    """
    Four-panel figure documenting the dissolution of the collinear structure.
    Plain journal style: white ground, black marks, dash-coded series.

    (a) equilibrium abscissa, full balance vs the Earth-free hovering law
    (b) standoff from the Earth in Hill radii, with both thresholds
    (c) saddle strength s(beta), four decades of collapse
    (d) linear rates converging on the Keplerian degeneracy
    """
    from src.paperstyle import use, panel_label, thin_guide
    use()
    import matplotlib.pyplot as plt

    if verbose:
        print("  Computing dissolution sweep ...")

    betas = np.concatenate([
        np.logspace(-5, -2.2, n // 2),
        np.linspace(10**-2.2, 0.60, n // 2),
    ])

    rH = hill_radius(mu)
    x_full, x_hov, d_rH, s_arr = [], [], [], []
    lam_u, omega, nu = [], [], []

    for b in betas:
        m = linear_modes(b, mu)
        x_full.append(m['x'])
        x_hov.append(hover_radius(b, mu))
        r2 = (1.0 - mu) - m['x']
        d_rH.append(r2 / rH)
        s_arr.append(mu / r2**3)
        lam_u.append(m['lam_u'])
        omega.append(m['omega'])
        nu.append(m['nu'])

    x_full = np.array(x_full); x_hov = np.array(x_hov)
    d_rH = np.array(d_rH);     s_arr = np.array(s_arr)
    lam_u = np.array(lam_u);   omega = np.array(omega); nu = np.array(nu)

    b_hill = critical_beta_hill(mu)
    b_tide = critical_beta_tidal(mu)

    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.9))

    # -- (a) equilibrium position ------------------------------------------
    ax = axes[0, 0]
    ax.semilogx(betas, x_full, 'k-', lw=1.0,
                label='full three-body balance')
    l, = ax.semilogx(betas, x_hov, 'k-', lw=1.0,
                     label=r'Earth deleted: $[(1-\beta)(1-\mu)]^{1/3}$')
    l.set_dashes([5, 2])
    ax.set_ylim(0.715, 1.02)
    thin_guide(ax, y=1.0 - mu)
    ax.text(1.4e-5, 1.0 - mu + 0.004, 'Earth', fontsize=7.5, color='0.3')
    ax.set_xlabel(r'$\beta$')
    ax.set_ylabel(r'$x_{\rm eq}$  [nd]')
    ax.legend(loc='lower left', fontsize=7.2)
    panel_label(ax, '(a)')

    axin = ax.inset_axes([0.55, 0.40, 0.41, 0.27])
    sel = betas > 1e-3
    axin.loglog(betas[sel], np.abs(x_full - x_hov)[sel] * AU_KM, 'k-', lw=0.9)
    axin.set_title(r"Earth's entire effect [km]", fontsize=6.6, pad=3)
    axin.tick_params(labelsize=5.8)
    axin.set_facecolor('white')

    # -- (b) standoff in Hill radii ----------------------------------------
    ax = axes[0, 1]
    ax.loglog(betas, d_rH, 'k-', lw=1.0)
    thin_guide(ax, y=1.0)
    thin_guide(ax, y=3.0**(1/3))
    ax.text(1.4e-5, 1.04, r'$r_2 = r_H$', fontsize=7.0, color='0.3')
    ax.text(1.4e-5, 3.0**(1/3) * 1.06, r'$r_2 = 3^{1/3} r_H$',
            fontsize=7.0, color='0.3')
    for bc, lab in ((b_hill, rf'$\beta={b_hill:.1e}$'),
                    (b_tide, rf'$\beta={b_tide:.4f}$')):
        thin_guide(ax, x=bc)
        ax.annotate(lab, xy=(bc, 2.6), fontsize=6.8, rotation=90,
                    va='bottom', ha='right')
    ax.set_xlabel(r'$\beta$')
    ax.set_ylabel(r'$r_2 / r_H$')
    panel_label(ax, '(b)')

    # -- (c) saddle strength -----------------------------------------------
    ax = axes[1, 0]
    ax.loglog(betas, s_arr, 'k-', lw=1.0)
    ax.set_ylim(8e-5, 12.0)
    thin_guide(ax, y=1.0)
    thin_guide(ax, x=b_tide)
    ax.annotate(rf'$s=1$ at $\beta={b_tide:.4f}$', xy=(b_tide * 1.4, 2.2),
                fontsize=7.2)
    s_half = saddle_strength(0.5, mu)
    ax.plot([0.5], [s_half], 'ko', ms=3.5)
    ax.annotate(rf'$\beta=0.5$: $s={s_half:.1e}$',
                xy=(0.5, s_half), xytext=(0.0016, 1.1e-3),
                fontsize=7.2, va='center',
                arrowprops=dict(arrowstyle='-', color='black', lw=0.6,
                                shrinkA=4, shrinkB=3))
    ax.set_xlabel(r'$\beta$')
    ax.set_ylabel(r'$s = \mu / r_2^{\,3}$')
    panel_label(ax, '(c)')

    # -- (d) linear rates --------------------------------------------------
    ax = axes[1, 1]
    ax.semilogx(betas, omega, 'k-', lw=1.0, label=r'$\omega$ (in-plane)')
    l2, = ax.semilogx(betas, nu, 'k-', lw=1.0, label=r'$\nu=\sqrt{A}$ (vertical)')
    l2.set_dashes([5, 2])
    l3, = ax.semilogx(betas, lam_u, 'k-', lw=1.0, label=r'$\lambda_u$ (saddle)')
    l3.set_dashes([1, 1.6])
    thin_guide(ax, y=1.0)
    ax.text(1.4e-5, 1.05, 'mean motion', fontsize=7.0, color='0.3')
    thin_guide(ax, x=b_tide)
    ax.set_ylim(-0.12, 2.9)
    ax.set_xlabel(r'$\beta$')
    ax.set_ylabel('linear rate  [nd]')
    ax.legend(loc='lower left', fontsize=7.2)
    panel_label(ax, '(d)')

    fig.suptitle(r'Dissolution of the Sun--Earth collinear structure with '
                 r'sail lightness number', fontsize=9.5, y=0.998)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output)
    plt.close(fig)
    if verbose:
        print(f"  Saved -> {output}")

    return dict(betas=betas, x_full=x_full, x_hover=x_hov, d_rH=d_rH,
                s=s_arr, lam_u=lam_u, omega=omega, nu=nu,
                beta_hill=b_hill, beta_tidal=b_tide)


if __name__ == '__main__':
    print("\n── Critical β analysis ───────────────────────────────")
    summary()
    print()
    fig_structure_dissolution()
