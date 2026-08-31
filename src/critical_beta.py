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

      For Sun-Earth this gives beta_crit ~ 0.0286 — inside the range of
      *current* sail technology (beta ~ 0.01-0.05).

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
    """Sail equilibrium on the x-axis between the primaries (full three-body)."""
    return brentq(_f_axis, 1e-3, (1.0 - mu) - 1e-12,
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


def critical_beta_tidal(mu: float = MU_SE) -> float:
    """beta at tidal parity, s(beta) = 1  <=>  r2 = mu^(1/3), eq. (5)."""
    g = lambda b: saddle_strength(b, mu) - 1.0
    return brentq(g, 1e-9, 0.5, xtol=1e-15)


def summary(mu: float = MU_SE, verbose: bool = True) -> dict:
    """Compute and optionally print the headline numbers."""
    rH = hill_radius(mu)
    b_hill = critical_beta_hill(mu)
    b_tide = critical_beta_tidal(mu)

    out = dict(
        r_hill=rH,
        beta_hill=b_hill,
        beta_tidal=b_tide,
        r2_tidal=mu ** (1.0 / 3.0),
        r2_over_rH_tidal=3.0 ** (1.0 / 3.0),
        x_classical=equilibrium(0.0, mu),
        x_tidal=equilibrium(b_tide, mu),
    )

    if verbose:
        print(f"  Hill radius                r_H       = {rH:.6e} nd"
              f"  ({rH*AU_KM:,.0f} km)")
        print(f"  Classical L1 standoff      r2/r_H    = "
              f"{((1-mu)-out['x_classical'])/rH:.4f}   (inside the Hill sphere)")
        print(f"  (A) Hill-sphere exit       beta_crit = {b_hill:.4e}")
        print(f"  (B) Tidal parity s=1       beta_crit = {b_tide:.6f}")
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

    (a) equilibrium abscissa, full balance vs the Earth-free hovering law
    (b) standoff from the Earth in Hill radii, with both thresholds
    (c) saddle strength s(beta), four decades of collapse
    (d) linear frequencies converging on the Keplerian degeneracy
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if verbose:
        print("  Computing dissolution sweep …")

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

    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.6))
    fig.suptitle('Dissolution of the Sun–Earth collinear structure '
                 'with sail lightness number',
                 fontsize=13.5, fontweight='semibold', y=0.975, color=C_INK)

    def _style(ax):
        ax.grid(True, which='major', color=C_GRID, lw=0.6, alpha=0.9)
        ax.set_axisbelow(True)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        for sp in ('left', 'bottom'):
            ax.spines[sp].set_color(C_MUTED)
            ax.spines[sp].set_linewidth(0.8)
        ax.tick_params(colors=C_INK, labelsize=9, width=0.8)

    # ── (a) equilibrium position ────────────────────────────────────────────
    ax = axes[0, 0]; _style(ax)
    ax.semilogx(betas, x_full, lw=2.0, color=C_BLUE,
                label='full three-body balance')
    ax.semilogx(betas, x_hov, lw=2.0, ls=(0, (5, 3)), color=C_ORANGE,
                label=r'Earth deleted:  $[(1-\beta)(1-\mu)]^{1/3}$')
    ax.axhline(1.0 - mu, color=C_MUTED, lw=1.0, ls=':')
    ax.set_ylim(0.715, 1.022)
    ax.text(1.4e-5, 1.0 - mu + 0.005, 'Earth', fontsize=8.5, color=C_MUTED)
    ax.set_xlabel(r'sail lightness number  $\beta$', fontsize=10)
    ax.set_ylabel(r'equilibrium abscissa  $x_{\rm eq}$  [nd]', fontsize=10)
    ax.set_title('(a)  the equilibrium is a heliocentric hovering point',
                 fontsize=10.5, loc='left', color=C_INK, pad=8)
    ax.legend(frameon=False, fontsize=8.8, loc=(0.02, 0.055))

    axin = ax.inset_axes([0.50, 0.22, 0.45, 0.30])
    sel = betas > 1e-3
    axin.loglog(betas[sel], np.abs(x_full - x_hov)[sel] * AU_KM,
                lw=1.8, color=C_GREEN)
    axin.set_title(r"Earth's entire effect on $x_{\rm eq}$  [km]",
                   fontsize=7.4, pad=4, color=C_INK)
    axin.tick_params(labelsize=6.4, colors=C_INK, width=0.6)
    axin.grid(True, color=C_GRID, lw=0.5)
    axin.set_axisbelow(True)
    axin.set_facecolor('white')
    for sp in ('top', 'right'):
        axin.spines[sp].set_visible(False)

    # ── (b) standoff in Hill radii ──────────────────────────────────────────
    ax = axes[0, 1]; _style(ax)
    ax.loglog(betas, d_rH, lw=2.0, color=C_BLUE)
    ax.axhline(1.0, color=C_ORANGE, lw=1.4, ls=(0, (5, 3)))
    ax.axhline(3.0**(1/3), color=C_PURPLE, lw=1.4, ls=(0, (1, 2)))
    ax.text(1.4e-5, 1.06, r'Hill sphere,  $r_2 = r_H$',
            fontsize=8.4, color=C_ORANGE)
    ax.text(1.4e-5, 3.0**(1/3) * 1.10,
            r'tidal parity,  $r_2 = 3^{1/3} r_H$',
            fontsize=8.4, color=C_PURPLE)
    for bc, col, lab in ((b_hill, C_ORANGE, rf'$\beta={b_hill:.1e}$'),
                         (b_tide, C_PURPLE, rf'$\beta={b_tide:.4f}$')):
        ax.axvline(bc, color=col, lw=0.9, alpha=0.55)
        ax.annotate(lab, xy=(bc, 12), fontsize=8.2, color=col,
                    rotation=90, va='bottom', ha='right')
    ax.set_xlabel(r'sail lightness number  $\beta$', fontsize=10)
    ax.set_ylabel(r'standoff from Earth  $r_2 / r_H$', fontsize=10)
    ax.set_title('(b)  the point leaves the Earth’s neighbourhood',
                 fontsize=10.5, loc='left', color=C_INK, pad=8)

    # ── (c) saddle strength ─────────────────────────────────────────────────
    ax = axes[1, 0]; _style(ax)
    ax.loglog(betas, s_arr, lw=2.0, color=C_BLUE)
    ax.axhline(1.0, color=C_PURPLE, lw=1.4, ls=(0, (1, 2)))
    ax.axvline(b_tide, color=C_PURPLE, lw=0.9, alpha=0.55)
    ax.annotate(rf'$s=1$ at $\beta={b_tide:.4f}$',
                xy=(b_tide * 1.35, 1.9), fontsize=8.6, color=C_PURPLE)
    s_half = saddle_strength(0.5, mu)
    ax.set_ylim(8e-5, 12.0)
    ax.plot([0.5], [s_half], 'o', ms=7, color=C_ORANGE,
            markeredgecolor='white', markeredgewidth=1.4, zorder=5)
    ax.annotate(rf'$\beta=0.5$:  $s={s_half:.1e}$'
                '\n' r'($\sim\!10^{4}\times$ collapse from $\beta=0$)',
                xy=(0.5, s_half), xytext=(0.021, 1.45e-4),
                fontsize=8.4, color=C_ORANGE, va='top', ha='left',
                arrowprops=dict(arrowstyle='-', color=C_ORANGE, lw=0.9,
                                shrinkA=6, shrinkB=4,
                                connectionstyle='angle,angleA=0,angleB=90,rad=0'))
    ax.set_xlabel(r'sail lightness number  $\beta$', fontsize=10)
    ax.set_ylabel(r'saddle strength  $s = \mu / r_2^{\,3}$', fontsize=10)
    ax.set_title('(c)  the hyperbolicity is entirely of Earth origin',
                 fontsize=10.5, loc='left', color=C_INK, pad=8)

    # ── (d) linear frequencies ──────────────────────────────────────────────
    ax = axes[1, 1]; _style(ax)
    ax.semilogx(betas, omega, lw=2.0, color=C_BLUE,
                label=r'in-plane centre  $\omega$')
    ax.semilogx(betas, nu, lw=2.0, ls=(0, (5, 3)), color=C_GREEN,
                label=r'out-of-plane  $\nu=\sqrt{A}$')
    ax.semilogx(betas, lam_u, lw=2.0, ls=(0, (1, 2)), color=C_ORANGE,
                label=r'saddle exponent  $\lambda_u$')
    ax.axhline(1.0, color=C_MUTED, lw=1.0, ls=':')
    ax.text(1.4e-5, 1.06, 'mean motion = 1', fontsize=8.4, color=C_MUTED)
    ax.axvline(b_tide, color=C_PURPLE, lw=0.9, alpha=0.55)
    ax.set_xlabel(r'sail lightness number  $\beta$', fontsize=10)
    ax.set_ylabel(r'linear rate  [nd]', fontsize=10)
    ax.set_ylim(-0.12, 2.85)
    ax.set_title('(d)  the structure degenerates to Keplerian',
                 fontsize=10.5, loc='left', color=C_INK, pad=8)
    ax.legend(frameon=False, fontsize=8.8, loc='upper right')

    fig.text(0.5, 0.012,
             r'Sun–Earth, $\mu = 3.003\times10^{-6}$, face-on sail '
             r'($\alpha = 0$).  Non-dimensional units; $r_H=(\mu/3)^{1/3}$.',
             ha='center', fontsize=8.4, color=C_MUTED)

    fig.tight_layout(rect=[0, 0.028, 1, 0.955])
    fig.savefig(output, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    if verbose:
        print(f"  ✓ Saved → {output}")

    return dict(betas=betas, x_full=x_full, x_hover=x_hov, d_rH=d_rH,
                s=s_arr, lam_u=lam_u, omega=omega, nu=nu,
                beta_hill=b_hill, beta_tidal=b_tide)


if __name__ == '__main__':
    print("\n── Critical β analysis ───────────────────────────────")
    summary()
    print()
    fig_structure_dissolution()
