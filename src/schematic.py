"""
schematic.py — Figure 1, the geometry of the face-on solar-sail CR3BP.

Two panels:
  (a) the rotating frame: primaries, the classical L1, the sail equilibrium
      migrating sunward with beta, and the standoff r2 against the Hill radius
      and the tidal-parity radius mu^(1/3) = 3^(1/3) r_H.
  (b) the sail attitude convention: the orthonormal frame {r_hat, p_hat, q_hat},
      the cone angle alpha from the Sun line to the normal, the clock angle
      delta about the Sun line, and the ideal-reflector force along n_hat with
      magnitude proportional to cos^2(alpha).

Panel (a) is drawn with the Earth-Sun separation to scale but the standoff
distances exaggerated, because r2/1 ~ 1e-2 for Sun-Earth and an honest scale
would collapse every feature onto the Earth.  The exaggeration factor is stated
in the caption and on the axis.
"""

from __future__ import annotations

import numpy as np

from src.critical_beta import (MU_SE, critical_beta_tidal_exact, equilibrium,
                               hill_radius)


def fig_schematic(output: str = 'fig1_schematic.png', mu: float = MU_SE,
                  zoom: float = 18.0, verbose: bool = True) -> dict:
    from src.paperstyle import use, panel_label
    use()
    import matplotlib.pyplot as plt
    from matplotlib.patches import Arc, FancyArrowPatch

    b_tide = critical_beta_tidal_exact(mu)
    rH = hill_radius(mu)
    x_cls = equilibrium(0.0, mu)
    x_tide = equilibrium(b_tide, mu)
    r2_cls = (1 - mu) - x_cls
    r2_tide = (1 - mu) - x_tide

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1))

    # ── (a) rotating frame, drawn as the 1-D geometry it is ───────────────
    ax = axes[0]
    xe = 1.0

    def X(x_nd):
        """Exaggerate the Earth neighbourhood by `zoom`."""
        return xe + (x_nd - xe) * zoom

    ax.axhline(0, color='0.6', lw=0.8, zorder=0)
    ax.plot([0.0], [0.0], 'ko', ms=9.0, zorder=3)
    ax.annotate('Sun', xy=(0.0, -0.10), fontsize=7.8, ha='center', va='top')
    ax.plot([xe], [0.0], 'ko', ms=4.5, zorder=3)
    ax.annotate('Earth', xy=(xe, -0.10), fontsize=7.8, ha='center', va='top')

    # equilibria on the axis
    ax.plot([X(x_cls)], [0.0], 'ks', ms=5.0, mfc='white', mew=1.0, zorder=4)
    ax.plot([X(x_tide)], [0.0], 'kD', ms=4.6, zorder=4)
    ax.annotate(r'$L_1$ ($\beta=0$)', xy=(X(x_cls) + 0.03, 0.155),
                fontsize=7.2, ha='left', va='bottom')
    ax.plot([X(x_cls), X(x_cls)], [0.0, 0.150], color='0.55', lw=0.6, zorder=1)
    ax.annotate('tidal parity' + '\n' + rf'$\beta={b_tide:.4f}$',
                xy=(X(x_tide) - 0.03, 0.29), fontsize=7.2, ha='right',
                va='bottom')
    ax.plot([X(x_tide), X(x_tide)], [0.0, 0.285], color='0.55', lw=0.6,
            zorder=1)

    # standoff spans, measured sunward from the Earth
    for r, lab, yv in ((rH, r'$r_H=(\mu/3)^{1/3}$', -0.34),
                       (mu**(1 / 3), r'$\mu^{1/3}=3^{1/3}r_H$', -0.56)):
        x0 = X(xe - r)
        ax.annotate('', xy=(x0, yv), xytext=(xe, yv),
                    arrowprops=dict(arrowstyle='<->', color='0.35', lw=0.8,
                                    shrinkA=0, shrinkB=0))
        for xv in (x0, xe):
            ax.plot([xv, xv], [yv, 0.0], color='0.80', lw=0.5, zorder=0)
        ax.annotate(lab, xy=((x0 + xe) / 2, yv - 0.045), fontsize=7.0,
                    ha='center', va='top', color='0.25')

    ax.annotate('', xy=(X(x_tide) - 0.015, 0.065),
                xytext=(X(x_cls) + 0.015, 0.065),
                arrowprops=dict(arrowstyle='-|>', color='black', lw=0.9))
    ax.annotate(r'increasing $\beta$', xy=(X(x_cls) + 0.05, 0.050),
                fontsize=7.0, ha='left', va='top')

    ax.set_xlim(-0.14, 1.16)
    ax.set_ylim(-0.78, 0.46)
    ax.set_xlabel(r'$x$  [nd]  ' + rf'(Earth neighbourhood $\times${zoom:g})',
                  labelpad=2)
    ax.set_yticks([])
    ax.set_xticks([0.0, 0.5, 1.0])
    for sp in ('left', 'right', 'top'):
        ax.spines[sp].set_visible(False)
    panel_label(ax, '(a)', dx=-0.05, dy=1.02)

    # ── (b) sail attitude ─────────────────────────────────────────────────
    ax = axes[1]
    O = np.array([0.0, 0.0])
    r_hat = np.array([1.0, 0.0])
    q_hat = np.array([0.0, 1.0])
    alpha = np.radians(35.0)
    n_hat = np.array([np.cos(alpha), np.sin(alpha)])

    def arrow(vec, lab, lw=1.0, dash=None, lab_off=(0.05, 0.03), col='black'):
        ax.add_patch(FancyArrowPatch(
            O, vec, arrowstyle='-|>', mutation_scale=9, lw=lw, color=col,
            linestyle='-' if dash is None else (0, tuple(dash))))
        ax.annotate(lab, xy=(vec[0] + lab_off[0], vec[1] + lab_off[1]),
                    fontsize=7.8, color=col)

    ax.plot([-0.34, 1.16], [0, 0], color='0.72', lw=0.6, zorder=0)
    ax.plot([-0.32], [0.0], 'ko', ms=6.5)
    ax.annotate('to Sun', xy=(-0.32, -0.12), fontsize=7.2, ha='center',
                va='top')

    arrow(r_hat * 0.95, r'$\hat{\mathbf{r}}$', lab_off=(0.04, -0.05))
    arrow(q_hat * 0.60, r'$\hat{\mathbf{q}}$', lab_off=(-0.13, 0.02))
    arrow(n_hat * 0.80, r'$\hat{\mathbf{n}}$', lw=1.4, lab_off=(0.04, 0.02))

    t = np.array([-n_hat[1], n_hat[0]])
    m0, m1 = n_hat * 0.80 - t * 0.28, n_hat * 0.80 + t * 0.28
    ax.plot([m0[0], m1[0]], [m0[1], m1[1]], 'k-', lw=3.0,
            solid_capstyle='round')
    ax.annotate('sail', xy=(m1[0] + 0.05, m1[1] - 0.04), fontsize=7.4)

    ax.add_patch(Arc(O, 0.58, 0.58, theta1=0.0, theta2=35.0, lw=0.8,
                     color='black'))
    ax.annotate(r'$\alpha$', xy=(0.335, 0.078), fontsize=8.4)

    arrow(np.array([0.26, -0.40]), r'$\hat{\mathbf{p}}$', dash=[3, 2],
          lab_off=(0.04, -0.07), col='0.35')
    ax.annotate(r'$\delta$ about $\hat{\mathbf{r}}$, from '
                r'$\hat{\mathbf{p}}$ to $\hat{\mathbf{q}}$',
                xy=(0.46, -0.42), fontsize=7.0, color='0.30')

    ax.annotate(r'$\mathbf{a}_{\rm sail}=\beta\,\dfrac{1-\mu}{r_1^{2}}\,'
                r'\cos^{2}\!\alpha\;\hat{\mathbf{n}}$',
                xy=(-0.34, 1.02), fontsize=8.6)
    ax.annotate(r'face-on ($\alpha=0$): $\hat{\mathbf{n}}=\hat{\mathbf{r}}$,'
                ' force radial,' '\n' 'hence conservative',
                xy=(-0.34, 0.80), fontsize=7.0, color='0.30')

    ax.set_xlim(-0.38, 1.22)
    ax.set_ylim(-0.62, 1.24)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    panel_label(ax, '(b)', dx=-0.02, dy=1.02)

    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)
    if verbose:
        print(f"  Saved -> {output}")
    return dict(beta_tidal=b_tide, r_hill=rH, x_classical=x_cls,
                x_tidal=x_tide, r2_classical=r2_cls, r2_tidal=r2_tide)


if __name__ == '__main__':
    fig_schematic()
