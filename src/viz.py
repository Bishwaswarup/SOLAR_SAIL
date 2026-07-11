"""
viz.py — Publication-quality figures for the solar sail CR3BP paper.

All functions accept an optional `ax` argument so they can be embedded into
a multi-panel figure.  Functions that need a full trajectory re-integrate
internally using DOP853 at rtol=1e-9 — same tolerances as the corrector.
No computation beyond integration is done here; callers pass pre-solved data.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401  registers 3-D projection
from scipy.integrate import solve_ivp
from .dynamics import cr3bp_sail_eom

# ── publication style ─────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':    'serif',
    'font.size':      11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 9,
    'figure.dpi':     150,
    'lines.linewidth': 1.4,
})

# Consistent palette (change once here, propagates everywhere)
_C_SUN   = '#FDB813'
_C_EARTH = '#4A90D9'
_C_L1    = '#E74C3C'
_C_HALO  = '#27AE60'
_C_SAIL  = '#8E44AD'
_C_MAN_U = '#E67E22'
_C_MAN_S = '#2980B9'


# ── internal helpers ──────────────────────────────────────────────────────────

def _propagate(state0, T, mu, alpha=0.0, delta=0.0, beta=0.0, n=1000):
    """Integrate one orbit period; return (6, n) array."""
    res = solve_ivp(
        cr3bp_sail_eom, [0.0, T], np.asarray(state0),
        args=(alpha, delta, beta, mu),
        t_eval=np.linspace(0.0, T, n),
        method='DOP853', rtol=1e-9, atol=1e-9,
    )
    return res.y


def _ax2d(ax):
    """Return a fresh 2-D Axes if none supplied."""
    if ax is None:
        _, ax = plt.subplots()
    return ax


def _ax3d(ax):
    """Return a fresh 3-D Axes if none supplied."""
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    return ax


def _pick(traj, proj):
    """Select two coordinate rows from a (6, N) trajectory for 2-D plots."""
    table = {'xz': (0, 2), 'xy': (0, 1), 'yz': (1, 2)}
    i, j = table[proj]
    return traj[i], traj[j]


def _axis_labels(ax, proj):
    labels = {'xz': ('x [non-dim]', 'z [non-dim]'),
              'xy': ('x [non-dim]', 'y [non-dim]'),
              'yz': ('y [non-dim]', 'z [non-dim]')}
    ax.set_xlabel(labels[proj][0])
    ax.set_ylabel(labels[proj][1])


# ── public API ────────────────────────────────────────────────────────────────

def plot_system(mu, ax=None, proj='xz', eq_pos=None):
    """
    Draw Sun (●), Earth (●), and an optional artificial L-point (▲).

    Parameters
    ----------
    eq_pos : array-like (3,) or None — position of the artificial equilibrium.
    proj   : '3d' | 'xz' | 'xy' | 'yz'

    Returns ax.
    """
    sun_x,   earth_x  = -mu,   1.0 - mu
    sun_pos  = np.array([-mu,        0.0, 0.0])
    earth_pos = np.array([1.0 - mu,  0.0, 0.0])

    if proj == '3d':
        ax = _ax3d(ax)
        ax.scatter(*sun_pos,   s=200, c=_C_SUN,   zorder=5, label='Sun')
        ax.scatter(*earth_pos, s=120, c=_C_EARTH, zorder=5, label='Earth')
        if eq_pos is not None:
            ax.scatter(*eq_pos, s=80, c=_C_L1, marker='^', zorder=6,
                       label='L₁ (sail)')
        ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
    else:
        ax = _ax2d(ax)
        ix, iy = {'xz': (0, 2), 'xy': (0, 1), 'yz': (1, 2)}[proj]
        ax.scatter(sun_pos[ix],   sun_pos[iy],   s=200, c=_C_SUN,
                   zorder=5, label='Sun')
        ax.scatter(earth_pos[ix], earth_pos[iy], s=120, c=_C_EARTH,
                   zorder=5, label='Earth')
        if eq_pos is not None:
            ax.scatter(eq_pos[ix], eq_pos[iy], s=80, c=_C_L1,
                       marker='^', zorder=6, label='L₁ (sail)')
        _axis_labels(ax, proj)

    return ax


def plot_orbit(state0, T, mu, alpha=0.0, delta=0.0, beta=0.0,
               ax=None, proj='xz', color=_C_HALO,
               label='Halo orbit', n=1000, **kw):
    """
    Integrate and plot a halo orbit.

    Parameters
    ----------
    proj : '3d' | 'xz' | 'xy' | 'yz'

    Returns ax.
    """
    traj = _propagate(state0, T, mu, alpha, delta, beta, n)

    if proj == '3d':
        ax = _ax3d(ax)
        ax.plot(traj[0], traj[1], traj[2], color=color, label=label, **kw)
        ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
    else:
        ax = _ax2d(ax)
        xi, yi = _pick(traj, proj)
        ax.plot(xi, yi, color=color, label=label, **kw)
        _axis_labels(ax, proj)

    return ax


def plot_manifold_tube(strands, ax=None, proj='xz',
                       color=_C_MAN_U, alpha=0.30,
                       label='Manifold', **kw):
    """
    Plot a manifold tube as a bundle of strands.

    Parameters
    ----------
    strands : list of (6, N) arrays — from compute_manifold().

    Returns ax.
    """
    if proj == '3d':
        ax = _ax3d(ax)
        for k, s in enumerate(strands):
            ax.plot(s[0], s[1], s[2], color=color, alpha=alpha,
                    label=(label if k == 0 else None), **kw)
        ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
    else:
        ax = _ax2d(ax)
        for k, s in enumerate(strands):
            xi, yi = _pick(s, proj)
            ax.plot(xi, yi, color=color, alpha=alpha,
                    label=(label if k == 0 else None), **kw)
        _axis_labels(ax, proj)

    return ax


def plot_poincare(crossings, ax=None, coords=(0, 3),
                  color=_C_MAN_U, label='', marker='o', **kw):
    """
    Scatter-plot Poincaré section crossings.

    Parameters
    ----------
    crossings : list of (6,) arrays — from poincare_section().
    coords    : (i, j) — which state indices to use for x and y axes.
                (0, 3) = (x, vx)   ← default (position–momentum map)
                (0, 2) = (x, z)    ← position-space section
    """
    if not crossings:
        return _ax2d(ax)

    ax  = _ax2d(ax)
    data = np.array(crossings)
    ax.scatter(data[:, coords[0]], data[:, coords[1]],
               color=color, label=label, marker=marker, **kw)

    coord_names = {0: 'x', 1: 'y', 2: 'z', 3: 'vₓ', 4: 'v_y', 5: 'v_z'}
    ax.set_xlabel(f'{coord_names[coords[0]]} [non-dim]')
    ax.set_ylabel(f'{coord_names[coords[1]]} [non-dim]')
    return ax


def plot_reachable_set(cloud, ax=None, color=_C_SAIL, alpha=0.4, **kw):
    """
    3-D scatter of all achievable sail accelerations (the 'control bubble').

    Parameters
    ----------
    cloud : (N, 3) array — from reachable_set().
    """
    ax = _ax3d(ax)
    ax.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2],
               c=color, alpha=alpha, s=5, **kw)
    ax.set_xlabel('aₓ [non-dim]')
    ax.set_ylabel('a_y [non-dim]')
    ax.set_zlabel('a_z [non-dim]')
    ax.set_title('Reachable sail acceleration set')
    return ax


def plot_transfer(strand_u, strand_s,
                  state_u=None, state_s=None,
                  ax=None, proj='xz'):
    """
    Plot a transfer between two manifold strands with a ΔV arrow at the
    Poincaré crossing.

    Parameters
    ----------
    strand_u : (6, N) — unstable manifold strand (departure leg).
    strand_s : (6, N) — stable  manifold strand (arrival leg).
    state_u  : (6,) optional — crossing state on unstable side.
    state_s  : (6,) optional — crossing state on stable side.
    """
    ax = _ax2d(ax)
    xi_idx, yi_idx = {'xz': (0, 2), 'xy': (0, 1), 'yz': (1, 2)}[proj]

    ax.plot(strand_u[xi_idx], strand_u[yi_idx], color=_C_MAN_U, lw=1.8,
            label='Unstable manifold (departure)')
    ax.plot(strand_s[xi_idx], strand_s[yi_idx], color=_C_MAN_S, lw=1.8,
            label='Stable manifold (arrival)')

    if state_u is not None:
        ax.scatter(state_u[xi_idx], state_u[yi_idx],
                   c=_C_MAN_U, s=90, zorder=6, marker='X')
    if state_s is not None:
        ax.scatter(state_s[xi_idx], state_s[yi_idx],
                   c=_C_MAN_S, s=90, zorder=6, marker='X')

    if state_u is not None and state_s is not None:
        dv_mag = np.linalg.norm(state_s[3:] - state_u[3:])
        ax.annotate(
            f'Δv = {dv_mag:.4f} [non-dim]',
            xy=(state_s[xi_idx], state_s[yi_idx]),
            xytext=(state_u[xi_idx], state_u[yi_idx]),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
            fontsize=9,
        )

    _axis_labels(ax, proj)
    ax.legend()
    return ax


def make_paper_figure(
        state0_class, T_class, mu,
        state0_sail,  T_sail,  alpha_sail, delta_sail, beta_sail,
        strands_u, strands_s,
        crossings_u, crossings_s,
        cloud,
        eq_pos_sail=None,
        figsize=(14, 10)):
    """
    Compose the full 4-panel paper figure.

    Panel A (top-left)  : Classical vs sail halo orbits, x-z projection.
    Panel B (top-right) : Manifold tubes, x-z projection.
    Panel C (bottom-left): Poincaré section at y=0, (x, vx) coordinates.
    Panel D (bottom-right): Reachable sail acceleration set (3-D).

    Parameters
    ----------
    state0_class, T_class : classical halo (beta=0) initial state and period.
    state0_sail, T_sail, alpha_sail, delta_sail, beta_sail :
                            sail halo initial state, period, and sail angles.
    strands_u, strands_s  : unstable / stable manifold strand lists.
    crossings_u, crossings_s : Poincaré crossing states (from poincare_section).
    cloud                 : (N, 3) reachable set from reachable_set().
    eq_pos_sail           : (3,) position of artificial equilibrium (optional).

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig = plt.figure(figsize=figsize)
    fig.suptitle(
        'Solar Sail CR3BP — Halo Orbits, Manifold Tubes & Control Authority',
        fontsize=13, fontweight='bold', y=0.99,
    )

    # ── Panel A: halo orbits ──────────────────────────────────────────────────
    ax_a = fig.add_subplot(2, 2, 1)
    plot_system(mu, ax=ax_a, proj='xz', eq_pos=eq_pos_sail)
    plot_orbit(state0_class, T_class, mu,
               ax=ax_a, proj='xz', color=_C_HALO,
               label='Classical L₁ halo (β=0)')
    plot_orbit(state0_sail, T_sail, mu, alpha_sail, delta_sail, beta_sail,
               ax=ax_a, proj='xz', color=_C_SAIL,
               label=f'Sail halo (β={beta_sail})', linestyle='--')
    ax_a.set_title('A   Halo orbits (x–z plane)')
    ax_a.legend()

    # ── Panel B: manifold tubes ───────────────────────────────────────────────
    ax_b = fig.add_subplot(2, 2, 2)
    plot_manifold_tube(strands_u, ax=ax_b, proj='xz',
                       color=_C_MAN_U, label='Unstable manifold (+)')
    plot_manifold_tube(strands_s, ax=ax_b, proj='xz',
                       color=_C_MAN_S, label='Stable manifold (−)')
    ax_b.set_title('B   Manifold tubes (x–z plane)')
    ax_b.legend()

    # ── Panel C: Poincaré section ─────────────────────────────────────────────
    ax_c = fig.add_subplot(2, 2, 3)
    plot_poincare(crossings_u, ax=ax_c, coords=(0, 3),
                  color=_C_MAN_U, label='Unstable', s=25)
    plot_poincare(crossings_s, ax=ax_c, coords=(0, 3),
                  color=_C_MAN_S, label='Stable', marker='s', s=25)
    ax_c.set_title('C   Poincaré section (y = 0, x vs vₓ)')
    ax_c.legend()

    # ── Panel D: reachable set ────────────────────────────────────────────────
    ax_d = fig.add_subplot(2, 2, 4, projection='3d')
    plot_reachable_set(cloud, ax=ax_d, color=_C_SAIL)
    ax_d.set_title('D   Control authority')

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig
