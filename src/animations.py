"""
animations.py — Solar Sail CR3BP animations.

Produces two animations (GIF or MP4):

  1. beta_sweep_animation.mp4 / .gif
     The halo orbit morphs from the classical L₁ family (β=0) to the
     sail-displaced family (β=0.5).  The equilibrium point slides sunward
     with each frame; orbit colour transitions from steel-blue → orange.

  2. manifold_deployment.mp4 / .gif
     Unstable manifold strands launch off the L₁ halo orbit one by one,
     building up the tube that spacecraft can ride away for free.

Usage (from project root)
──────────────────────────
    python src/animations.py              # writes both MP4s to cwd
    python src/animations.py beta         # only beta-sweep
    python src/animations.py manifold     # only manifold
    python src/animations.py both         # explicit both  (default)
    python src/animations.py both gif     # use GIF instead of MP4

Both runners are also importable:
    from src.animations import make_beta_sweep_animation, make_manifold_animation
"""

import sys
import os

# ── path bootstrap (works both as package member and standalone script) ───────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp

from src.equilibria import find_artificial_equilibrium
from src.orbits     import compute_halo_orbit, _richardson_guess
from src.manifolds  import compute_monodromy, compute_manifold
from src.dynamics   import cr3bp_sail_eom

# ── shared CR3BP constant ─────────────────────────────────────────────────────
MU = 3.003e-6


def _save_animation(ani, output, fps, bg_color):
    """Save animation to MP4 (ffmpeg) or GIF (pillow) based on file extension."""
    ext = os.path.splitext(output)[1].lower()
    if ext == '.mp4':
        writer = animation.FFMpegWriter(
            fps=fps, bitrate=1800,
            extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
        ani.save(output, writer=writer,
                 savefig_kwargs={'facecolor': bg_color})
    else:
        ani.save(output, writer='pillow', fps=fps,
                 savefig_kwargs={'facecolor': bg_color})


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _integrate_orbit(state0, period, mu,
                     alpha=0.0, delta=0.0, beta=0.0,
                     n_pts=400):
    """Return (x, y, z) arrays for one full orbit."""
    res = solve_ivp(
        cr3bp_sail_eom,
        [0.0, period],
        state0,
        args=(alpha, delta, beta, mu),
        method='DOP853',
        rtol=1e-10, atol=1e-10,
        dense_output=True,
    )
    t_eval = np.linspace(0, period, n_pts)
    sol = res.sol(t_eval)
    return sol[0], sol[1], sol[2]


def _lissajous_approx(eq_pos, Az, mu, alpha=0.0, delta=0.0, beta=0.0,
                      n_pts=300):
    """
    Fast Lissajous-type approximation used for smooth animation frames
    where convergence of the full corrector would be slow.
    Falls back to Richardson first-order guess — good enough visually.
    """
    try:
        s0, T = compute_halo_orbit(eq_pos, Az, mu, alpha, delta, beta)
        x, y, z = _integrate_orbit(s0, T, mu, alpha, delta, beta, n_pts)
    except Exception:
        # fall back to analytic Richardson curve
        s0, T_half = _richardson_guess(eq_pos, Az, mu)
        T = 2 * T_half
        res = solve_ivp(
            cr3bp_sail_eom,
            [0.0, T],
            s0,
            args=(alpha, delta, beta, mu),
            method='DOP853',
            rtol=1e-9, atol=1e-9,
            t_eval=np.linspace(0, T, n_pts),
        )
        x, y, z = res.y[0], res.y[1], res.y[2]
    return x, y, z


# ─────────────────────────────────────────────────────────────────────────────
# Animation 1 — β-sweep
# ─────────────────────────────────────────────────────────────────────────────

def make_beta_sweep_animation(
        output='beta_sweep_animation.mp4',
        betas=None,
        Az=0.003,
        fps=8,
        verbose=True):
    """
    Animate the halo orbit family from β=0 to β=0.5.

    Parameters
    ----------
    output : str      Output filename.  Extension controls format:
                        .mp4 → H.264 via ffmpeg (recommended)
                        .gif → animated GIF via pillow
    betas  : array    β values to sweep (default: 11 steps 0 → 0.5).
    Az     : float    Out-of-plane amplitude (non-dim).
    fps    : int      Frames per second.
    verbose: bool     Print progress.

    Returns
    -------
    Path to written file.
    """
    if betas is None:
        betas = np.linspace(0.0, 0.5, 11)

    # ── precompute all orbits ─────────────────────────────────────────────────
    x_eq_list, orbits_xz, orbits_xy = [], [], []

    # classical L₁ as anchor
    eq0 = find_artificial_equilibrium(0.0, 0.0, 0.0, MU, [0.99, 0.0, 0.0])
    x_guess = eq0[0]

    for b in betas:
        if verbose:
            print(f"  β={b:.2f}  computing orbit …", end=' ')
        try:
            eq = find_artificial_equilibrium(0.0, 0.0, b, MU,
                                             [x_guess - 0.02, 0.0, 0.0])
            x_guess = eq[0]   # warm start for next β
            x_eq_list.append(eq[0])
            x, y, z = _lissajous_approx(eq, Az, MU, 0.0, 0.0, b)
            orbits_xz.append((x, z))
            orbits_xy.append((x, y))
            if verbose:
                print(f"done  x_eq={eq[0]:.4f}")
        except Exception as e:
            if verbose:
                print(f"SKIP ({e})")
            if orbits_xz:
                orbits_xz.append(orbits_xz[-1])
                orbits_xy.append(orbits_xy[-1])
                x_eq_list.append(x_eq_list[-1])

    n_frames = len(betas)
    cmap = plt.cm.RdYlBu_r
    colors = [cmap(i / (n_frames - 1)) for i in range(n_frames)]

    # ── figure layout ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5),
                             facecolor='#0d0d1a')
    for ax in axes:
        ax.set_facecolor('#0d0d1a')
        for sp in ax.spines.values():
            sp.set_color('#444466')
        ax.tick_params(colors='#aaaacc', labelsize=8)
        ax.xaxis.label.set_color('#aaaacc')
        ax.yaxis.label.set_color('#aaaacc')

    ax_xz, ax_xy = axes

    # fixed axes limits — cover all orbits
    all_x = np.concatenate([o[0] for o in orbits_xz])
    all_z = np.concatenate([o[1] for o in orbits_xz])
    all_y = np.concatenate([o[1] for o in orbits_xy])

    pad_x = 0.002; pad_z = max(Az * 1.4, 0.001)
    xlim = (all_x.min() - pad_x, all_x.max() + pad_x)
    zlim = (-pad_z, pad_z)
    ylim = (all_y.min() - 0.001, all_y.max() + 0.001)

    ax_xz.set_xlim(xlim); ax_xz.set_ylim(zlim)
    ax_xy.set_xlim(xlim); ax_xy.set_ylim(ylim)
    ax_xz.set_xlabel('x  [non-dim]'); ax_xz.set_ylabel('z  [non-dim]')
    ax_xy.set_xlabel('x  [non-dim]'); ax_xy.set_ylabel('y  [non-dim]')
    ax_xz.set_title('x-z  plane', color='#ccccee', fontsize=10)
    ax_xy.set_title('x-y  plane', color='#ccccee', fontsize=10)

    # Sun marker
    for ax in axes:
        ax.axvline(0.0, color='#ffcc44', lw=0.5, alpha=0.3)
        ax.scatter([-(MU)], [0], marker='*', s=120,
                   color='#ffdd55', zorder=5, label='Sun')
        ax.scatter([1 - MU], [0], marker='o', s=40,
                   color='#4488ff', zorder=5, label='Earth')

    # artist handles
    orbit_lines_xz = []
    orbit_lines_xy = []
    for k in range(n_frames):
        c = colors[k]
        lw = 0.6 + 0.8 * k / (n_frames - 1)
        alpha = 0.25 + 0.75 * (k == 0)
        lxz, = ax_xz.plot([], [], color=c, lw=lw, alpha=alpha, zorder=3)
        lxy, = ax_xy.plot([], [], color=c, lw=lw, alpha=alpha, zorder=3)
        orbit_lines_xz.append(lxz)
        orbit_lines_xy.append(lxy)

    eq_dot_xz, = ax_xz.plot([], [], 'o', color='white', ms=5, zorder=6)
    eq_dot_xy, = ax_xy.plot([], [], 'o', color='white', ms=5, zorder=6)

    beta_text = ax_xz.text(
        0.03, 0.93, '', transform=ax_xz.transAxes,
        color='white', fontsize=11, fontweight='bold',
        va='top', ha='left',
        bbox=dict(boxstyle='round,pad=0.3', fc='#222244', ec='none', alpha=0.8))

    fig.suptitle('Solar-sail halo orbit family  (Sun-Earth L₁,  Az = 0.003)',
                 color='#eeeeff', fontsize=12, y=1.01)

    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 0.5))
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.set_label('β  (sail lightness)', color='#aaaacc', fontsize=9)
    cb.ax.yaxis.set_tick_params(color='#aaaacc')
    plt.setp(cb.ax.yaxis.get_ticklabels(), color='#aaaacc')

    plt.tight_layout(rect=[0, 0, 0.91, 1])

    # ── animation update ──────────────────────────────────────────────────────
    def update(frame):
        # show all orbits up to current frame (ghost trail)
        for k in range(n_frames):
            if k <= frame:
                orbit_lines_xz[k].set_data(*orbits_xz[k])
                orbit_lines_xy[k].set_data(*orbits_xy[k])
                a = 0.15 + 0.85 * (k == frame)
                orbit_lines_xz[k].set_alpha(a)
                orbit_lines_xy[k].set_alpha(a)
            else:
                orbit_lines_xz[k].set_data([], [])
                orbit_lines_xy[k].set_data([], [])

        # equilibrium dot
        xe = x_eq_list[frame]
        eq_dot_xz.set_data([xe], [0])
        eq_dot_xy.set_data([xe], [0])

        beta_text.set_text(f'β = {betas[frame]:.2f}')
        return (orbit_lines_xz + orbit_lines_xy +
                [eq_dot_xz, eq_dot_xy, beta_text])

    ani = animation.FuncAnimation(
        fig, update, frames=n_frames,
        interval=1000 // fps, blit=True)

    _save_animation(ani, output, fps, '#0d0d1a')
    plt.close(fig)
    if verbose:
        print(f"\n  ✓  Saved → {output}")
    return output


# ─────────────────────────────────────────────────────────────────────────────
# Animation 2 — manifold deployment
# ─────────────────────────────────────────────────────────────────────────────

def make_manifold_animation(
        output='manifold_deployment.mp4',
        n_strands=24,
        t_max=2.5 * np.pi,
        eps=1e-6,
        fps=8,
        beta=0.0,
        verbose=True):
    """
    Animate unstable (+ and −) manifold strands launching one by one
    off the classical L₁ halo orbit (or sail orbit if beta > 0).

    Parameters
    ----------
    output    : str    Output filename.  Extension controls format:
                         .mp4 → H.264 via ffmpeg (recommended)
                         .gif → animated GIF via pillow
    n_strands : int    Number of strands per branch (total = 2 × n_strands).
    t_max     : float  Max propagation time per strand [non-dim].
    eps       : float  Perturbation size along eigenvector.
    fps       : int    Frames per second.
    beta      : float  Sail lightness (0 = classical).
    verbose   : bool   Print progress.

    Returns
    -------
    Path to written file.
    """
    alpha, delta = 0.0, 0.0

    # ── step 1: compute halo orbit ────────────────────────────────────────────
    if verbose:
        print("  Computing halo orbit …", end=' ', flush=True)
    eq = find_artificial_equilibrium(alpha, delta, beta, MU,
                                     [0.99 - beta * 0.2, 0.0, 0.0])
    state0, T = compute_halo_orbit(eq, 0.003, MU, alpha, delta, beta)
    if verbose:
        print(f"done  T={T:.4f}")

    # ── step 2: compute reference orbit trajectory ────────────────────────────
    x_orb, y_orb, z_orb = _integrate_orbit(
        state0, T, MU, alpha, delta, beta, n_pts=300)

    # ── step 3: compute strands ───────────────────────────────────────────────
    if verbose:
        print(f"  Computing {n_strands} strands per branch …", flush=True)

    strands_p = compute_manifold(
        state0, T, MU, alpha, delta, beta,
        direction='unstable', branch='+',
        n_points=n_strands, eps=eps, t_max=t_max)

    strands_m = compute_manifold(
        state0, T, MU, alpha, delta, beta,
        direction='unstable', branch='-',
        n_points=n_strands, eps=eps, t_max=t_max)

    if verbose:
        print(f"  Strands: +branch={len(strands_p)}, -branch={len(strands_m)}")

    # strands_p[k] shape: (6, n_steps) — rows are state components
    def _strand_xy(s):
        return s[0], s[1]

    def _strand_xz(s):
        return s[0], s[2]

    # ── figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5),
                             facecolor='#0a0a18')
    for ax in axes:
        ax.set_facecolor('#0a0a18')
        for sp in ax.spines.values():
            sp.set_color('#333355')
        ax.tick_params(colors='#aaaacc', labelsize=8)
        ax.xaxis.label.set_color('#aaaacc')
        ax.yaxis.label.set_color('#aaaacc')

    ax_xy, ax_xz = axes

    # axis limits from all strand data
    all_xs = np.concatenate([s[0] for s in strands_p + strands_m] + [x_orb])
    all_ys = np.concatenate([s[1] for s in strands_p + strands_m] + [y_orb])
    all_zs = np.concatenate([s[2] for s in strands_p + strands_m] + [z_orb])

    xpad, ypad, zpad = 0.02, 0.005, 0.005
    ax_xy.set_xlim(all_xs.min() - xpad, all_xs.max() + xpad)
    ax_xy.set_ylim(all_ys.min() - ypad, all_ys.max() + ypad)
    ax_xz.set_xlim(all_xs.min() - xpad, all_xs.max() + xpad)
    ax_xz.set_ylim(all_zs.min() - zpad, all_zs.max() + zpad)

    ax_xy.set_xlabel('x  [non-dim]'); ax_xy.set_ylabel('y  [non-dim]')
    ax_xz.set_xlabel('x  [non-dim]'); ax_xz.set_ylabel('z  [non-dim]')
    ax_xy.set_title('x-y  plane', color='#ccccee', fontsize=10)
    ax_xz.set_title('x-z  plane', color='#ccccee', fontsize=10)

    # fixed markers
    for ax in (ax_xy, ax_xz):
        ax.scatter([-(MU)], [0], marker='*', s=140,
                   color='#ffdd55', zorder=8, label='Sun')
        ax.scatter([1 - MU], [0], marker='o', s=45,
                   color='#4488ff', zorder=8, label='Earth')
        ax.scatter([eq[0]], [0], marker='^', s=60,
                   color='#ff8844', zorder=7, label='L₁')

    # halo orbit (static)
    ax_xy.plot(x_orb, y_orb, color='#88aaff', lw=1.2, zorder=4, alpha=0.7)
    ax_xz.plot(x_orb, z_orb, color='#88aaff', lw=1.2, zorder=4, alpha=0.7)

    # strand artist containers
    strand_lines_xy_p = []
    strand_lines_xz_p = []
    strand_lines_xy_m = []
    strand_lines_xz_m = []

    for _ in strands_p:
        lxy, = ax_xy.plot([], [], color='#ff7733', lw=0.7, alpha=0.7, zorder=3)
        lxz, = ax_xz.plot([], [], color='#ff7733', lw=0.7, alpha=0.7, zorder=3)
        strand_lines_xy_p.append(lxy)
        strand_lines_xz_p.append(lxz)

    for _ in strands_m:
        lxy, = ax_xy.plot([], [], color='#33ccff', lw=0.7, alpha=0.7, zorder=3)
        lxz, = ax_xz.plot([], [], color='#33ccff', lw=0.7, alpha=0.7, zorder=3)
        strand_lines_xy_m.append(lxy)
        strand_lines_xz_m.append(lxz)

    count_text = ax_xy.text(
        0.03, 0.93, '', transform=ax_xy.transAxes,
        color='white', fontsize=10, va='top',
        bbox=dict(boxstyle='round,pad=0.3', fc='#111133', ec='none', alpha=0.8))

    fig.suptitle(
        f'Unstable manifold deployment  (β={beta:.2f})\n'
        '■ orange: +branch   ■ cyan: −branch',
        color='#eeeeff', fontsize=11, y=1.01)

    for ax in (ax_xy, ax_xz):
        ax.legend(loc='lower right', fontsize=7,
                  facecolor='#111133', edgecolor='none',
                  labelcolor='#aaaacc')

    plt.tight_layout()

    n_frames = max(len(strands_p), len(strands_m))

    def update(frame):
        artists = []
        for k, (lxy, lxz) in enumerate(
                zip(strand_lines_xy_p, strand_lines_xz_p)):
            if k <= frame and k < len(strands_p):
                lxy.set_data(*_strand_xy(strands_p[k]))
                lxz.set_data(*_strand_xz(strands_p[k]))
            artists += [lxy, lxz]

        for k, (lxy, lxz) in enumerate(
                zip(strand_lines_xy_m, strand_lines_xz_m)):
            if k <= frame and k < len(strands_m):
                lxy.set_data(*_strand_xy(strands_m[k]))
                lxz.set_data(*_strand_xz(strands_m[k]))
            artists += [lxy, lxz]

        n_shown = min(frame + 1, n_strands)
        count_text.set_text(
            f'Strands: {n_shown} / {n_strands}  per branch')
        artists.append(count_text)
        return artists

    ani = animation.FuncAnimation(
        fig, update, frames=n_frames,
        interval=1000 // fps, blit=True)

    _save_animation(ani, output, fps, '#0a0a18')
    plt.close(fig)
    if verbose:
        print(f"  ✓  Saved → {output}")
    return output


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # Only accept recognised positional args; ignore shell comment fragments (#…)
    _args  = [a for a in sys.argv[1:] if not a.startswith('#')]
    which  = _args[0] if len(_args) > 0 else 'both'
    _fmt   = _args[1] if len(_args) > 1 else 'mp4'
    fmt    = _fmt if _fmt in ('mp4', 'gif') else 'mp4'   # safe default
    ext    = f'.{fmt}'

    if which in ('beta', 'both'):
        print("\n── Animation 1: β-sweep ──────────────────────────")
        make_beta_sweep_animation(output=f'beta_sweep_animation{ext}')

    if which in ('manifold', 'both'):
        print("\n── Animation 2: manifold deployment ─────────────")
        make_manifold_animation(output=f'manifold_deployment{ext}')

    print("\nDone.")
