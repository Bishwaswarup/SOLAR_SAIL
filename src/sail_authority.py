"""
sail_authority.py — what a solar sail can actually do as a control system.

Bug A3, stated precisely
────────────────────────
stationkeeping.py builds its LQR on

    B_CTRL = 6x3,  identity in the acceleration rows

which declares an unconstrained three-axis thruster: any direction, any sign,
magnitude decoupled from direction, saturated only in norm.  A solar sail is
none of those things.  It has TWO inputs (the cone angle alpha and the clock
angle delta), it can only push into the hemisphere away from the Sun, and its
direction and magnitude are rigidly coupled through

    a_sail = beta (1-mu)/r1^2 cos^2(alpha) n_hat(alpha, delta)              (1)

so steering off-axis to rotate the thrust simultaneously weakens it by cos^2.
Note also that alpha and delta never enter stationkeeping.py's controlled EOM:
cr3bp_sail_eom is called with the FIXED nominal angles and the LQR correction is
added on top as a separate acceleration.  The sail is a constant background
force and the controller is a fictitious thruster beside it.  Nothing in that
loop is sail attitude control, despite the figure title.

The true control Jacobian
─────────────────────────
Linearising (1) in the two angles about a nominal (alpha0, delta0) gives a 6x2

    B_sail = [ 0_3x2 ; da/dalpha  da/ddelta ]                              (2)

with, at alpha0 = 0 exactly,

    da/dalpha = beta (1-mu)/r1^2 ( cos(delta0) p_hat + sin(delta0) q_hat )
    da/ddelta = 0                                                          (3)

Two consequences follow from (3), both verified numerically in report():

  * The clock angle has NO first-order authority when the sail is face-on.
    delta only rotates a normal that is already parallel to r_hat, so at
    alpha0 = 0 the sail is a ONE-input system, and the single direction it can
    push is fixed by the constant delta0.

  * That direction is purely TRANSVERSE: the radial component of da/dalpha is
    zero at alpha0 = 0, because a . r_hat = beta(1-mu)/r1^2 cos^3(alpha) and
    d/dalpha cos^3 = -3 cos^2 sin -> 0.

CONTROLLABILITY — the result that matters
─────────────────────────────────────────
Rank of B is not the test; Coriolis coupling can carry a transverse input into
the radial direction.  Applying the Kalman rank condition to the linearised
CR3BP at the beta = 0.05 equilibrium (A = 1.409194):

    B_CTRL (the 6x3 thruster)                    rank 6/6   controllable
    sail, alpha0 = 0,  delta0 = 0                rank 4/6   UNCONTROLLABLE
    sail, alpha0 = 0,  delta0 = 90 deg           rank 2/6   UNCONTROLLABLE
    sail, alpha0 = 0.5 deg                       rank 6/6   controllable
    sail, alpha0 = 2, 5, 15, 35, 45 deg          rank 6/6   controllable

The in-plane 4-state block IS controllable by a single transverse input
(rank 4/4) — Coriolis does the work, so the absence of radial authority is not
by itself fatal.  What fails is the out-of-plane mode: z decouples at linear
order (z_ddot = -A z), and because da/ddelta = 0 the sail cannot redirect thrust
into z without changing alpha0.  So a face-on sail reaches the in-plane pair OR
the vertical mode, never both.

    alpha0 = 0 is a singular nominal for sail station-keeping.

Any non-zero cone angle restores full controllability — 0.5 deg is enough for
rank 6, though the second singular value there is 4.5e-4 against 5.2e-2 for the
first, a 115:1 conditioning penalty, so the authority is nominally present and
practically negligible until alpha0 reaches a few degrees.

What this means for fig5
────────────────────────
The minimum-beta-for-LQR-stability sweep, the control-authority map and the
closed-loop simulation all characterise a spacecraft with an idealised
omnidirectional bounded-magnitude thruster.  They are not statements about a
solar sail, and at the face-on nominal they describe a system the sail provably
cannot realise.  Either rebuild them on B_sail with alpha0 != 0 -- which makes
the nominal cone angle a design variable and is a genuine result -- or drop them.
This module supplies the B they would need; it deliberately does not rewrite
stationkeeping.py, because that is a scientific choice, not a bug fix.

Face-on results elsewhere are untouched.  The dissolution analysis, the closed
form and the halo families are all alpha = 0 open-loop dynamics with no control
in them.
"""

from __future__ import annotations

import numpy as np

from src.dynamics import sail_acceleration, sail_frame

MU_SE = 3.003e-6


def sail_control_jacobian(pos, alpha0: float, delta0: float,
                          beta: float, mu: float = MU_SE,
                          h: float = 1e-6) -> np.ndarray:
    """
    The true 6x2 control Jacobian of a solar sail, eq. (2).

    Columns are d(acceleration)/d(alpha) and d(acceleration)/d(delta),
    central-differenced about (alpha0, delta0).  Rows 0-2 are zero: the sail
    acts on velocity, not position.
    """
    state = np.array([pos[0], pos[1], pos[2], 0.0, 0.0, 0.0], dtype=float)
    B = np.zeros((6, 2))
    for j, (da, dd) in enumerate([(h, 0.0), (0.0, h)]):
        plus = sail_acceleration(state, alpha0 + da, delta0 + dd, beta, mu)
        minus = sail_acceleration(state, alpha0 - da, delta0 - dd, beta, mu)
        B[3:, j] = (np.asarray(plus) - np.asarray(minus)) / (2.0 * h)
    return B


def linearised_cr3bp(A_param: float) -> np.ndarray:
    """
    6x6 state matrix of the CR3BP linearised at a collinear equilibrium.

    U_xx = 1 + 2A,  U_yy = 1 - A,  U_zz = -A, with the Coriolis coupling.
    """
    M = np.zeros((6, 6))
    M[0, 3] = M[1, 4] = M[2, 5] = 1.0
    M[3, 0] = 1.0 + 2.0 * A_param
    M[3, 4] = 2.0
    M[4, 1] = 1.0 - A_param
    M[4, 3] = -2.0
    M[5, 2] = -A_param
    return M


def controllability_rank(A_mat: np.ndarray, B: np.ndarray,
                         tol: float = 1e-9) -> int:
    """Kalman rank of [B, AB, A^2 B, ...].  Full rank = 6 means controllable."""
    n = A_mat.shape[0]
    blocks = [B]
    for i in range(1, n):
        blocks.append(np.linalg.matrix_power(A_mat, i) @ B)
    return int(np.linalg.matrix_rank(np.hstack(blocks), tol=tol))


def thruster_jacobian() -> np.ndarray:
    """The 6x3 B that stationkeeping.py actually uses.  NOT a sail."""
    B = np.zeros((6, 3))
    B[3, 0] = B[4, 1] = B[5, 2] = 1.0
    return B


def report(beta: float = 0.05, A_param: float = 1.409194,
           x_eq: float = 0.9804, mu: float = MU_SE,
           verbose: bool = True) -> dict:
    """Reproduce the controllability table in the module docstring."""
    A_mat = linearised_cr3bp(A_param)
    pos = (x_eq, 0.0, 0.0)
    rows = []

    r_th = controllability_rank(A_mat, thruster_jacobian())
    if verbose:
        print(f"  linearised at beta = {beta}, A = {A_param}, "
              f"x_eq = {x_eq}")
        print()
        print(f"  {'B model':<34}{'sv1':>11}{'sv2':>11}{'ctrb':>8}  verdict")
        print(f"  {'B_CTRL: ideal 6x3 thruster':<34}{'-':>11}{'-':>11}"
              f"{f'{r_th}/6':>8}  "
              f"{'controllable' if r_th == 6 else 'UNCONTROLLABLE'}")

    for a0d, d0d in [(0.0, 0.0), (0.0, 90.0), (0.5, 0.0), (2.0, 0.0),
                     (5.0, 0.0), (15.0, 0.0), (35.0, 0.0), (45.0, 0.0)]:
        B = sail_control_jacobian(pos, np.radians(a0d), np.radians(d0d),
                                  beta, mu)
        sv = np.linalg.svd(B[3:, :], compute_uv=False)
        r = controllability_rank(A_mat, B)
        rows.append(dict(alpha0_deg=a0d, delta0_deg=d0d, sv1=float(sv[0]),
                         sv2=float(sv[1]), rank=r, controllable=(r == 6)))
        if verbose:
            lbl = f"sail: alpha0={a0d:g} deg, delta0={d0d:g}"
            print(f"  {lbl:<34}{sv[0]:>11.3e}{sv[1]:>11.3e}{f'{r}/6':>8}  "
                  f"{'controllable' if r == 6 else 'UNCONTROLLABLE'}")

    # in-plane block alone, single transverse input
    A_in = np.array([[0, 0, 1, 0], [0, 0, 0, 1],
                     [1 + 2 * A_param, 0, 0, 2],
                     [0, 1 - A_param, -2, 0]], dtype=float)
    B_in = np.zeros((4, 1))
    B_in[3, 0] = 1.0
    blocks = [B_in]
    for i in range(1, 4):
        blocks.append(np.linalg.matrix_power(A_in, i) @ B_in)
    r_in = int(np.linalg.matrix_rank(np.hstack(blocks), tol=1e-9))

    if verbose:
        print()
        print(f"  in-plane 4-state block, single transverse input: "
              f"rank {r_in}/4 -> "
              f"{'controllable (Coriolis carries it)' if r_in == 4 else 'not'}")
        print(f"  out-of-plane mode at alpha0 = 0: unreachable, because "
              f"da/ddelta = 0")
        print()
        print("  => alpha0 = 0 is a SINGULAR nominal for sail station-keeping.")

    return dict(thruster_rank=r_th, sail=rows, inplane_rank=r_in)


if __name__ == '__main__':
    print("\n== Sail control authority (bug A3) ==\n")
    report()
    print()


# ── sweeps and the figure ─────────────────────────────────────────────────────

# Exact optimum of the clock-angle authority.  |da/ddelta| ~ cos^2(a) sin(a),
# maximised where d/da[cos^2 a sin a] = 0 -> cos^2 a = 2 sin^2 a -> tan a = 1/sqrt2.
ALPHA_STAR = np.arctan(1.0 / np.sqrt(2.0))          # 35.264 deg
COS2_ALPHA_STAR = 2.0 / 3.0                          # exact
COND_AT_STAR = 1.0 / 3.0                             # sv2/sv1, exact

# This angle is the classical optimal sail cone angle for maximum transverse
# thrust (McInnes 1999, sec. 2.6); recovering it from the control Jacobian is a
# benchmark check on the Jacobian, not a new result.  What IS new here is that
# the same angle maximises out-of-plane CONTROL AUTHORITY, with sv2/sv1 = 1/3
# exactly, and that the authority vanishes identically at alpha0 = 0.


def authority_sweep(beta: float, mu: float = MU_SE, n: int = 400,
                    alpha_max_deg: float = 89.0) -> dict:
    """Singular values of B_sail against nominal cone angle, at fixed beta."""
    from src.critical_beta import equilibrium
    x_eq = equilibrium(beta, mu)
    al = np.radians(np.linspace(1e-6, alpha_max_deg, n))
    sv = np.array([
        np.linalg.svd(sail_control_jacobian((x_eq, 0.0, 0.0), a, 0.0, beta, mu)[3:, :],
                      compute_uv=False) for a in al])
    return dict(alpha=al, alpha_deg=np.degrees(al), sv1=sv[:, 0], sv2=sv[:, 1],
                cond=sv[:, 1] / sv[:, 0], beta=beta, x_eq=x_eq)


def authority_vs_beta(betas=None, mu: float = MU_SE) -> dict:
    """Out-of-plane authority at the optimal cone angle, across the sail band."""
    from src.critical_beta import equilibrium
    if betas is None:
        betas = np.logspace(np.log10(0.001), np.log10(0.05), 60)
    betas = np.asarray(betas, dtype=float)
    sv2 = []
    for b in betas:
        x_eq = equilibrium(b, mu)
        B = sail_control_jacobian((x_eq, 0.0, 0.0), ALPHA_STAR, 0.0, b, mu)
        sv2.append(float(np.linalg.svd(B[3:, :], compute_uv=False)[1]))
    return dict(betas=betas, sv2=np.array(sv2))


def fig_control_authority(output: str = 'fig5_control_authority.png',
                          beta: float = 0.05, mu: float = MU_SE,
                          verbose: bool = True) -> dict:
    """
    Figure 5 — what a solar sail can actually control at an artificial
    equilibrium, built on the true 6x2 sail Jacobian.

    Replaces the earlier LQR suite, which was computed on a 6x3 unconstrained
    thruster (bug A3) and therefore described a spacecraft the sail cannot be.
    """
    from src.paperstyle import use, panel_label, thin_guide
    from src.critical_beta import critical_beta_tidal_exact
    use()
    import matplotlib.pyplot as plt

    if verbose:
        print("  sweeping cone angle ...")
    sw = authority_sweep(beta, mu)
    vb = authority_vs_beta(mu=mu)
    a_star_deg = np.degrees(ALPHA_STAR)
    b_tide = critical_beta_tidal_exact(mu)

    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.9))

    # (a) the two singular values -------------------------------------------
    ax = axes[0, 0]
    ax.semilogy(sw['alpha_deg'], sw['sv1'], 'k-', lw=1.0,
                label=r'$\sigma_1$  (cone angle $\alpha$)')
    l2, = ax.semilogy(sw['alpha_deg'], sw['sv2'], 'k-', lw=1.0,
                      label=r'$\sigma_2$  (clock angle $\delta$)')
    l2.set_dashes([5, 2])
    thin_guide(ax, x=a_star_deg)
    ax.set_ylim(1e-6, 3e-1)
    ax.annotate(rf'$\alpha_0^\ast = {a_star_deg:.2f}^\circ$',
                xy=(a_star_deg + 2.5, 1.1e-1), fontsize=7.0, ha='left')
    ax.annotate(r'$\sigma_2 \to 0$ as $\alpha_0 \to 0$',
                xy=(4.0, 1.5e-5), fontsize=7.0, ha='left', va='bottom')
    ax.set_xlabel(r'nominal cone angle  $\alpha_0$  [deg]')
    ax.set_ylabel(r'singular values of $B_{\rm sail}$')
    ax.set_xlim(0, 90)
    ax.legend(loc='lower right', fontsize=7.2)
    panel_label(ax, '(a)')

    # (b) conditioning + the controllability verdict -------------------------
    ax = axes[0, 1]
    ax.plot(sw['alpha_deg'], sw['cond'], 'k-', lw=1.0)
    thin_guide(ax, y=COND_AT_STAR)
    thin_guide(ax, x=a_star_deg)
    ax.annotate(r'$\sigma_2/\sigma_1 = 1/3$ exactly',
                xy=(a_star_deg + 2.5, COND_AT_STAR * 0.90), fontsize=7.0)
    ax.axvspan(0, 0.8, color='0.82', zorder=0)
    ax.annotate(r'$\alpha_0 = 0$: rank 4/6,' + '\n' + 'uncontrollable',
                xy=(0.8, 0.010), xytext=(11.0, 0.028), fontsize=6.9,
                color='0.25', ha='left', va='center',
                arrowprops=dict(arrowstyle='-', color='0.45', lw=0.6,
                                shrinkA=2, shrinkB=1))
    ax.annotate(r'rank 6/6 for every $\alpha_0 > 0$', xy=(52.0, 0.115),
                fontsize=7.0, ha='center')
    ax.set_xlabel(r'nominal cone angle  $\alpha_0$  [deg]')
    ax.set_ylabel(r'conditioning  $\sigma_2/\sigma_1$')
    ax.set_xlim(0, 90)
    ax.set_ylim(0, 0.37)
    panel_label(ax, '(b)')

    # (c) authority against beta, at the optimal angle -----------------------
    ax = axes[1, 0]
    ax.loglog(vb['betas'], vb['sv2'], 'k-', lw=1.0)
    thin_guide(ax, x=b_tide)
    ax.annotate(rf'tidal parity $\beta={b_tide:.4f}$',
                xy=(b_tide * 0.92, vb['sv2'].max() * 0.30), fontsize=6.9,
                rotation=90, ha='right', va='center')
    ax.set_xlabel(r'$\beta$')
    ax.set_ylabel(r'$\sigma_2$ at $\alpha_0^\ast$   [nd]')
    ax.annotate(r'$\sigma_2 \propto \beta$', xy=(1.4e-3, 1.1e-2), fontsize=7.4)
    panel_label(ax, '(c)')

    # (d) the design trade, both curves dimensionless (single axis) ----------
    ax = axes[1, 1]
    thrust_frac = np.cos(sw['alpha'])**2
    auth_frac = sw['sv2'] / sw['sv2'].max()
    ax.plot(sw['alpha_deg'], thrust_frac, 'k-', lw=1.0,
            label=r'thrust  $\cos^2\alpha_0$')
    l3, = ax.plot(sw['alpha_deg'], auth_frac, 'k-', lw=1.0,
                  label=r'authority  $\sigma_2/\sigma_2^{\max}$')
    l3.set_dashes([5, 2])
    thin_guide(ax, x=a_star_deg)
    thin_guide(ax, y=COS2_ALPHA_STAR)
    ax.annotate(r'$\cos^2\alpha_0^\ast = 2/3$', xy=(2.0, COS2_ALPHA_STAR + 0.035),
                fontsize=7.0, ha='left')
    ax.annotate(rf'$\alpha_0^\ast = {a_star_deg:.2f}^\circ$',
                xy=(a_star_deg + 2.0, 0.10), fontsize=7.0, ha='left')
    ax.set_xlabel(r'nominal cone angle  $\alpha_0$  [deg]')
    ax.set_ylabel('fraction of maximum')
    ax.set_xlim(0, 90)
    ax.set_ylim(0, 1.06)
    ax.legend(loc='upper right', fontsize=7.2)
    panel_label(ax, '(d)')

    fig.suptitle(r'Control authority of a solar sail at the artificial '
                 rf'equilibrium  ($\beta = {beta}$)', fontsize=9.5, y=0.999)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output, dpi=200)
    plt.close(fig)
    if verbose:
        print(f"  Saved -> {output}")
    return dict(sweep=sw, vs_beta=vb, alpha_star_deg=a_star_deg)
