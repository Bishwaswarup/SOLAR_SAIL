"""
Solar Sail CR3BP — test suite
Usage:
    python test.py                  # run all tests
    python test.py dynamics         # test dynamics.py only
    python test.py equilibria       # test equilibria.py only
    python test.py orbits           # test orbits.py only
"""
import sys
import os
import argparse
import numpy as np

# Ensure the project root (parent of src/) is on sys.path so that
# 'from src.X import ...' works regardless of where the script is invoked from.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

MU = 3.003e-6  # Sun-Earth mass parameter


# ── dynamics ──────────────────────────────────────────────────────────────────

def test_dynamics():
    from src.dynamics import sail_acceleration, cr3bp_sail_eom
    print("=== dynamics.py ===")
    errors = []

    state = [0.99, 0.0, 0.0, 0.0, 0.0, 0.0]

    # 1. beta=0 → zero thrust always
    a = sail_acceleration(state, alpha=0.0, delta=0.0, beta=0.0, mu=MU)
    if np.allclose(a, 0):
        print("  PASS  1  beta=0 → [0, 0, 0]")
    else:
        print(f"  FAIL  1  beta=0: got {a}"); errors.append(1)

    # 2. alpha=π/2 → edge-on, zero thrust
    a = sail_acceleration(state, alpha=np.pi/2, delta=0.0, beta=0.05, mu=MU)
    if np.allclose(a, 0, atol=1e-10):
        print("  PASS  2  alpha=90° → [0, 0, 0]")
    else:
        print(f"  FAIL  2  alpha=90°: got {a}"); errors.append(2)

    # 3. alpha=0, on x-axis → purely radial thrust
    a = sail_acceleration(state, alpha=0.0, delta=0.0, beta=0.05, mu=MU)
    r1 = 0.99 + MU
    expected = 0.05 * (1 - MU) / r1**2
    if abs(a[0] - expected) < 1e-10 and np.allclose(a[1:], 0):
        print(f"  PASS  3  radial thrust magnitude = {a[0]:.8f}")
    else:
        print(f"  FAIL  3  got {a}, expected [{expected}, 0, 0]"); errors.append(3)

    # 4. EOM returns (6,) array; velocities zero for zero-velocity state
    d = cr3bp_sail_eom(0.0, state, alpha=0.0, delta=0.0, beta=0.05, mu=MU)
    if d.shape == (6,) and np.allclose(d[:3], 0):
        print(f"  PASS  4  EOM shape=(6,), velocities zero  accel={d[3:]}")
    else:
        print(f"  FAIL  4  EOM output: {d}"); errors.append(4)

    return errors


# ── equilibria ────────────────────────────────────────────────────────────────

def test_equilibria():
    from src.equilibria import find_artificial_equilibrium
    print("\n=== equilibria.py ===")
    errors = []

    # 5. beta=0 → recover classical L1 near x≈0.990
    eq0 = find_artificial_equilibrium(0.0, 0.0, 0.0, MU, [0.99, 0.0, 0.0])
    if abs(eq0[0] - 0.990) < 0.005 and np.allclose(eq0[1:], 0, atol=1e-8):
        print(f"  PASS  5  classical L1 recovered  x={eq0[0]:.8f}")
    else:
        print(f"  FAIL  5  L1={eq0}"); errors.append(5)

    # 6. beta=0.05, alpha=0 → L1 shifts sunward
    eq_sail = find_artificial_equilibrium(0.0, 0.0, 0.05, MU, [0.99, 0.0, 0.0])
    shift = eq0[0] - eq_sail[0]
    if shift > 0:
        print(f"  PASS  6  sail shifts L1 sunward by {shift:.6f} AU  (new x={eq_sail[0]:.8f})")
    else:
        print(f"  FAIL  6  shift={shift} (expected > 0)"); errors.append(6)

    return errors


# ── orbits ────────────────────────────────────────────────────────────────────

def test_orbits():
    from src.equilibria import find_artificial_equilibrium
    from src.orbits import compute_halo_orbit
    print("\n=== orbits.py ===")
    errors = []

    # 7. Classical halo orbit (beta=0)
    # Az=0.003 ≈ 450,000 km — a realistic Sun-Earth L1 northern halo amplitude.
    # Az=0.008 is ~80% of the Hill-sphere radius; the Richardson series is not
    # accurate there, and the unconstrained Newton step collapses T to zero.
    eq0 = find_artificial_equilibrium(0.0, 0.0, 0.0, MU, [0.99, 0.0, 0.0])
    try:
        state0, T = compute_halo_orbit(eq0, Az=0.003, mu=MU,
                                       alpha=0.0, delta=0.0, beta=0.0)
        sym_ok = (abs(state0[1]) < 1e-8 and
                  abs(state0[3]) < 1e-8 and
                  abs(state0[5]) < 1e-8)
        if T > 0 and sym_ok:
            print(f"  PASS  7  classical halo  T={T:.6f}  x0={state0[0]:.8f}  vy0={state0[4]:.8f}")
        else:
            print(f"  FAIL  7  T={T}, state0={state0}"); errors.append(7)
    except Exception as e:
        print(f"  FAIL  7  exception: {e}"); errors.append(7)

    # 8. Sail halo orbit (beta=0.05)
    eq_sail = find_artificial_equilibrium(0.0, 0.0, 0.05, MU, [0.99, 0.0, 0.0])
    try:
        state0_s, T_s = compute_halo_orbit(eq_sail, Az=0.005, mu=MU,
                                            alpha=0.0, delta=0.0, beta=0.05)
        if T_s > 0:
            print(f"  PASS  8  sail halo  T={T_s:.6f}  x0={state0_s[0]:.8f}  vy0={state0_s[4]:.8f}")
        else:
            print(f"  FAIL  8  T={T_s}"); errors.append(8)
    except Exception as e:
        print(f"  FAIL  8  exception: {e}"); errors.append(8)

    return errors


# ── manifolds ─────────────────────────────────────────────────────────────────

def test_manifolds():
    from src.equilibria import find_artificial_equilibrium
    from src.orbits import compute_halo_orbit
    from src.manifolds import compute_monodromy, manifold_directions, compute_manifold
    print("\n=== manifolds.py ===")
    errors = []

    # Shared setup: classical halo at Az=0.003
    eq0    = find_artificial_equilibrium(0.0, 0.0, 0.0, MU, [0.99, 0.0, 0.0])
    state0, T = compute_halo_orbit(eq0, Az=0.003, mu=MU,
                                   alpha=0.0, delta=0.0, beta=0.0)
    M = compute_monodromy(state0, T, MU, 0.0, 0.0, 0.0)

    # 9. Monodromy eigenvalue structure (Hamiltonian symplectic constraints)
    #    - One real pair (λ_u, λ_s) with λ_u >> 1 and λ_u · λ_s ≈ 1.
    #    - Four eigenvalues near the unit circle |λ| ≈ 1 (two center pairs).
    #    - One of those four is always +1 (Floquet: orbit direction).
    try:
        w = np.linalg.eigvals(M)
        abs_w   = np.sort(np.abs(w))          # ascending: [λ_s, ..., λ_u]
        lambda_s = abs_w[0]
        lambda_u = abs_w[-1]
        product  = lambda_u * lambda_s        # should be ≈ 1
        neutral  = abs_w[1:-1]               # four middle eigenvalues ≈ 1

        reciprocal_ok = abs(product - 1.0) < 0.02   # within 2 % of 1
        unstable_ok   = lambda_u > 50               # Sun-Earth L1: λ_u ~ thousands
        neutral_ok    = all(abs(x - 1.0) < 0.05 for x in neutral)

        if reciprocal_ok and unstable_ok and neutral_ok:
            print(f"  PASS  9  monodromy OK: λ_u={lambda_u:.3e}  λ_s={lambda_s:.3e}"
                  f"  λ_u·λ_s={product:.6f}  neutral≈{neutral.round(4)}")
        else:
            print(f"  FAIL  9  λ_u={lambda_u:.3e}  λ_s={lambda_s:.3e}"
                  f"  product={product:.4f}  neutral={neutral}")
            errors.append(9)
    except Exception as e:
        print(f"  FAIL  9  exception: {e}"); errors.append(9)

    # 10. Manifold strand — shape, no NaN, and exponential departure from orbit.
    #     The unstable manifold eigenvalue for Sun-Earth L1 is λ_u ~ thousands per
    #     period T.  After integrating for t_max = π ≈ T/2, the initial ε=1e-6
    #     perturbation should have grown to at least 1e-4 (factor 100, very
    #     conservative given the actual growth).
    try:
        strands = compute_manifold(
            state0, T, MU, 0.0, 0.0, 0.0,
            direction='unstable', branch='+',
            n_points=6, eps=1e-6, t_max=np.pi,
        )
        strand  = strands[0]                          # shape (6, n_eval)
        no_nan  = np.all(np.isfinite(strand))
        # Displacement of final point from initial point on the strand
        displacement = np.linalg.norm(strand[:3, -1] - strand[:3, 0])

        shape_ok = (strand.ndim == 2 and strand.shape[0] == 6 and strand.shape[1] > 1)
        grows_ok = displacement > 1e-4

        if shape_ok and no_nan and grows_ok:
            print(f"  PASS 10  strand shape={strand.shape}  "
                  f"displacement={displacement:.3e}  no NaN={no_nan}")
        else:
            print(f"  FAIL 10  shape={strand.shape}  displacement={displacement:.3e}"
                  f"  finite={no_nan}")
            errors.append(10)
    except Exception as e:
        print(f"  FAIL 10  exception: {e}"); errors.append(10)

    return errors


# ── sail_control ───────────────────────────────────────────────────────────────

def test_sail_control():
    from src.equilibria import find_artificial_equilibrium
    from src.orbits import compute_halo_orbit
    from src.sail_control import optimal_sail_angles, reachable_set, station_keeping
    print("\n=== sail_control.py ===")
    errors = []

    # Shared state: a point on the x-axis near L1, typical sail setup
    state = [0.988, 0.0, 0.003, 0.0, 0.01, 0.0]
    beta, mu = 0.05, MU

    # 11. optimal_sail_angles: closed-form vs brute-force grid
    #     For a purely radial target direction (+x̂ ≈ sunward), α*=0 and the
    #     resulting thrust should equal the maximum any grid sample achieves.
    try:
        target_radial = [1.0, 0.0, 0.0]
        a_opt, d_opt, a_vec = optimal_sail_angles(state, target_radial, beta, mu)

        # Brute-force: max projection over a dense (α, δ) grid
        cloud = reachable_set(state, beta, mu, n_alpha=60, n_delta=72)
        best_grid = np.max(cloud @ np.array(target_radial))
        opt_proj  = np.dot(a_vec, target_radial)

        # Analytic result should match (or exceed) grid best to within grid resolution
        close_enough = abs(opt_proj - best_grid) < 1e-4

        if close_enough and 0.0 <= a_opt <= np.pi / 2:
            print(f"  PASS 11  optimal α={np.degrees(a_opt):.2f}°  δ={np.degrees(d_opt):.2f}°"
                  f"  proj={opt_proj:.6f}  grid_best={best_grid:.6f}")
        else:
            print(f"  FAIL 11  opt_proj={opt_proj:.6f}  grid_best={best_grid:.6f}"
                  f"  α={np.degrees(a_opt):.2f}°")
            errors.append(11)
    except Exception as e:
        print(f"  FAIL 11  exception: {e}"); errors.append(11)

    # 12. reachable_set: shape, finiteness, and α=0 row is along r̂ (radial)
    try:
        cloud = reachable_set(state, beta, mu, n_alpha=30, n_delta=36)
        shape_ok  = cloud.shape == (30 * 36, 3)
        finite_ok = np.all(np.isfinite(cloud))

        # α=0 (first row, any δ) → thrust is purely radial regardless of δ
        # because sin(0)=0 zeroes the transverse terms.  Verify direction matches r̂.
        r_vec = np.array([state[0] + mu, state[1], state[2]])
        r_hat = r_vec / np.linalg.norm(r_vec)
        a0    = cloud[0]                         # alpha=0, delta=0
        a0_hat = a0 / np.linalg.norm(a0)
        aligned = abs(np.dot(a0_hat, r_hat) - 1.0) < 1e-10   # should be ≈ 1

        if shape_ok and finite_ok and aligned:
            print(f"  PASS 12  reachable_set shape={cloud.shape}  "
                  f"α=0 row aligned with r̂: cos(θ)={np.dot(a0_hat, r_hat):.10f}")
        else:
            print(f"  FAIL 12  shape={cloud.shape}  finite={finite_ok}"
                  f"  aligned={aligned}  cos(θ)={np.dot(a0_hat, r_hat):.6f}")
            errors.append(12)
    except Exception as e:
        print(f"  FAIL 12  exception: {e}"); errors.append(12)

    # 13. station_keeping: one correction reduces terminal velocity error.
    #
    # Design: the reference orbit is the sail halo at alpha=0, beta=0.05.
    # We fly it at a small nominal alpha (5°) so that delta also has an effect
    # and the 2×2 sensitivity matrix is non-singular.  Both "before" and "after"
    # use the same state_pert and the same nominal alpha, so the comparison is fair:
    #   err0 = terminal error when flying uncorrected at alpha_nom
    #   err1 = terminal error when flying at (alpha_new, delta_new) from station_keeping
    # The perturbation is 1e-6 (≈150 km) to stay well within the linear regime.
    try:
        eq0       = find_artificial_equilibrium(0.0, 0.0, 0.05, MU, [0.99, 0.0, 0.0])
        state0, T = compute_halo_orbit(eq0, Az=0.005, mu=MU,
                                       alpha=0.0, delta=0.0, beta=0.05)

        from scipy.integrate import solve_ivp
        from src.dynamics import cr3bp_sail_eom

        def y_cross(t, sv, *a): return sv[1]
        y_cross.terminal = True; y_cross.direction = -1

        alpha_nom = np.radians(5.0)           # 5° nominal — non-degenerate
        state_pert = state0.copy()
        state_pert[0] += 1e-6                 # small position offset (≈150 km)

        # err0: fly perturbed orbit at uncorrected alpha_nom
        res0 = solve_ivp(cr3bp_sail_eom, [0, T], state_pert,
                         events=y_cross, args=(alpha_nom, 0.0, beta, MU),
                         rtol=1e-10, atol=1e-10)
        sf0  = res0.y_events[0][0]
        err0 = abs(sf0[3]) + abs(sf0[5])

        # Apply one correction
        alpha_new, delta_new = station_keeping(
            state_pert, state0, T / 2, MU, alpha_nom, 0.0, beta
        )

        # err1: same perturbed orbit at corrected angles
        res1 = solve_ivp(cr3bp_sail_eom, [0, T], state_pert,
                         events=y_cross, args=(alpha_new, delta_new, beta, MU),
                         rtol=1e-10, atol=1e-10)
        sf1  = res1.y_events[0][0]
        err1 = abs(sf1[3]) + abs(sf1[5])

        if err1 < err0:
            print(f"  PASS 13  station-keeping reduces error:"
                  f"  before={err0:.2e}  after={err1:.2e}"
                  f"  α_new={np.degrees(alpha_new):.3f}°  δ_new={np.degrees(delta_new):.3f}°")
        else:
            print(f"  FAIL 13  error did not decrease: before={err0:.2e}  after={err1:.2e}"
                  f"  α_new={np.degrees(alpha_new):.3f}°  δ_new={np.degrees(delta_new):.3f}°")
            errors.append(13)
    except Exception as e:
        print(f"  FAIL 13  exception: {e}"); errors.append(13)

    return errors


# ── transfer ──────────────────────────────────────────────────────────────────

def test_transfer():
    from src.equilibria import find_artificial_equilibrium
    from src.orbits import compute_halo_orbit
    from src.manifolds import compute_manifold
    from src.transfer import poincare_section, match_manifolds, transfer_dv
    print("\n=== transfer.py ===")
    errors = []

    # 14. poincare_section: interpolation accuracy on a known analytic trajectory.
    #     Construct a simple helix that crosses y=0 exactly at a known state,
    #     then verify the interpolated crossing matches to near-machine precision.
    try:
        # Parametric: x=1, y=sin(t), z=0, vx=0, vy=cos(t), vz=0
        # y=0 crossing (downward, sin going − ) at t = π → state = [1,0,0,0,-1,0]
        t = np.linspace(0, 2 * np.pi, 10001)
        traj = np.array([
            np.ones_like(t),   # x
            np.sin(t),         # y
            np.zeros_like(t),  # z
            np.zeros_like(t),  # vx
            np.cos(t),         # vy
            np.zeros_like(t),  # vz
        ])  # shape (6, 10001)

        crossings = poincare_section([traj], section='y', value=0.0, direction=-1)

        # Expect one crossing: at y=0 going downward (vy < 0) → vy_cross ≈ -1
        expected = np.array([1.0, 0.0, 0.0, 0.0, -1.0, 0.0])
        interp_err = np.linalg.norm(crossings[0] - expected)

        if len(crossings) == 1 and interp_err < 1e-6:
            print(f"  PASS 14  poincare_section interpolation error = {interp_err:.2e}")
        else:
            print(f"  FAIL 14  n_crossings={len(crossings)}  interp_err={interp_err:.2e}")
            errors.append(14)
    except Exception as e:
        print(f"  FAIL 14  exception: {e}"); errors.append(14)

    # 15. match_manifolds: finds the minimum-cost pair from known crossing lists.
    #     Build two synthetic crossing lists where the true best match is (1, 2)
    #     and verify the function returns that pair.
    try:
        np.random.seed(42)
        # Unstable crossings: 5 random states
        cu = [np.random.randn(6) * 0.1 for _ in range(5)]
        # Stable crossings: 5 random states, but make cu[1] ≈ cs[2] (best match)
        cs = [np.random.randn(6) * 0.1 for _ in range(5)]
        cs[2] = cu[1] + np.array([0.0, 0.0, 0.0, 1e-5, 1e-5, 1e-5])  # tiny Δv only

        (i_best, j_best), su, ss, dv_vec = match_manifolds(cu, cs, w_pos=1000.0)

        if i_best == 1 and j_best == 2:
            dv_mag = np.linalg.norm(dv_vec)
            print(f"  PASS 15  match_manifolds found (1,2)  |Δv|={dv_mag:.2e}")
        else:
            print(f"  FAIL 15  got pair ({i_best},{j_best}) instead of (1,2)")
            errors.append(15)
    except Exception as e:
        print(f"  FAIL 15  exception: {e}"); errors.append(15)

    # 16. transfer_dv: algebraic check on known states.
    #     state_u has velocity [1,0,0], state_s has velocity [1,0.01,0].
    #     Δv should be [0, 0.01, 0], |Δv|=0.01, pos_residual=0.
    try:
        state_u = np.array([1.0, 0.0, 0.0, 1.0, 0.00, 0.0])
        state_s = np.array([1.0, 0.0, 0.0, 1.0, 0.01, 0.0])
        dv_mag, dv_vec, pos_res = transfer_dv(state_u, state_s)

        mag_ok = abs(dv_mag - 0.01) < 1e-12
        vec_ok = np.allclose(dv_vec, [0.0, 0.01, 0.0])
        pos_ok = pos_res < 1e-12

        if mag_ok and vec_ok and pos_ok:
            print(f"  PASS 16  transfer_dv |Δv|={dv_mag:.4f}  "
                  f"vec={dv_vec}  pos_res={pos_res:.2e}")
        else:
            print(f"  FAIL 16  |Δv|={dv_mag}  vec={dv_vec}  pos_res={pos_res}")
            errors.append(16)
    except Exception as e:
        print(f"  FAIL 16  exception: {e}"); errors.append(16)

    # 17. End-to-end: unstable manifold strands from the classical halo cross y=0,
    #     and the best self-match (same orbit, opposite branches) has small |Δv|.
    #     This verifies the full pipeline: manifold → section → match → Δv.
    try:
        eq0      = find_artificial_equilibrium(0.0, 0.0, 0.0, MU, [0.99, 0.0, 0.0])
        state0, T = compute_halo_orbit(eq0, Az=0.003, mu=MU,
                                       alpha=0.0, delta=0.0, beta=0.0)

        strands_p = compute_manifold(state0, T, MU, 0.0, 0.0, 0.0,
                                     direction='unstable', branch='+',
                                     n_points=20, eps=1e-6, t_max=2.5*np.pi)
        strands_m = compute_manifold(state0, T, MU, 0.0, 0.0, 0.0,
                                     direction='unstable', branch='-',
                                     n_points=20, eps=1e-6, t_max=2.5*np.pi)

        # Poincaré section at y=0, accept either crossing direction for richer coverage
        cross_p = poincare_section(strands_p, section='y', value=0.0, direction=0)
        cross_m = poincare_section(strands_m, section='y', value=0.0, direction=0)

        # We expect at least some strands to reach y=0
        if not cross_p or not cross_m:
            raise RuntimeError(f"Too few crossings: +branch={len(cross_p)}, -branch={len(cross_m)}")

        (ib, jb), su, ss, dv_vec = match_manifolds(cross_p, cross_m)
        dv_mag, _, pos_res = transfer_dv(su, ss)

        # Crossings should be finite and the best pair should have a small position residual
        finite_ok = np.all(np.isfinite(su)) and np.all(np.isfinite(ss))

        if finite_ok and pos_res < 0.1:
            print(f"  PASS 17  end-to-end pipeline OK  "
                  f"+crossings={len(cross_p)}  -crossings={len(cross_m)}  "
                  f"|Δv|={dv_mag:.4f}  pos_res={pos_res:.4f}")
        else:
            print(f"  FAIL 17  finite={finite_ok}  pos_res={pos_res:.4f}  |Δv|={dv_mag:.4f}")
            errors.append(17)
    except Exception as e:
        print(f"  FAIL 17  exception: {e}"); errors.append(17)

    return errors


# ── viz ───────────────────────────────────────────────────────────────────────

def test_viz():
    """
    Smoke tests for viz.py.  No display is needed — we use the Agg backend so
    no GUI window opens.  Each test checks:
      • the function returns without raising,
      • the returned Axes / Figure has the expected type and structure,
      • axis labels are set correctly.
    Actual pixel-level rendering is left to manual inspection of saved PNGs.
    """
    import matplotlib
    matplotlib.use('Agg')          # non-interactive; safe in headless CI
    import matplotlib.pyplot as plt
    from src.viz import (
        plot_system, plot_orbit, plot_manifold_tube,
        plot_poincare, plot_reachable_set, plot_transfer,
        make_paper_figure,
    )
    from src.sail_control import reachable_set
    print("\n=== viz.py ===")
    errors = []

    # Minimal synthetic data that exercises every code path without a real solve.
    # One period of a circular "orbit" in the x-z plane.
    T_fake = 2.0 * np.pi
    t_eval = np.linspace(0, T_fake, 200)
    fake_traj = np.array([
        0.99 + 0.003 * np.cos(t_eval),   # x
        np.zeros_like(t_eval),            # y
        0.003 * np.sin(t_eval),           # z
        -0.003 * np.sin(t_eval),          # vx
        np.zeros_like(t_eval),            # vy
        0.003 * np.cos(t_eval),           # vz
    ])  # shape (6, 200)

    # Fake equilibrium position
    eq_pos = np.array([0.985, 0.0, 0.0])

    # Fake Poincaré crossings (a few 6-element state vectors)
    crossings_u = [np.array([0.99, 0.0, 0.003, 0.01, 0.0, 0.005]) + i * 1e-4
                   for i in range(8)]
    crossings_s = [np.array([0.99, 0.0, 0.003, -0.01, 0.0, 0.005]) + i * 1e-4
                   for i in range(8)]

    # Fake reachable set
    cloud = reachable_set([0.99, 0.0, 0.0, 0.0, 0.01, 0.0], beta=0.05, mu=MU,
                          n_alpha=10, n_delta=12)

    # 18. plot_system — 2-D (xz) and 3-D projections both return Axes.
    try:
        fig2d, ax2d = plt.subplots()
        ax_ret = plot_system(MU, ax=ax2d, proj='xz', eq_pos=eq_pos)
        label_ok = (ax2d.get_xlabel() != '' and ax2d.get_ylabel() != '')

        fig3d = plt.figure()
        ax3d  = fig3d.add_subplot(111, projection='3d')
        ax_ret3 = plot_system(MU, ax=ax3d, proj='3d', eq_pos=eq_pos)

        if ax_ret is ax2d and label_ok and ax_ret3 is ax3d:
            print("  PASS 18  plot_system: 2-D and 3-D return correct axes, labels set")
        else:
            print(f"  FAIL 18  ax_ret={ax_ret}  label_ok={label_ok}  ax_ret3={ax_ret3}")
            errors.append(18)
        plt.close('all')
    except Exception as e:
        print(f"  FAIL 18  exception: {e}"); errors.append(18)
        plt.close('all')

    # 19. plot_orbit / plot_manifold_tube / plot_poincare / plot_reachable_set /
    #     plot_transfer — each must run without error and return an Axes object.
    try:
        failed_funcs = []

        # plot_orbit with fake state0 and fake period — _propagate re-integrates,
        # so we need *real* dynamics.  Use classical halo from test 7 setup.
        from src.equilibria import find_artificial_equilibrium
        from src.orbits import compute_halo_orbit
        eq0 = find_artificial_equilibrium(0.0, 0.0, 0.0, MU, [0.99, 0.0, 0.0])
        s0, T = compute_halo_orbit(eq0, Az=0.003, mu=MU,
                                   alpha=0.0, delta=0.0, beta=0.0)

        fig, ax = plt.subplots()
        ret = plot_orbit(s0, T, MU, ax=ax, proj='xz', n=100)
        if ret is not ax:
            failed_funcs.append('plot_orbit')

        fig, ax = plt.subplots()
        ret = plot_manifold_tube([fake_traj], ax=ax, proj='xz')
        if ret is not ax:
            failed_funcs.append('plot_manifold_tube')

        fig, ax = plt.subplots()
        ret = plot_poincare(crossings_u, ax=ax, coords=(0, 3))
        if ret is not ax:
            failed_funcs.append('plot_poincare (u)')

        fig, ax = plt.subplots()
        ret = plot_poincare(crossings_s, ax=ax, coords=(0, 2))
        if ret is not ax:
            failed_funcs.append('plot_poincare (s)')

        fig3d = plt.figure()
        ax3d  = fig3d.add_subplot(111, projection='3d')
        ret = plot_reachable_set(cloud, ax=ax3d)
        if ret is not ax3d:
            failed_funcs.append('plot_reachable_set')

        fig, ax = plt.subplots()
        ret = plot_transfer(fake_traj, fake_traj,
                            state_u=fake_traj[:, 50],
                            state_s=fake_traj[:, 100],
                            ax=ax, proj='xz')
        if ret is not ax:
            failed_funcs.append('plot_transfer')

        plt.close('all')

        if not failed_funcs:
            print("  PASS 19  plot_orbit, manifold_tube, poincare (×2),"
                  " reachable_set, transfer — all return correct axes")
        else:
            print(f"  FAIL 19  wrong return value from: {failed_funcs}")
            errors.append(19)
    except Exception as e:
        print(f"  FAIL 19  exception: {e}"); errors.append(19)
        plt.close('all')

    # 20. make_paper_figure — returns a Figure with exactly 4 Axes, titles set.
    try:
        from src.equilibria import find_artificial_equilibrium
        from src.orbits import compute_halo_orbit
        eq0 = find_artificial_equilibrium(0.0, 0.0, 0.0, MU, [0.99, 0.0, 0.0])
        s0_c, T_c = compute_halo_orbit(eq0, Az=0.003, mu=MU,
                                       alpha=0.0, delta=0.0, beta=0.0)
        eq_s = find_artificial_equilibrium(0.0, 0.0, 0.05, MU, [0.99, 0.0, 0.0])
        s0_s, T_s = compute_halo_orbit(eq_s, Az=0.003, mu=MU,
                                       alpha=0.0, delta=0.0, beta=0.05)

        fig = make_paper_figure(
            s0_c, T_c, MU,
            s0_s, T_s, 0.0, 0.0, 0.05,
            [fake_traj], [fake_traj],      # strands_u, strands_s
            crossings_u, crossings_s,
            cloud,
            eq_pos_sail=eq_s,
        )

        import matplotlib.figure
        n_axes   = len(fig.axes)
        is_fig   = isinstance(fig, matplotlib.figure.Figure)
        # Panels A–D must have non-empty titles
        titles   = [ax.get_title() for ax in fig.axes if hasattr(ax, 'get_title')]
        has_titles = sum(1 for t in titles if t.strip()) >= 4

        plt.close('all')

        if is_fig and n_axes == 4 and has_titles:
            print(f"  PASS 20  make_paper_figure: Figure OK  axes={n_axes}"
                  f"  panel titles={[t[:20] for t in titles if t.strip()]}")
        else:
            print(f"  FAIL 20  is_fig={is_fig}  axes={n_axes}  has_titles={has_titles}")
            errors.append(20)
    except Exception as e:
        print(f"  FAIL 20  exception: {e}"); errors.append(20)
        plt.close('all')

    return errors


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Solar Sail CR3BP test suite")
    parser.add_argument(
        "component",
        nargs="?",
        default="all",
        choices=["dynamics", "equilibria", "orbits", "manifolds", "sail_control", "transfer", "viz", "all"],
        help="Component to test  (default: all)",
    )
    args = parser.parse_args()

    all_errors = []
    if args.component in ("dynamics", "all"):
        all_errors += test_dynamics()
    if args.component in ("equilibria", "all"):
        all_errors += test_equilibria()
    if args.component in ("orbits", "all"):
        all_errors += test_orbits()
    if args.component in ("manifolds", "all"):
        all_errors += test_manifolds()
    if args.component in ("sail_control", "all"):
        all_errors += test_sail_control()
    if args.component in ("transfer", "all"):
        all_errors += test_transfer()
    if args.component in ("viz", "all"):
        all_errors += test_viz()

    print()
    if all_errors:
        print(f"FAILED tests: {all_errors}")
        sys.exit(1)
    else:
        print("All selected tests passed.")


if __name__ == "__main__":
    main()
