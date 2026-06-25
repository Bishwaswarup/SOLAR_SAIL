import numpy as np
from scipy.integrate import solve_ivp

from .orbits import _eom_stm
from .dynamics import cr3bp_sail_eom


def compute_monodromy(state0: list[float], period: float, mu: float, alpha: float, delta: float,
                      beta: float) -> np.ndarray:
    """
    Integrates the state and State Transition Matrix (STM) for exactly
    one full period to compute the monodromy matrix.

    Parameters
    ----------
    state0 : list[float] or np.ndarray
        The initial 6-element state vector on the periodic orbit.
    period : float
        The full orbital period.
    mu : float
        CR3BP mass parameter.
    alpha, delta, beta : float
        Solar sail parameters.

    Returns
    -------
    np.ndarray
        The 6x6 monodromy matrix M = Phi(T, 0).
    """
    # 42-element initial state: [x,y,z,vx,vy,vz,  Phi_00, Phi_01... Phi_55]
    # Phi(0, 0) is the 6x6 identity matrix
    phi0 = np.eye(6).flatten()
    state42_0 = np.concatenate((state0, phi0))

    # Integrate for exactly one period
    res = solve_ivp(
        _eom_stm,
        [0.0, period],
        state42_0,
        args=(alpha, delta, beta, mu),
        rtol=1e-12,
        atol=1e-12
    )

    # Extract the final STM from the last timestep
    M = res.y[6:, -1].reshape((6, 6))
    return M


def manifold_directions(M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Performs eigendecomposition on the monodromy matrix M to extract
    the stable and unstable directions.

    Parameters
    ----------
    M : np.ndarray
        The 6x6 monodromy matrix.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (v_stable, v_unstable): The real parts of the eigenvectors corresponding
        to the stable (magnitude < 1) and unstable (magnitude > 1) eigenvalues.
    """
    w, v = np.linalg.eig(M)

    # The unstable eigenvalue has the largest absolute magnitude
    idx_u = np.argmax(np.abs(w))
    # The stable eigenvalue has the smallest absolute magnitude
    idx_s = np.argmin(np.abs(w))

    # Extract eigenvectors, taking only the real part to eliminate numerical noise,
    # and normalize to unit length.
    v_u = np.real(v[:, idx_u])
    v_u /= np.linalg.norm(v_u)

    v_s = np.real(v[:, idx_s])
    v_s /= np.linalg.norm(v_s)

    return v_s, v_u


def compute_manifold(state0: list[float], period: float, mu: float, alpha: float, delta: float, beta: float,
                     direction: str = 'unstable', branch: str = '+',
                     n_points: int = 50, eps: float = 1e-6, t_max: float = 3 * np.pi) -> list[np.ndarray]:
    """
    Generates a manifold tube by sampling N points along the halo orbit,
    perturbing the state along the local manifold direction, and integrating.

    Parameters
    ----------
    state0 : list[float] or np.ndarray
        Initial state on the halo orbit.
    period : float
        Full period of the halo orbit.
    mu, alpha, delta, beta : float
        CR3BP and solar sail parameters.
    direction : str, optional
        'unstable' (integrate forward) or 'stable' (integrate backward), default 'unstable'.
    branch : str, optional
        '+' for positive perturbation, '-' for negative perturbation, default '+'.
    n_points : int, optional
        Number of trajectories (strands) to generate along the halo orbit, default 50.
    eps : float, optional
        Perturbation magnitude, default 1e-6.
    t_max : float, optional
        Maximum integration time for each manifold strand, default 3*pi.

    Returns
    -------
    list[np.ndarray]
        A list of N trajectory arrays, where each array has shape (6, n_steps).
    """
    # 1. Integrate the nominal orbit for one full period to get the states
    # and STMs at n_points equally spaced intervals.
    t_eval = np.linspace(0, period, n_points + 1)
    phi0 = np.eye(6).flatten()
    state42_0 = np.concatenate((state0, phi0))

    res_orbit = solve_ivp(
        _eom_stm,
        [0, period],
        state42_0,
        t_eval=t_eval,
        args=(alpha, delta, beta, mu),
        rtol=1e-12,
        atol=1e-12
    )

    # Extract the monodromy matrix (Phi(T, 0)) and eigendecompose
    M = res_orbit.y[6:, -1].reshape((6, 6))
    v_s, v_u = manifold_directions(M)

    # 2. Configure integration settings based on direction and branch
    if direction == 'unstable':
        v_base = v_u
        t_span = [0, t_max]
    elif direction == 'stable':
        v_base = v_s
        t_span = [0, -t_max]
    else:
        raise ValueError("direction must be either 'unstable' or 'stable'")

    sign = 1.0 if branch == '+' else -1.0

    # Setup evaluation times to ensure identically shaped output arrays (good for plotting/saving)
    t_eval_man = np.linspace(0, t_span[1], 1000)

    trajectories = []

    # 3. Loop over the points on the orbit, perturb, and integrate
    for i in range(n_points):
        # Extract local state and STM Phi(t_i, 0)
        state_i = res_orbit.y[:6, i]
        Phi_i = res_orbit.y[6:, i].reshape((6, 6))

        # Map the base eigenvector to the local time t_i
        v_local = Phi_i @ v_base
        v_local /= np.linalg.norm(v_local)

        # Apply the perturbation
        state_pert = state_i + (sign * eps * v_local)

        # Integrate the perturbed state
        res_man = solve_ivp(
            cr3bp_sail_eom,
            t_span,
            state_pert,
            t_eval=t_eval_man,
            args=(alpha, delta, beta, mu),
            rtol=1e-10,
            atol=1e-10
        )

        trajectories.append(res_man.y)

    return trajectories