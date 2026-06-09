# Artificial L-point solver
import numpy as np
from scipy.optimize import fsolve

# Import your equations of motion from the dynamics module
from .dynamics import cr3bp_sail_eom


def find_artificial_equilibrium(alpha: float, delta: float, beta: float, mu: float, x0: list[float]) -> np.ndarray:
    """
    Finds the artificial Lagrange point (equilibrium position) for a solar sail
    in the CR3BP for a fixed attitude.

    Parameters
    ----------
    alpha : float
        Cone angle in radians.
    delta : float
        Clock angle in radians.
    beta : float
        Lightness number of the sail.
    mu : float
        Mass parameter of the CR3BP system.
    x0 : list[float] or numpy.ndarray
        The 3-element initial guess for the equilibrium position [x, y, z].

    Returns
    -------
    numpy.ndarray
        The 3-element array [x*, y*, z*] representing the exact artificial
        equilibrium position.
    """

    def residual(pos):
        # At an equilibrium point, the spacecraft is stationary in the rotating frame.
        # We form the state vector with zero velocities.
        state = [pos[0], pos[1], pos[2], 0.0, 0.0, 0.0]

        # cr3bp_sail_eom returns [vx, vy, vz, ax, ay, az].
        # We evaluate the total acceleration at this static point.
        # (Coriolis terms naturally vanish because vx = vy = vz = 0)
        state_dot = cr3bp_sail_eom(0.0, state, alpha, delta, beta, mu)

        # The residual is the acceleration vector [ax, ay, az].
        # fsolve will drive this to [0, 0, 0].
        return state_dot[3:6]

    # fsolve takes the residual function and the initial guess,
    # returning the coordinates where the residual is zero.
    eq_pos, info, ier, msg = fsolve(residual, x0, full_output=True)
    if ier != 1:
        raise RuntimeError(f"find_artificial_equilibrium did not converge: {msg}")
    return eq_pos

    return eq_pos