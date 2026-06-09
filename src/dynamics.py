# Sun-Earth CR3BP EOM + sail acceleration
import numpy as np

def sail_acceleration(state: list[float], alpha: float, delta: float, beta: float, mu: float) -> list[float]:
    """
    Computes the non-dimensional solar sail acceleration vector in the Circular
    Restricted Three-Body Problem (CR3BP).

    Parameters
    ----------
    state : list[float] or numpy.ndarray
        The 6-element state vector of the spacecraft in the rotating frame,
        ordered as [x, y, z, vx, vy, vz].
    alpha : float
        The cone angle (pitch) in radians. The angle between the sun-to-spacecraft
        vector and the sail normal vector.
    delta : float
        The clock angle (roll/azimuth) in radians. The rotation of the sail normal
        around the sun-line vector.
    beta : float
        The lightness number. A dimensionless parameter representing the ratio of
        maximum solar radiation pressure to solar gravity.
    mu : float
        The mass parameter of the CR3BP system (e.g., Earth mass / total mass).
        The Sun's non-dimensional mass is evaluated as (1 - mu).

    Returns
    -------
    list[float] or numpy.ndarray
        The 3-element acceleration vector [a_x, a_y, a_z] imparted by the solar sail
        in the non-dimensional rotating frame.

    Notes
    -----
    The magnitude of the acceleration is given by:
    |a_sail| = beta * ((1 - mu) / r_1^2) * cos^2(alpha)
    where r_1 is the distance from the Sun (located at x = -mu) to the spacecraft.
    """
    # Extracting the position
    x, y, z = state[0], state[1], state[2]

    # Sun-to-spacecraft vector (Sun is at (-mu, 0, 0)):
    r_vec = np.array([x + mu, y, z])
    r1 = np.linalg.norm(r_vec)
    r_hat = r_vec / r1

    # Build the two perpendicular basis vectors:
    k_hat = np.array([0., 0., 1.])
    t_hat = np.cross(k_hat, r_hat)
    t_norm = np.linalg.norm(t_hat) # for preventing diving by zero
    if t_norm > 1e-12:
        t_hat = t_hat / t_norm
    else:
        t_hat = np.array([1.0, 0.0, 0.0])

    # Sail normal from cone and clock angles:
    n_hat = (np.cos(alpha) * r_hat +
             np.sin(alpha) * np.cos(delta) * t_hat +
             np.sin(alpha) * np.sin(delta) * k_hat)

    # Acceleration magnitude:
    a_mag = beta * (1 - mu) / (r1**2) * (np.cos(alpha)**2)

    a_sail = a_mag * n_hat

    return a_sail


def cr3bp_sail_eom(t: float, state: list[float], alpha: float, delta: float, beta: float, mu: float) -> np.ndarray:
    """
    Computes the equations of motion for a solar sail spacecraft in the
    Circular Restricted Three-Body Problem (CR3BP).

    Parameters
    ----------
    t : float
        Time (required by scipy.integrate.solve_ivp, though the system is autonomous).
    state : list[float] or numpy.ndarray
        The 6-element state vector [x, y, z, vx, vy, vz].
    alpha : float
        Cone angle in radians.
    delta : float
        Clock angle in radians.
    beta : float
        Lightness number of the sail.
    mu : float
        Mass parameter of the CR3BP system.

    Returns
    -------
    numpy.ndarray
        The derivative of the state vector [vx, vy, vz, ax, ay, az].
    """

    # Unpack the state
    x, y, z, vx, vy, vz = state

    # Compute distances to the two primaries
    r1 = np.sqrt((x+mu)**2 + y**2 + z**2)       # distance to sun
    r2 = np.sqrt((x - (1-mu))**2 + y**2 + z**2) # distance to earth

    # Compute standard CR3BP accelerations (moving Coriolis to RHS)
    ax_cr3bp =   2 * vy + x  - ((1 - mu) * (x + mu) / r1**3) - (mu * (x - (1 - mu)) / r2**3)
    ay_cr3bp = - 2 * vx + y  - ((1 - mu) * y / r1**3)        - (mu * y / r2**3)
    az_cr3bp =               - ((1 - mu) * z / r1**3)        - (mu * z / r2**3)

    # Get sail acceleration from the helper function
    a_sail = sail_acceleration(state, alpha, delta, beta, mu)

    # Add sail thrust to the natural dynamics
    a_x = a_sail[0] + ax_cr3bp
    a_y = a_sail[1] + ay_cr3bp
    a_z = a_sail[2] + az_cr3bp

    # Return the state derivative
    return np.array([vx, vy, vz, a_x, a_y, a_z])




