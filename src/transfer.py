# Transfer ΔV with sail arc segments
import numpy as np


def poincare_section(trajectories: list[np.ndarray], section: str = 'y', value: float = 0.0, direction: int = -1) -> \
list[np.ndarray]:
    """
    Finds the first crossing of a specified Poincaré section for a list of trajectories.
    Uses linear interpolation to find the exact state at the crossing plane.

    Parameters
    ----------
    trajectories : list[np.ndarray]
        List of trajectory arrays, each of shape (6, n_steps).
    section : str, optional
        The axis to use for the section plane ('x', 'y', or 'z'), default 'y'.
    value : float, optional
        The coordinate value of the section plane, default 0.0.
    direction : int, optional
        -1 for positive-to-negative crossing, 1 for negative-to-positive,
        0 for any crossing. Default is -1.

    Returns
    -------
    list[np.ndarray]
        A list of 6-element state vectors representing the exact interpolated
        state at the crossing for each strand that reached the section.
    """
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    if section not in axis_map:
        raise ValueError("section must be 'x', 'y', or 'z'")

    idx = axis_map[section]
    crossings = []

    for traj in trajectories:
        vals = traj[idx, :]

        # Scan through the trajectory steps to find where it crosses the plane
        for i in range(len(vals) - 1):
            v1, v2 = vals[i], vals[i + 1]
            is_crossing = False

            if direction == -1 and v1 > value and v2 <= value:
                is_crossing = True
            elif direction == 1 and v1 < value and v2 >= value:
                is_crossing = True
            elif direction == 0 and (v1 - value) * (v2 - value) <= 0:
                is_crossing = True

            if is_crossing:
                # Linear interpolation for sub-step accuracy
                if v2 != v1:
                    frac = (value - v1) / (v2 - v1)
                else:
                    frac = 0.0

                state_cross = traj[:, i] + frac * (traj[:, i + 1] - traj[:, i])
                crossings.append(state_cross)
                # We only care about the FIRST crossing for Poincaré maps
                break

    return crossings


def match_manifolds(crossings_u: list[np.ndarray], crossings_s: list[np.ndarray],
                    w_pos: float = 1000.0,
                    exclude_states: list[np.ndarray] | None = None,
                    min_sep: float = 0.0) -> tuple:
    """
    Finds the best matching pair of states between an unstable and stable manifold
    on a Poincaré section by minimizing position residual and velocity difference (ΔV).

    IMPORTANT — self-intersection
    ─────────────────────────────
    W^u and W^s of the *same* periodic orbit both contain that orbit, so if both
    crossing lists are generated from a single orbit this function will return the
    orbit's own self-intersection: dr → 0, dv → 0, and an apparently "free"
    transfer that is not a transfer at all.  A genuine heteroclinic connection
    requires TWO DISTINCT orbits, and by construction costs zero ΔV — any nonzero
    result is a nearby non-heteroclinic transfer and must be described as such.

    Use `exclude_states` + `min_sep` to forbid matches that sit on (or within
    `min_sep` of) the originating orbits.

    Parameters
    ----------
    crossings_u : list[np.ndarray]
        List of 6-element crossing states from the unstable manifold.
    crossings_s : list[np.ndarray]
        List of 6-element crossing states from the stable manifold.
    w_pos : float, optional
        Weighting factor applied to the position residual to heavily penalize
        trajectories that don't meet at the same physical location.
    exclude_states : list[np.ndarray], optional
        States (e.g. the originating orbits' own section crossings) near which a
        match is rejected.  Guards against returning a self-intersection.
    min_sep : float, optional
        Minimum non-dimensional distance a crossing must keep from every entry of
        `exclude_states`.  Ignored when `exclude_states` is None.

    Returns
    -------
    tuple
        (best_indices, state_u, state_s, delta_v_vec)
        where best_indices is a tuple (i, j) of the best matching strand indices.

    Raises
    ------
    ValueError
        If either list is empty, or if the exclusion guard rejects every pair.
    """
    if not crossings_u or not crossings_s:
        raise ValueError("One or both crossing lists are empty.")

    def _too_close(state: np.ndarray) -> bool:
        if not exclude_states or min_sep <= 0.0:
            return False
        return any(np.linalg.norm(state[:3] - ex[:3]) < min_sep
                   for ex in exclude_states)

    best_cost = np.inf
    best_pair = (None, None)

    for i, su in enumerate(crossings_u):
        if _too_close(su):
            continue
        for j, ss in enumerate(crossings_s):
            if _too_close(ss):
                continue

            # Position error (r_u - r_s)
            dr = np.linalg.norm(su[:3] - ss[:3])

            # Velocity error (v_s - v_u) -- this represents the required delta-V
            dv = np.linalg.norm(ss[3:] - su[3:])

            # Total cost function
            cost = w_pos * dr + dv

            if cost < best_cost:
                best_cost = cost
                best_pair = (i, j)

    i, j = best_pair
    if i is None:
        raise ValueError(
            "No admissible pair: the exclusion guard (min_sep="
            f"{min_sep:g}) rejected every crossing. Either the manifolds have "
            "not separated from their originating orbits within t_max, or both "
            "crossing lists came from the same orbit.")
    best_su = crossings_u[i]
    best_ss = crossings_s[j]

    # Delta-v vector is (Velocity of Stable branch) - (Velocity of Unstable branch)
    # because we are transferring FROM the unstable TO the stable trajectory.
    dv_vec = best_ss[3:] - best_su[3:]

    return (i, j), best_su, best_ss, dv_vec


def transfer_dv(state_u: np.ndarray, state_s: np.ndarray) -> tuple[float, np.ndarray, float]:
    """
    Computes the instantaneous Delta-V and spatial residual between two states.

    Parameters
    ----------
    state_u : np.ndarray
        6-element state vector on the departing (unstable) manifold.
    state_s : np.ndarray
        6-element state vector on the arriving (stable) manifold.

    Returns
    -------
    tuple[float, np.ndarray, float]
        (dv_mag, dv_vec, pos_residual)
    """
    pos_residual = np.linalg.norm(state_u[:3] - state_s[:3])

    dv_vec = state_s[3:] - state_u[3:]
    dv_mag = np.linalg.norm(dv_vec)

    return float(dv_mag), dv_vec, float(pos_residual)