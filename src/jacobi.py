"""
jacobi.py — Jacobi integral of the solar-sail CR3BP.

Kept in its own module so that orbits.py can use it without importing
jacobi_match.py (which imports orbits.py).

Which quantity is actually an integral
──────────────────────────────────────
The orbits in this project are integrated under the FACE-ON sail potential

    U_sail = (x^2 + y^2)/2 + (1-beta)(1-mu)/r1 + mu/r2                    (1)

because a face-on sail (alpha = 0) exerts a purely radial, conservative force
that simply rescales solar gravity.  The conserved quantity is therefore

    C_sail = 2 U_sail - v^2                                               (2)

The *gravitational* Jacobi constant, built from U_grav with (1-mu) in place of
(1-beta)(1-mu), is NOT conserved for beta != 0.  The two differ by

    C_grav = C_sail + 2 beta (1-mu) / r1                                  (3)

and since r1 varies along the orbit, C_grav varies with it.  Differentiating,

    d(C_grav)/dt = -2 v . a_sail                                          (4)

which vanishes only where the sail does no work.  Measured on an atlas member
(beta = 0.001, Az = 5.04e-4, one full period):

    C_sail spread = 1.8e-15      <- integral, at machine precision
    C_grav spread = 5.8e-06      <- not an integral (3.3e9 x larger)

Over an exactly closed orbit r1 returns to its start, so C_grav returns to its
initial value: the failure mode is not a secular drift but that the reported
number depends on WHERE on the orbit it was sampled.  Sampling always at the
same crossing makes the value reproducible and smooth in Az — which is what hid
this bug rather than exposing it.

Why `beta` is a required argument
─────────────────────────────────
An earlier version of this module exposed `jacobi_constant(state, mu)` next to
`jacobi_constant_sail(state, mu, beta)`.  The sail version was never called: two
call sites (continuation.py, orbits.py) silently labelled sail orbits with the
gravitational quantity.  A dead function that was supposed to be called is a
failure mode worth designing out, so `beta` is now REQUIRED and positional.
Any legacy two-argument call raises TypeError immediately instead of returning a
plausible wrong number.  For the deliberate beta = 0 case, either pass 0.0 or
call `jacobi_constant_gravitational` and say so.

CAVEAT for alpha != 0.  A steered sail is non-conservative and NO Jacobi
integral exists.  Neither function in this module is meaningful there; the
energy bookkeeping must be abandoned, not adapted.
"""

import numpy as np


def _radii(state, mu: float):
    x, y, z = state[0], state[1], state[2]
    r1 = np.sqrt((x + mu)**2 + y * y + z * z)
    r2 = np.sqrt((x - (1.0 - mu))**2 + y * y + z * z)
    return r1, r2


def jacobi_constant(state, mu: float, beta: float) -> float:
    """
    Jacobi integral of the face-on solar-sail CR3BP, eq. (2).

    `beta` is REQUIRED — see the module docstring.  Pass beta = 0.0 for the
    unsailed problem.  Valid only for alpha = 0.

    Sign convention: C decreases as the orbit grows.
    """
    x, y, z, vx, vy, vz = state
    r1, r2 = _radii(state, mu)
    U = (0.5 * (x * x + y * y)
         + (1.0 - beta) * (1.0 - mu) / r1
         + mu / r2)
    return float(2.0 * U - (vx * vx + vy * vy + vz * vz))


# Explicit alias.  Prefer this name at call sites where a reader might otherwise
# wonder whether the sail was folded in.
jacobi_constant_sail = jacobi_constant


def jacobi_constant_gravitational(state, mu: float) -> float:
    """
    The purely GRAVITATIONAL Jacobi constant, U_grav with (1-mu) unscaled.

    This is an integral ONLY for beta = 0.  For beta != 0 it is not conserved
    (eq. 3-4) and must not be used to label orbits, compare family members, or
    energy-match orbits for manifold connections.  Provided for the beta = 0
    problem and for diagnostics that deliberately want the difference.
    """
    x, y, z, vx, vy, vz = state
    r1, r2 = _radii(state, mu)
    U = 0.5 * (x * x + y * y) + (1.0 - mu) / r1 + mu / r2
    return float(2.0 * U - (vx * vx + vy * vy + vz * vz))


def jacobi_offset(state, mu: float, beta: float) -> float:
    """
    C_grav - C_sail = 2 beta (1-mu) / r1, eq. (3).

    The entire discrepancy between the two quantities.  Useful in tests: its
    variation along an orbit is exactly the spurious variation of C_grav.
    """
    r1, _ = _radii(state, mu)
    return float(2.0 * beta * (1.0 - mu) / r1)
