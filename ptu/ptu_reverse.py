# ------------------------------------------------------------------------------
#
#   PTU Reverse -- given all sector angles, all in-dihedral angles, and
#   ONE known out-dihedral angle, find ALL valid pairs of the remaining
#   two out-dihedral angles.
#
#   The forward PTU (ptu_2.py) computes theta via arccos which always
#   returns a positive value, and then uses two RBMs (rigid body modes)
#   giving +/- signs on theta. However, on the spherical triangle each
#   theta_i can independently be +/- (the dihedral of the spherical
#   triangle edge). Combined with the two RBM sign conventions this
#   yields up to 16 candidate solutions. By fixing one known output
#   dihedral we filter to only the geometrically consistent ones.
#
# ------------------------------------------------------------------------------

import numpy as np
from numpy import pi, cos, sin, arccos, real, clip, sign, multiply, power
import itertools

EPS = 1e-5
ANGLE_TOL = 1e-4  # tolerance for matching known dihedral angle


# ======================== rotation helpers (same as ptu_2.py) ==================

def Rx(theta):
    return np.asarray([
        [1, 0,          0          ],
        [0, cos(theta), -sin(theta)],
        [0, sin(theta),  cos(theta)],
    ])

def Rz(theta):
    return np.asarray([
        [cos(theta), -sin(theta), 0],
        [sin(theta),  cos(theta), 0],
        [0,           0,          1],
    ])

def transform_fold_forw(angle_z, angle_x):
    return np.matmul(Rz(angle_z), Rx(angle_x))

def transform_fold_rev(angle_x, angle_z):
    return np.matmul(Rx(angle_x), Rz(angle_z))


# ======================== unit angle helpers (same as ptu_2.py) ================

def compute_folded_unit(alpha_arr, rho):
    p_0 = np.array([1, 0, 0])
    alpha = list(alpha_arr[:-1])
    res = np.identity(3)
    for a, r in zip(alpha, rho):
        res = res.dot(transform_fold_forw(a, r))
    res = res.dot(Rz(alpha_arr[-1]))
    p_m = np.matmul(res, p_0)
    return arccos(clip(p_0.dot(p_m), -1., 1.))


def beta_delta(p_1, p_max, unit_angle, first_sector):
    gamma = arccos(clip(p_1.T.dot(p_max), -1., 1.))
    p_max_z = p_max[2] if not np.isclose(p_max[2], 0., rtol=1e-05, atol=1e-08) else 1e-05
    sgn = -np.sign(p_max_z)
    temp_enum = cos(gamma) - multiply(cos(first_sector), cos(unit_angle))
    temp_denom = multiply(sin(first_sector), sin(unit_angle))
    if abs(temp_denom) < EPS:
        temp_denom = sign(temp_denom) * EPS
    temp_denom = power(temp_denom, -1)
    temp = multiply(temp_enum, temp_denom)
    temp = clip(temp, -1., 1.)
    return sgn * arccos(temp)


def get_unit_angle(alpha_vec, rho_vec):
    """Compute U_j, beta_j, delta_j for one unit (same logic as ptu_2.py)."""
    p_j0 = np.array([1, 0, 0])
    m = len(alpha_vec)
    Mffw = np.identity(3)
    last_rot_z = Rz(alpha_vec[-1])

    if m > 1:
        for i in range(m - 1):
            forw = transform_fold_forw(alpha_vec[i], rho_vec[i])
            Mffw = Mffw.dot(forw)
        Mffw = Mffw.dot(last_rot_z)
        p_jm = Mffw.dot(p_j0)
        temp = clip(p_j0.dot(p_jm), -1., 1.)
        U_j = arccos(temp)
        if abs(U_j) < EPS:
            U_j = EPS
        p_j1 = np.matmul(Rz(alpha_vec[0]), p_j0)
        beta_j = beta_delta(p_j1, p_jm, U_j, alpha_vec[0])

        # reverse kinematics for delta
        Mffw = np.identity(3)
        Mffw = np.matmul(Rz(-alpha_vec[-1]), Mffw)
        alpha_vec_rev = list(alpha_vec[:-1])
        rho_vec_rev = list(rho_vec)
        alpha_vec_rev.reverse()
        rho_vec_rev.reverse()
        for i in range(m - 1):
            Mffw = np.matmul(Mffw, transform_fold_rev(-rho_vec_rev[i], -alpha_vec_rev[i]))
        p_j0_prime = Mffw.dot(p_j0)
        p_jm1_prime = np.matmul(Rz(-alpha_vec[-1]), p_j0)
        delta_j = beta_delta(p_jm1_prime, p_j0_prime, U_j, alpha_vec[-1])
    else:
        U_j = alpha_vec[0]
        beta_j = 0
        delta_j = 0

    return U_j, beta_j, delta_j


def get_theta_unsigned(U):
    """Return the three *positive* theta values (magnitudes)."""
    def theta(u_i, u_j, u_k):
        ratio = (cos(u_i) - cos(u_j) * cos(u_k)) / (sin(u_j) * sin(u_k))
        ratio = np.clip(ratio, -1., 1.)
        return arccos(ratio)

    t1 = theta(U[0], U[1], U[2])
    t2 = theta(U[1], U[0], U[2])
    t3 = theta(U[2], U[0], U[1])
    return [real(t1), real(t2), real(t3)]


def _wrap(angle):
    """Wrap angle to [-pi, pi]."""
    while angle > pi:
        angle -= 2 * pi
    while angle < -pi:
        angle += 2 * pi
    return angle


def _compute_phi(rbm_sign, beta, delta, theta_signed):
    """
    Compute phi given a sign convention.
    
    rbm_sign = +1 corresponds to RBM1:  phi = beta + pi - theta + delta
    rbm_sign = -1 corresponds to RBM2:  phi = beta + theta - pi + delta
    
    With signed theta, each theta_i can independently be +/- of the unsigned
    value, so we get:
        phi_i = beta_{i+2} + rbm_sign * (pi - theta_i) + delta_{i+1}
    which unifies both RBM into:
        RBM1: phi_i = beta_{(i+2)%3} + (pi - theta_i)   + delta_{(i+1)%3}
        RBM2: phi_i = beta_{(i+2)%3} + (theta_i - pi)    + delta_{(i+1)%3}
    """
    t = theta_signed
    b = beta
    d = delta
    if rbm_sign == 1:
        phi1 = b[2] + pi - t[0] + d[1]
        phi2 = b[0] + pi - t[1] + d[2]
        phi3 = b[1] + pi - t[2] + d[0]
    else:
        phi1 = b[2] + t[0] - pi + d[1]
        phi2 = b[0] + t[1] - pi + d[2]
        phi3 = b[1] + t[2] - pi + d[0]
    return [_wrap(phi1), _wrap(phi2), _wrap(phi3)]


def _normalize_angle(a):
    """Normalize angle: wrap to [-pi, pi] and map -pi to +pi."""
    a = _wrap(a)
    if abs(a + pi) < 1e-8:
        a = pi
    return a


def _angles_close(a, b, tol=ANGLE_TOL):
    """Check if two angles are close, accounting for wrapping and ±π."""
    diff = abs(_wrap(a - b))
    return diff < tol


def _is_valid_theta_combination(theta_unsigned, signs):
    """
    Check that the signed thetas still satisfy the spherical triangle
    constraints (all positive unit angles means thetas must form valid 
    triangle on the sphere).
    Since theta values come from arccos (always positive), the sign
    combinations are constrained: on a spherical triangle the angles 
    must satisfy 0 < theta_i < pi and theta_1 + theta_2 + theta_3 > pi.
    But we allow all sign combos and filter by the known dihedral constraint.
    """
    return True  # We filter by matching the known dihedral instead


def ptu_reverse(sector_angles, list_in_dihedral_angles, known_phi_index,
                known_phi_value, tol=ANGLE_TOL):
    """
    Reverse PTU: given all sector angles, all in-dihedral angles, and one
    known output dihedral angle, enumerate ALL valid solutions for the
    remaining two output dihedral angles.

    Parameters
    ----------
    sector_angles : list of 3 lists of floats
        Sector angles for each of the 3 units.
    list_in_dihedral_angles : list of 3 lists of floats
        In-dihedral angles for each of the 3 units.
    known_phi_index : int (0, 1, or 2)
        Which output dihedral is known (0-indexed, BEFORE the roll applied
        in calc_ptu_2). This is the "raw" phi index:
          raw index 0 → φ₁ (between unit 3-end and unit 2-start)
          raw index 1 → φ₂ (between unit 1-end and unit 3-start)
          raw index 2 → φ₃ (between unit 2-end and unit 1-start)
    known_phi_value : float
        The known dihedral angle value in radians.
    tol : float
        Tolerance for matching the known angle.

    Returns
    -------
    solutions : list of list[float]
        Each element is [phi1, phi2, phi3] (raw ordering, before roll).
        Only solutions where phi[known_phi_index] matches known_phi_value
        are returned.
    """

    # Step 1: compute U, beta, delta for each unit
    U = []
    betas = []
    deltas = []
    for j in range(3):
        alpha_vec = list(sector_angles[j])
        rho_vec = list(list_in_dihedral_angles[j])
        U_j, beta_j, delta_j = get_unit_angle(alpha_vec, rho_vec)
        U.append(U_j)
        betas.append(beta_j)
        deltas.append(delta_j)

    # Triangle inequality check
    U_sorted = sorted(U)
    if U_sorted[0] + U_sorted[1] < U_sorted[2] - EPS:
        return []

    # Step 2: get unsigned thetas
    theta_mag = get_theta_unsigned(U)

    # Step 3: enumerate all 2^3 sign combos × 2 RBMs = 16 candidates
    solutions = []
    seen = set()

    for signs in itertools.product([1, -1], repeat=3):
        theta_signed = [s * t for s, t in zip(signs, theta_mag)]
        for rbm in [1, -1]:
            phi = _compute_phi(rbm, betas, deltas, theta_signed)

            # Check if the known dihedral matches
            if _angles_close(phi[known_phi_index], known_phi_value, tol):
                # Normalize and dedup (treat π ≡ -π)
                phi_norm = [_normalize_angle(p) for p in phi]
                key = tuple(round(p, 4) for p in phi_norm)
                if key not in seen:
                    seen.add(key)
                    solutions.append(phi_norm)

    return solutions


def calc_ptu_reverse(sector_angles, list_in_dihedral_angles,
                     known_phi_index, known_phi_value, tol=ANGLE_TOL):
    """
    User-friendly wrapper that applies the same roll convention as calc_ptu_2.

    Parameters
    ----------
    sector_angles : list of 3 lists
    list_in_dihedral_angles : list of 3 lists
    known_phi_index : int (0, 1, or 2)
        Index of the known output dihedral in the *rolled* ordering
        (the same indexing used in the output of calc_ptu_2).
        In calc_ptu_2, the output is rolled as: [phi3, phi1, phi2]
        So rolled index 0 → raw phi3 (index 2)
           rolled index 1 → raw phi1 (index 0)
           rolled index 2 → raw phi2 (index 1)
    known_phi_value : float
        The known dihedral angle value.
    tol : float
        Matching tolerance.

    Returns
    -------
    solutions : list of [phi_rolled_0, phi_rolled_1, phi_rolled_2]
        All valid output dihedral angle triples in the rolled ordering.
    """
    # Map rolled index back to raw index
    rolled_to_raw = {0: 2, 1: 0, 2: 1}
    raw_index = rolled_to_raw[known_phi_index]

    raw_solutions = ptu_reverse(
        sector_angles, list_in_dihedral_angles,
        raw_index, known_phi_value, tol
    )

    # Apply roll: raw [phi1, phi2, phi3] → rolled [phi3, phi1, phi2]
    rolled_solutions = []
    for sol in raw_solutions:
        rolled = [float(sol[2]), float(sol[0]), float(sol[1])]
        rolled_solutions.append(rolled)

    return rolled_solutions


# ================================ demo / test =================================

def _fmt(phi_list):
    """Pretty-print angles as multiples of π."""
    def _to_pi(x):
        r = x / pi
        if abs(r - round(r, 1)) < 0.01:
            r = round(r, 1)
        if r == 1.0:
            return "π"
        elif r == -1.0:
            return "-π"
        elif r == 0.0:
            return "0"
        elif r == 0.5:
            return "π/2"
        elif r == -0.5:
            return "-π/2"
        else:
            return f"{r:.4f}π"
    return "[" + ", ".join(_to_pi(x) for x in phi_list) + "]"


if __name__ == "__main__":
    from ptu import calc_ptu

    sector_angles = [[pi/2, pi/2], [pi/2], [pi/2]]
    in_dihedral   = [[-pi/2], [], []]

    print("=" * 60)
    print("Forward PTU (calc_ptu_2):")
    result = calc_ptu(sector_angles, in_dihedral)
    if result[0]:
        print(f"  M1 (rolled): {_fmt(result[1])}")
        print(f"  M2 (rolled): {_fmt(result[2])}")
    else:
        print("  No valid solution from forward PTU.")

    # Test 1: know phi at rolled index 0 = pi, find the others
    for known_idx in range(3):
        for known_val in [result[1][known_idx], result[2][known_idx]]:
            print()
            print("=" * 60)
            print(f"Reverse PTU: known_phi[{known_idx}] = {_fmt([known_val])}")
            solutions = calc_ptu_reverse(
                sector_angles, in_dihedral,
                known_phi_index=known_idx,
                known_phi_value=known_val
            )
            for i, sol in enumerate(solutions):
                print(f"  Solution {i+1}: {_fmt(sol)}")
            if not solutions:
                print("  (no solutions found)")
