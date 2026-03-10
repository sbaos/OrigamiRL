# ------------------------------------------------------------------------------
#
#   PTU kinematics -- implementation of the PTU kinematic model
#   as described in paper:
#
#   Luca Zimmermann et al.:
#   "Conditions for Rigid and Flat Foldability of Degree-n Vertices in Origami"
#
#   author: Jeremia Geiger, Master Thesis: RL for Rigid Origami
#   under supervision of Oliver Richter and Karolis Martinkus
#   ETH Zurich, DISCO lab, December 2020
#
# ------------------------------------------------------------------------------


# ------------------------------- dependencies ---------------------------------

import numpy as np
from numpy import *

EPS = 1e-5

# ------------------------------------------------------------------------------
#
#   Rotation Matrix and forward and backward transformation definitions
#   for single vertex kinematics computation
#
# ------------------------------------------------------------------------------

# === rotation matrix about x-axis
def Rx(theta):
  return np.asarray([[ 1, 0           , 0     ],
                   [ 0, cos(theta),-sin(theta)],
                   [ 0, sin(theta), cos(theta)]])

# === rotation matrix about z-axis
def Rz(theta):
  return np.asarray([[ cos(theta), -sin(theta), 0 ],
                   [ sin(theta), cos(theta) , 0 ],
                   [ 0         , 0        , 1 ]])

# === forward rotation, positive angles
def transform_fold_forw(angle_z,angle_x):
    return np.matmul(Rz(angle_z),Rx(angle_x))

# === reverse rotations, feed in negative angle values only!
def transform_fold_rev(angle_x,angle_z):
    return np.matmul(Rx(angle_x),Rz(angle_z))


def compute_folded_unit(alpha_arr,rho):
    # === the first edge marks the x-axis of the SV-coordinate frame
    p_0 = np.array([1,0,0])
    new_sector = alpha_arr[-1]
    alpha = list(alpha_arr[:-1])
    res = np.identity(3)
    all_angles = zip(alpha,rho)
    for a,r in all_angles:
        res = res.dot(transform_fold_forw(a,r))
    res = res.dot(Rz(new_sector))
    # === final 3D-point, marks the upper end of first unit
    p_m = np.matmul(res,p_0)
    # === return folded unit angle
    return np.arccos(np.clip(p_0.dot(p_m), -1., 1.))


def get_unit_angle(alpha_vec,rho_vec):
    # --- actual implementation of the kinematic model from PTU paper
    # --------------------------------------------------------------------------
    #
    #   takes:      alpha_vec: list of sector angles alpha
    #               rho_vec: list of dihedral angles rho
    #
    #   returns:    U_j: unit angle U_j
    #               beta_j: start angle beta_j of unit
    #               delta_j: end angle delta_j of unit
    #
    # --------------------------------------------------------------------------

    if np.any(np.isnan(alpha_vec)):
        print('alpha nan', alpha_vec)
        raise NotImplementedError

    if np.any(np.isnan(rho_vec)):
        print('rho nan', rho_vec)
        raise NotImplementedError

    # starting point
    p_j0 = np.array([1,0,0])
    # number of sectors in unit
    m = len(alpha_vec)
    # rotation matrix placeholder
    Mffw = np.identity(3)
    # last unit-sector rotation
    last_rot_z = Rz(alpha_vec[-1])

    if m>1:
        # === case for non-rigid units
        # === forward kinematics for start angle beta
        for i in range(m-1):
            forw = transform_fold_forw(alpha_vec[i],rho_vec[i])
            Mffw = Mffw.dot(forw)
        Mffw = Mffw.dot(last_rot_z)
        p_jm = Mffw.dot(p_j0)
        if abs(p_j0.dot(p_jm)) > 1.:
            print('pjm', p_jm)
            print('pj0', p_j0)
        temp = clip(p_j0.dot(p_jm), -1., 1.)
        U_j = arccos(temp)
        if np.isnan(U_j):
            print('uj nan')
            raise NotImplementedError
        if abs(U_j)<EPS:
            U_j = EPS
        p_j1 = np.matmul(Rz(alpha_vec[0]),p_j0)
        # start angle beta
        beta_j = beta_delta(p_j1,p_jm,U_j,alpha_vec[0])

        # === reverse kinematics for end angle delta
        Mffw = eye(3)
        p_jm_prime = p_j0.copy()
        Mffw = np.matmul(Rz(-alpha_vec[-1]),Mffw)
        alpha_vec_rev = alpha_vec.copy()
        alpha_vec_rev = alpha_vec_rev[:-1]
        rho_vec_rev = rho_vec.copy()
        alpha_vec_rev.reverse()
        rho_vec_rev.reverse()

        for i in range(m-1):
            Mffw = np.matmul(Mffw,transform_fold_rev(-rho_vec_rev[i],-alpha_vec_rev[i]))

        p_j0_prime = Mffw.dot(p_j0)
        p_jm1_prime = np.matmul(Rz(-alpha_vec[-1]),p_jm_prime)
        # end angle delta
        delta_j = beta_delta(p_jm1_prime,p_j0_prime,U_j,alpha_vec[-1])

    else:
        # --- case for rigid units only
        Mffw = last_rot_z
        p_jm = np.matmul(Mffw,p_j0)
        U_j = alpha_vec[0]
        gamma_sj = 0
        gamma_ej = 0

        # === start angle beta and end angle delta equal zero for 'rigid' units
        beta_j = 0
        delta_j = 0

    return [real(U_j),beta_j,delta_j]


def beta_delta(p_1,p_max,unit_angle,first_sector):
    # --------------------------------------------------------------------------
    #
    #   takes:      p_1: first next edge-point starting from a bdry unit point
    #               p_max: the last and max. unit point
    #               unit_angle: unit angle U_j(psi)
    #               first_sector: first sector angle alpha_i
    #
    #   returns:    res_angle: the start angle beta or end angle delta,
    #                   depending on the input points
    #
    # --------------------------------------------------------------------------
    gamma = arccos(clip(p_1.T.dot(p_max), -1., 1.))
    if np.isclose(
        p_max[2],0.,rtol=1e-05, atol=1e-08, equal_nan=False):
        p_max_z = 1e-05
    else:
        p_max_z = p_max[2]
    sgn = -np.sign(p_max_z)
    temp_enum = cos(gamma)-multiply(cos(first_sector),cos(unit_angle))
    temp_denom = multiply(sin(first_sector), sin(unit_angle))
    if abs(temp_denom) < EPS:
        temp_denom = sign(temp_denom) * EPS
    temp_denom = power(temp_denom,-1)
    temp = multiply(temp_enum,temp_denom)
    temp = clip(temp, -1., 1.)
    res_angle = sgn*arccos(temp)
    return res_angle


def get_theta(U):
    # --------------------------------------------------------------------------
    #
    #   takes:      U: a list of three unit angles U_j(psi), as a function of
    #                   the global driving angle psi
    #
    #   returns:    [theta_1,theta_2,theta_3]: a np array of three angles theta(psi)
    #
    # --------------------------------------------------------------------------

    def theta(u_i,u_j,u_k):
        ratio = multiply((cos(u_i)-multiply(cos(u_j),cos(u_k))),power(multiply(sin(u_j),sin(u_k)),-1))
        ratio = np.clip(ratio, -1., 1.)
        return arccos(ratio)
    theta_1 = theta(U[0],U[1],U[2])
    theta_2 = theta(U[1],U[0],U[2])
    theta_3 = theta(U[2],U[0],U[1])

    return np.array([real(theta_1),real(theta_2),real(theta_3)])


def get_phi(rbm,beta,delta,theta):
    # --------------------------------------------------------------------------
    #
    #   takes:      rbm: the rigid body mode
    #               beta: list of starting angles beta
    #               delta: list of end angles delta
    #               theta: list of theta angles
    #
    #   returns:    phi: list of three outgoing edge dihedral angles phi
    #
    # --------------------------------------------------------------------------

    if rbm == 1:
        phi1 = beta[2] + pi - theta[0] + delta[1]
        phi2 = beta[0] + pi - theta[1] + delta[2]
        phi3 = beta[1] + pi - theta[2] + delta[0]
    else:
        phi1 = beta[2] + theta[0] - pi + delta[1]
        phi2 = beta[0] + theta[1] - pi + delta[2]
        phi3 = beta[1] + theta[2] - pi + delta[0]

    return np.array([real(phi1),real(phi2),real(phi3)])


# ------------------------------------------------------------------------------
#
#   calc_ptu_2: wrapper function matching the interface used in gen_grid.py
#
#   takes:      sector_angles: list of 3 lists of sector angles (one per unit)
#               list_in_dihedral_angles: list of 3 lists of dihedral angles
#
#   returns:    [flattened_sector_angles, M1_rolled, M2_rolled]
#               or [[],[],[]] on failure
#
# ------------------------------------------------------------------------------

def calc_ptu_2(sector_angles, list_in_dihedral_angles):
    if not sector_angles or not list_in_dihedral_angles:
        return [[], [], []]

    U = []
    beta = []
    delta = []

    for j in range(3):
        alpha_vec = list(sector_angles[j])
        rho_vec = list(list_in_dihedral_angles[j])
        U_j, beta_j, delta_j = get_unit_angle(alpha_vec, rho_vec)
        U.append(U_j)
        beta.append(beta_j)
        delta.append(delta_j)

    # Triangle inequality check BEFORE computing theta/phi
    U_sorted = sorted(U)
    if U_sorted[0] + U_sorted[1] < U_sorted[2]:
        return [[], [], []]

    theta = get_theta(U)
    if np.any(np.isnan(theta)):
        return [[], [], []]

    # RBM 1
    M1 = get_phi(1, beta, delta, theta)
    M1 = np.where(M1 < -pi, 2*pi + M1, M1)
    M1 = np.where(M1 > pi, -2*pi + M1, M1)

    # RBM 2
    M2 = get_phi(2, beta, delta, theta)
    M2 = np.where(M2 < -pi, 2*pi + M2, M2)
    M2 = np.where(M2 > pi, -2*pi + M2, M2)

    M1 = M1.tolist()
    M2 = M2.tolist()

    if not M1 or not M2:
        return [[], [], []]

    # Roll by 1 to match output ordering convention
    M1 = [float(M1[2]), float(M1[0]), float(M1[1])]
    M2 = [float(M2[2]), float(M2[0]), float(M2[1])]

    t = [x for y in sector_angles for x in y]

    return [t, M1, M2]
