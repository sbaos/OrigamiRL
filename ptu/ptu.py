import numpy as np
import math
from enum import Enum
EPS = 1e-5

class EdgeType(Enum):
    Horizontal = "Horizontal"
    Vertical = "Vertical"
    Diagonal_left = "Diagonal\\"
    Diagonal_right = "Diagonal/"

class Point:
    def __init__(self, x, y, z = 0, point_idx = -1, point_root = [], children = [], is_actived = False, is_border = False):
        self.position = np.array([x,y,z])
        self.in_diheral_angles = []
        self.out_diheral_angles = []
        self.point_root = point_root
        self.children = children
        self.is_actived = is_actived
        self.point_idx = point_idx
        self.is_border = is_border
        self.level = 1

    def __repr__(self):
        return f"Point({self.point_idx})"

    def __str__(self):
        parent_ids = [p.point_idx for p in self.point_root] if self.point_root else []
        children_ids = [c.point_idx for c in self.children] if self.children else []
        return f"Point({self.point_idx}, pos={self.position}, parents={parent_ids}, children={children_ids})"

    def add_children(self,other_point,value):
        if other_point in self.children or other_point in self.point_root or other_point is None:
            return
        self.children.append(other_point)
        self.out_diheral_angles.append(value)

    def add_parent(self,point,value):
        if point in self.point_root or point in self.children or point is None:
            return
        self.point_root.append(point)
        self.in_diheral_angles.append(value)

    def update_level(self,new_level):
        if self.level <= new_level - 2 and self.is_actived:
            self.level = new_level - 1
        else:
            self.level = new_level

    def clone(self):
        point = Point(self.position[0], self.position[1], self.position[2], self.point_idx, self.point_root, self.children, self.is_actived)
        point.point_root = [i for i in self.point_root]
        point.children = [i for i in self.children]
        point.in_diheral_angles = [i for i in self.in_diheral_angles]
        point.out_diheral_angles = [i for i in self.out_diheral_angles]
        return point
class Edge:
    def __init__(self, u: Point, v: Point, value, attributes = [], edge_type = "", line_idx = -1, is_soft = False):
        self.u = u
        self.v = v
        self.value = value
        self.attributes = attributes
        self.edge_type = edge_type
        self.direction = "u -> v"
        self.line_idx = line_idx
        self.is_soft = is_soft
    
    def __str__(self):
        return f"Edge(u={self.u}, v={self.v}, value={self.value}, attributes={self.attributes}, edge_type={self.edge_type})"

    def __repr__(self):
        return str(self)

    def clone(self):
        return Edge(self.u, self.v, self.value, self.attributes, self.edge_type)


def Rx(theta):
  return np.asarray([[ 1, 0           , 0     ],
                   [ 0, np.cos(theta),-np.sin(theta)],
                   [ 0, np.sin(theta), np.cos(theta)]])

def Rz(theta):
  return np.asarray([[ np.cos(theta), -np.sin(theta), 0 ],
                   [ np.sin(theta), np.cos(theta) , 0 ],
                   [ 0         , 0        , 1 ]])

def calc_p_j_m(p0,sector_angles,list_in_diheral_angles):
    res = np.identity(3)
    m = len(sector_angles)
    for i in range(m-1):
        res = np.matmul(res,transform_fold_forw(sector_angles[i], list_in_diheral_angles[i]))
    res = res.dot(Rz(sector_angles[-1]))
    p_m = np.matmul(res,p0)
    return p_m

def transform_fold_forw(angle_z,angle_x):
    return np.matmul(Rz(angle_z),Rx(angle_x))

def transform_fold_rev(angle_x,angle_z):
    return np.matmul(Rx(angle_x),Rz(angle_z))

def beta_delta(p1, pm, u_j, alpha_j_1):
    gamma = np.arccos(np.clip(p1.dot(pm), -1., 1.))
    if np.isclose(pm[2],0., rtol = 1e-05, atol = 1e-8, equal_nan=False):
        pm_z = 1e-05
    else:
        pm_z = pm[2]
    sgn = -np.sign(pm_z)
    numerator = np.cos(gamma) - np.cos(alpha_j_1)*np.cos(u_j)
    denominator = np.sin(alpha_j_1)*np.sin(u_j)
    if abs(denominator) < EPS:
        denominator = np.sign(denominator)*EPS
    
    temp = np.clip(numerator/denominator,-1.,1.)
    return sgn*np.arccos(temp)

def calculate_theta(u1 ,u2 ,u3):
    def theta(u1,u2,u3):
       numerator = np.cos(u1) - np.cos(u2)*np.cos(u3)
       denominator = np.sin(u2)*np.sin(u3)
       argcos = np.clip(numerator/denominator,-1.,1)
       return np.acos(argcos)

    theta1 = theta(u1,u2,u3)
    theta2 = theta(u2,u1,u3)
    theta3 = theta(u3,u1,u2)
    return theta1, theta2, theta3
    
def compute_folded_unit(alpha_arr,rho):
   p_0 = np.array([1,0,0])
   res = np.identity(3)    

   for i in range(len(alpha_arr)-1):
        res = np.matmul(res,transform_fold_forw(alpha_arr[i], rho[i]))
   
   res = res.dot(Rz(alpha_arr[-1]))
   p_m = np.matmul(res,p_0)

   return np.arccos(np.clip(p_0.dot(p_m), -1., 1.))

def ptu(sector_angles:list[list[float]],list_in_diheral_angles:list[list[float]]) -> tuple[list[float],list[float]]:
    
    def calc_beta_delta(index):
        p0 = np.array([1,0,0])
        res = np.identity(3)    

        m = len(sector_angles[index])
        if m > 1:
            for i in range(len(sector_angles[index])-1):
                res = np.matmul(res,transform_fold_forw(sector_angles[index][i], list_in_diheral_angles[index][i]))
            res = res.dot(Rz(sector_angles[index][-1]))
            p_m = np.matmul(res,p0)
            p_1 = np.dot(Rz(sector_angles[index][0]),p0)
            u = compute_folded_unit(sector_angles[index],list_in_diheral_angles[index])
            beta = beta_delta(p_1,p_m,u,sector_angles[index][0])

            temp_sector_angles = sector_angles[index].copy()
            temp_sector_angles.reverse()
            temp_list_in_diheral_angles = list_in_diheral_angles[index].copy()
            temp_list_in_diheral_angles.reverse()
            p_j_0_revert = calc_p_j_m(p0,temp_sector_angles,temp_list_in_diheral_angles)
            p_jm1_revert = np.dot(Rz(temp_sector_angles[0]),p0)
            u = compute_folded_unit(temp_sector_angles,temp_list_in_diheral_angles)
            delta = beta_delta(p_jm1_revert, p_j_0_revert,u,temp_sector_angles[0])
            return beta,delta
        else:
            m_revert = Rz(sector_angles[index][-1])
            u = sector_angles[index][0]
            beta = 0
            delta = 0
            return beta,delta
    u1 = compute_folded_unit(sector_angles[0],list_in_diheral_angles[0])
    u2 = compute_folded_unit(sector_angles[1],list_in_diheral_angles[1])
    u3 = compute_folded_unit(sector_angles[2],list_in_diheral_angles[2])

    arr_u = np.array([u1,u2,u3])
    arr_u.sort()
    
    beta1,delta1 = calc_beta_delta(0)
    beta2,delta2 = calc_beta_delta(1)
    beta3,delta3 = calc_beta_delta(2)

    theta1, theta2, theta3 = calculate_theta(u1,u2,u3)

    phi1 = beta3 + np.pi - theta1 + delta2
    phi2 = beta1 + np.pi - theta2 + delta3
    phi3 = beta2 + np.pi - theta3 + delta1

    if phi1 > np.pi:
        phi1 -= 2*np.pi
    elif phi1 < -np.pi:
        phi1 += 2*np.pi

    if phi2 > np.pi:
        phi2 -= 2*np.pi
    elif phi2 < -np.pi:
        phi2 += 2*np.pi
    
    if phi3 > np.pi:
        phi3 -= 2*np.pi
    elif phi3 < -np.pi:
        phi3 += 2*np.pi

    M1:list[float] = [phi1,phi2,phi3]

    phi1 = beta3 + theta1 - np.pi + delta2
    phi2 = beta1 + theta2 - np.pi + delta3
    phi3 = beta2 + theta3 - np.pi + delta1

    if phi1 > np.pi:
        phi1 -= 2*np.pi
    elif phi1 < -np.pi:
        phi1 += 2*np.pi

    if phi2 > np.pi:
        phi2 -= 2*np.pi
    elif phi2 < -np.pi:
        phi2 += 2*np.pi
    
    if phi3 > np.pi:
        phi3 -= 2*np.pi
    elif phi3 < -np.pi:
        phi3 += 2*np.pi

    if (arr_u[0]+arr_u[1] - arr_u[2] > 0.001):
        pass
    else:
        return [],[]
    
    M2:list[float] = [phi1,phi2,phi3]
    return M1,M2


def calc_ptu(sector_angles:list[list[float]],list_in_diheral_angles:list[list[float]]) -> list[list[float]]:
   if (not sector_angles or not list_in_diheral_angles):
       return [[],[],[]]

   M1, M2 = ptu(sector_angles,list_in_diheral_angles)

   if M1 == [] or M2 == []:
       return [[],[],[]]
   M1 = [float(M1[2]),float(M1[0]),float(M1[1])]
   M2 = [float(M2[2]),float(M2[0]),float(M2[1])]
   t = [x for y in sector_angles for x in y]

   M1 = [0 if abs(angle) <=0.1 else angle for angle in M1]
   M2 = [0 if abs(angle) <=0.1 else angle for angle in M2]
   return [t, M1,M2]

