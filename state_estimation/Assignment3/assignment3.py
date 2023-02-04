import numpy as np
import scipy.io as scp
import matplotlib.pyplot as plt
import pylgmath.so3.operations as sop
import pylgmath.se3.operations as sep
from scipy.linalg import expm, logm

# global data
data = scp.loadmat('/home/aniket/Desktop/Courses_Fall/AER1513/Assignments/StateEstimation_AER1513/Assignment3/dataset3.mat')
# seperate out
theta_vk_i_gt = data['theta_vk_i']
r_i_vk_i_gt = data["r_i_vk_i"]
t = data["t"]
w_vk_vk_i = data["w_vk_vk_i"]
w_var = data["w_var"]
v_vk_vk_i = data["v_vk_vk_i"]
v_var = data["v_var"]
rho_i_pj_i = data["rho_i_pj_i"]
y_k_j = data["y_k_j"]
y_var = data["y_var"]
C_c_v = data["C_c_v"]
rho_v_c_v = data["rho_v_c_v"]
fu = data["fu"][0][0]
fv = data["fv"][0][0]
cu = data["cu"][0][0]
cv = data["cv"][0][0]
b = data["b"][0][0]

# time start - end
k1 = 1215
k2 = 1715

# initialization
def initialization():
    init_params = {}

    # number of timesteps
    init_params['T'] = data['t'].shape[1]

    # Q, R
    init_params['Q'] = np.zeros((6, 6))
    init_params['R'] = np.diag(y_var**2)   
    
    # generalized velocity
    init_params['gen_v_vk_vk_i'] = -np.vstack((v_vk_vk_i, w_vk_vk_i))
    return init_params

def visu(Top):
    # back to intertial frame
    Tiv = np.zeros(Top.shape)
    for k in range(Top.shape[2]):
        Tiv[:, :, k] = np.linalg.inv(Top[:, :, k])
    fig = plt.figure(1)
    ax = plt.axes(projection='3d')
    ax.plot3D(Tiv[0, 3, :], Tiv[1, 3, :], Tiv[2, 3, :], color='b')
    ax.plot3D(r_i_vk_i_gt[0, :], r_i_vk_i_gt[1, :], r_i_vk_i_gt[2, :], color='r')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()
    

def psi_C(psi):
    psi_norm = np.linalg.norm(psi)
    C = np.cos(psi_norm) * np.identity(3) + (1 - np.cos(psi_norm)) * (psi/psi_norm) @ ((psi/psi_norm).T) - np.sin(psi_norm) * sop.hat(psi/psi_norm)    
    
    return C

def circle_dot(V):
    dot = np.zeros((4, 6))
    dot[:3, :3] = np.identity(3)
    dot[:3, 3:] = -sop.hat(V[:3])

    return dot

def getG(Tcv, Tvi, z_op, rho_i_pj_i_homo): # z_op is the first non linearity (extrinsic transformation)
    x = z_op[0, 0]
    y = z_op[1, 0]
    z = z_op[2, 0]
    
    # S = del_S/del_Z
    S = np.array([[fu/z, 0, -fu*x/z**2], [0, fv/z, -fv*y/z**2], [fu/z, 0, -fu*(x-b)/z**2], [0, fv/z, -fv*y/z**2]])

    # Z
    Z = Tcv @ circle_dot(Tvi @ rho_i_pj_i_homo)

    # D
    D = np.zeros((3, 4))
    D[:, :3] = np.identity(3)

    # G
    G = S @ D @ Z

    return G

def errors_op_motion(Top_k, Top_kminus, gen_v_vk_vk_i, delta_t, flag):
    # linearized error terms
    # motion model - at k1 and rest
    if (flag == 0): # at k1
        e_v = sep.tran2vec(Top_kminus @ np.linalg.inv(Top_k)) 
    else:
        burg = expm(delta_t * sep.hat(gen_v_vk_vk_i.reshape(6, 1)))
        e_v = sep.tran2vec(burg @ Top_kminus @ np.linalg.inv(Top_k)) 
        
    return e_v

def errors_op_obs(Top, y_k):
    # linearized error terms for observations
    # get the Tcv - vehicle to camera
    Tcv = np.zeros((4, 4))
    Tcv[:3, :3] = C_c_v
    Tcv[:3, 3] = rho_v_c_v[0]
    Tcv[3, 3] = 1

    # get the Tvi - inertial frame to vehicle using the Top
    Tvi = np.zeros((4, 4))
    Tvi[:3, :3] = Top[:3, :3]
    Tvi[:3, 3] = -Top[:3, :3] @ Top[:3, 3]
    Tvi[3, 3] = 1

    D_T = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])

    # error initialization
    e_y = np.zeros((4, 1))

    # initialize G
    G_k = np.zeros((4, 6))

    for j in range(rho_i_pj_i.shape[1]):
        
        # write the point coordinates in homogenous system
        rho_i_pj_i_homo = np.ones((4, 1))
        rho_i_pj_i_homo[:3] = rho_i_pj_i[:, j].reshape((3, 1))

        # 1st non linearity - extrinsic transformation
        z_op = (D_T @ Tcv @ Tvi @ rho_i_pj_i_homo) 
        
        # 2nd non linearity - intrinsic transformation
        g = (1 / z_op[2][0]) * (np.array([[fu * z_op[0][0]], [fv * z_op[1][0]], [fu * (z_op[0][0] - b)], [fv * z_op[1][0]]])) + np.array([[cu], [cv], [cu], [cv]])
        if np.all(y_k[:, j] == -1):
            continue
        else:
            e = y_k[:, j].reshape(4, 1) - g.reshape(4, 1)
        
        e_y = np.vstack([e_y, e])

        G = getG(Tcv, Tvi, z_op, rho_i_pj_i_homo)
        G_k = np.vstack([G_k, G])

    # remove the first filler
    e_y = e_y[4:, 0]
    G_k = G_k[4:]

    return e_y, G_k

def setup(gen_v_vk_vk_i):
    flag = 0

    # params
    params = initialization()

    # initial estimate at k = 1215
    theta_gt = theta_vk_i_gt[:, k1].reshape(3, 1)
    C_0_vk_i = psi_C(theta_gt)
    r_0_i_vk_i = r_i_vk_i_gt[:, k1]

    T0_hat_gt = np.zeros((4, 4))
    T0_hat_gt[:3, :3] = C_0_vk_i
    T0_hat_gt[:3, 3] = -C_0_vk_i @ r_0_i_vk_i
    T0_hat_gt[3, 3] = 1
    
    # initialize Top
    Top = np.zeros((4, 4, k2-k1))
    Top[:, :, 0] = T0_hat_gt
    # initialize errors 
    e_v = np.zeros((6, 1, k2-k1))
    e_y = np.zeros((1, 1))
    
    # initialize Fk
    F_k = np.zeros((6, 6, k2 - k1))
    # initialize Gk
    G_k = np.zeros((80, 6, k2 - k1))

    # error at time k1
    e_v[:, :, 0] = errors_op_motion(np.identity(4), Top[:, :, 0], gen_v_vk_vk_i[:, k1], 0, flag)
    flag = 1

    for k in range(1, k2 - k1):
        # delta time
        delta_t = t[:, k1 + k] - t[:, k1 + k -1]

        # variance values with variable sampling
        params['Q'][:3, :3] = np.diag(v_var*delta_t**2)
        params['Q'][3:, 3:] = np.diag(w_var*delta_t**2)

        # dead reckon using the motion model
        Top[:, :, k] = sep.vec2tran(delta_t * (gen_v_vk_vk_i[:, k1 + k].reshape(6, 1))) @ Top[:, :, k-1]

        # error terms
        # motion model
        e_v[:, :, k] = errors_op_motion(Top[:, :, k], Top[:, :, k-1], gen_v_vk_vk_i[:, k1 + k], delta_t, flag)

        # observation model
        e_y_obs, G_k[:, :, k] = errors_op_obs(Top[:, :, k], y_k_j[:, k, :])
        print(e_y_obs)
        e_y = np.vstack([e_y, e_y_obs])

        # Fk
        F_k[:, :, k] = sep.tranAd(Top[:, :, k] @ np.linalg.inv(Top[:, :, k-1]))

    # stacked error
    print(e_y.shape)
    e_y = e_y[:, :, 1:]
    e_op = np.vstack((e_v[:, 0, :], e_y))
    
    # visualize the dead reckoning
    # visu(Top)

    return e_op, F_k, G_k

def gn_full(gen_v_vk_vk_i):
    e_op, F_k, G_k = setup(gen_v_vk_vk_i)
    
    # construct the H matrix
    k = k2 - k1
    H = np.zeros((2*(k + 1), k + 1))
    I = np.identity(k+1)
    # H[:k+2, :] = 

if __name__ == "__main__":
    params = initialization()
    # Top = setup(params['gen_v_vk_vk_i'])
    gn_full(params['gen_v_vk_vk_i'])
    

