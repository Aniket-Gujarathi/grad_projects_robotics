import scipy.io as scp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

# global data 

data = scp.loadmat('dataset2.mat')
#  seperate out
t = data['t']
t = t.reshape((12609,1))
x_true = data['x_true']
y_true = data['y_true']
th_true = data['th_true']
true_valid = data['true_valid']
l = data['l']
r = data['r']
r_var = data['r_var']
b = data['b']
b_var = data['b_var']
v = data['v']
v_var = data['v_var']
om = data['om']
om_var = data['om_var']
d = data['d']

# ask for rmax
val = input("r_max: ")
rmax = val
# CRLB
CRLB = input('CRLB (y/n): ')


# visualization script
def plotting(x_hat, sigx, sigy, sigth):
    # plot trajectory
    plt.figure(0)
    plt.plot(x_hat[:, 0], x_hat[:, 1], 'b', label='Estimated')
    plt.plot(x_true, y_true, 'y', label='Ground Truth')
    plt.suptitle("Estimated Robot Trajectory wrt Ground Truth")
    plt.title("rmax = {}m".format(rmax))
    plt.xlabel("x(m)")
    plt.ylabel("y(m)")
    plt.legend()
    # plt.savefig("/home/aniket/Desktop/Courses_Fall/AER1513/Assignments/StateEstimation_AER1513/assignment2/Media/TrajCRLB_poorinit_{}.jpeg".format(rmax))

    # error plots
    ex = (x_hat[:, 0].reshape((x_true.shape)) - x_true)
    ey = (x_hat[:, 1].reshape(y_true.shape) - y_true)
    etheta = (x_hat[:, 2].reshape(th_true.shape) - th_true)
    etheta = [wrap(x) for x in etheta]
    print(np.mean(ex),  np.mean(ey), np.mean(etheta))
    
    plt.figure(1)
    plt.plot(t[1:], ex[1:], 'r', label='Error')
    plt.plot(t[1:], np.sqrt(sigx[1:])*3, 'b', label="3-sigma bound")
    plt.plot(t[1:], -np.sqrt(sigx[1:])*3, 'b')
    plt.suptitle('Error Plots')
    plt.title('rmax = {}m'.format(rmax))
    plt.xlabel('timesteps')
    plt.ylabel('Error in x(m)')
    plt.legend()
    # plt.savefig("/home/aniket/Desktop/Courses_Fall/AER1513/Assignments/StateEstimation_AER1513/assignment2/Media/ErrorX_poorinit_{}.jpeg".format(rmax))
    
    plt.figure(2)
    plt.plot(t[1:], ey[1:], 'r', label='Error')
    plt.plot(t[1:], np.sqrt(sigy[1:])*3, 'b', label="3-sigma bound")
    plt.plot(t[1:], -np.sqrt(sigy[1:])*3, 'b')
    plt.suptitle('Error Plots')
    plt.title('rmax = {}m'.format(rmax))
    plt.xlabel('timesteps')
    plt.ylabel('Error in y(m)')
    plt.legend()
    # plt.savefig("/home/aniket/Desktop/Courses_Fall/AER1513/Assignments/StateEstimation_AER1513/assignment2/Media/ErrorY_poorinit_{}.jpeg".format(rmax))
    
    plt.figure(3)
    plt.plot(t[1:], etheta[1:], 'r', label='Error')
    plt.plot(t[1:], np.sqrt(sigth[1:])*3, 'b', label="3-sigma bound")
    plt.plot(t[1:], -np.sqrt(sigth[1:])*3, 'b')
    plt.suptitle('Error Plots')
    plt.title('rmax = {}m'.format(rmax))
    plt.xlabel('timesteps')
    plt.ylabel('Error in theta(rad)')
    plt.legend()
    # plt.savefig("/home/aniket/Desktop/Courses_Fall/AER1513/Assignments/StateEstimation_AER1513/assignment2/Media/ErrorTh_poorinit_{}.jpeg".format(rmax))
    
    # plt.show()

# function to draw the covariance ellipse
def cov_ellipse(P_hat, nstd=3):
    a, b, c  = P_hat[0, 0], P_hat[0, 1], P_hat[1, 1]
    lambda_1 = 0.5 * (a + c) + np.sqrt((0.5*(a - c))**2 + b**2)
    lambda_2 = 0.5 * (a + c) - np.sqrt((0.5*(a - c))**2 + b**2)
    if (b == 0 and a >= c):
        th = 0
    elif(b == 0 and a < c):
        th = np.pi / 2
    else:
        th = np.arctan2(lambda_1 - a, b)

    ra = np.sqrt(lambda_1)
    rb = np.sqrt(lambda_2)
    

    return ra, rb, th

# wrap angles from [-pi, pi]
def wrap(theta):
    if theta > np.pi:
        theta = theta - 2*np.pi
    elif theta < -np.pi:
        theta = theta + 2*np.pi
    
    return theta

# initializations
def initialization():
    init_params = {}

    # number of timesteps
    init_params['T'] = data['t'].size

    # x0_hat, P0_hat (use right answer for initialization)
    init_params['x0_hat'] = np.array([[data['x_true'][0][0]], [data['y_true'][0][0]], [wrap(data['th_true'][0][0])]])
    # part b
    # init_params['x0_hat'] = np.array([[1], [1], [0.1]])
    init_params['P0_hat'] = np.diag([1, 1, 0.1])
     
    # Process noise at kth timestep
    init_params['Q_k'] = np.diag([data['v_var'][0][0], data['om_var'][0][0]])

    # Correction noise at kth timestep
    init_params['R_k'] = np.diag([data['r_var'][0][0], data['b_var'][0][0]])


    return init_params


#correction step
def correction(lk, rk, bk, R_k, x_check, P_check, k):
    
    # landmark positions
    x_l = lk[0]
    y_l = lk[1]
    
    # robot states from prediction
    if(CRLB == 'n'):
        x_k = x_check[0][0]
        y_k = x_check[1][0]
        theta = x_check[2][0]
    elif(CRLB == 'y'):
        x_k = x_true[k][0]
        y_k = y_true[k][0]
        theta = th_true[k][0]
    
    # wrap theta 
    theta = wrap(theta)
    bk = wrap(bk)
    
    # distances
    dx = x_l - x_k - d[0][0]*np.cos(theta)
    dy = y_l - y_k - d[0][0]*np.sin(theta)

    # Jacobians
    G_k = np.zeros((2, 3))
    G_k[0][0] = -dx / np.sqrt(dx**2 + dy**2)
    G_k[0][1] = -dy / np.sqrt(dx**2 + dy**2)
    G_k[0][2] = (-dy*d*np.cos(theta) + dx*d*np.sin(theta)) / np.sqrt(dx**2 + dy**2)
    A = dy/dx
    G_k[1][0] = (1/(1 + A**2))*(dy / dx**2)
    G_k[1][1] = - (1 / (1 + A**2)) * (1 / dx)
    G_k[1][2] = (1 / (1 + A**2)) * ((-d*np.cos(theta)*dx + dy*d*np.sin(theta))/ dx**2) - 1

    M_k = np.identity(2)
    R_k_prime = M_k @ R_k @ M_k.T

    # Kalman gain
    K_k = P_check @ G_k.T @ (np.linalg.inv(G_k @ P_check @ G_k.T + R_k_prime))

    # Correction
    Phat_k = (np.identity(3) - K_k @ G_k) @ P_check
    
    g_xcheck = np.array([[np.sqrt(dx**2 + dy**2)], [wrap(np.arctan2(dy, dx) - theta)]], dtype='float32')
    y_meas = np.array([[rk],[wrap(bk)]])
    
    xhat_k = x_check + K_k @ (y_meas - g_xcheck)
    xhat_k[2] = wrap(xhat_k[2])
    return Phat_k, xhat_k

def prediction(x_hat, P_hat, v, om, Qk,  delta_t, k):
    # CRLB condition
    if(CRLB == 'n'):
        x_hat = x_hat
    else:
        x_hat = np.array([x_true[k], y_true[k], th_true[k]])

    # wrap theta
    theta = wrap(x_hat[2])

    ## Prediction Step
    # get the Jacobians
    # wrt X
    F_km = np.array([[1, 0, -delta_t*np.sin(theta)*v], [0, 1, delta_t*np.cos(theta)*v], [0, 0, 1]], dtype='float32')
    # wrt process noise
    W_k_prime = np.array([[delta_t*np.cos(theta), 0], [delta_t*np.sin(theta), 0], [0, delta_t]], dtype='float32')
    Qk_prime = W_k_prime @ Qk @ W_k_prime.T
    
    Pcheck_k = F_km @ P_hat @ F_km.T + Qk_prime
    
    Z = np.array([[np.cos(theta), 0], [np.sin(theta), 0], [0, 1]], dtype='float32')
    ip = np.array([[v[0]], [om[0]]], dtype='float32')
    
    xcheck_k = x_hat + delta_t * Z @ ip

    return xcheck_k, Pcheck_k



# ekf
def ekf():
    fig, ax = plt.subplots(1, 1)

    # initializations
    init_params = initialization()
    x_hat = init_params['x0_hat']
    P_hat = init_params['P0_hat']
    
    sig_x = init_params['P0_hat'][0][0]
    sig_y = init_params['P0_hat'][1][1]
    sig_th = init_params['P0_hat'][2][2]
    
    x_hat_plot = init_params['x0_hat'].T
    P_hat_plot = init_params['P0_hat']
    
    # main loop
    for k in range(1, init_params['T']):
        delta_t = t[k] - t[k-1]
        
        # Prediction Step
        xcheck_k, Pcheck_k = prediction(x_hat, P_hat, v[k], om[k], init_params['Q_k'], delta_t, k)

        # Check for landmark availability
        n = np.count_nonzero(r[k])
        idx = np.nonzero(r[k])
        idx_less = np.where(np.logical_and(r[k] > 0, r[k] < float(rmax)))
        if (n > 0):
            for i in idx[0]:
                if (r[k][i] < float(rmax)):
                    # Correction Step
                    P_hat, x_hat = correction(l[i], r[k][i], b[k][i], init_params['R_k'], xcheck_k, Pcheck_k, k)
                
                    xcheck_k = x_hat
                    Pcheck_k = P_hat
                else:
                    P_hat, x_hat = Pcheck_k, xcheck_k
                    
            x_hat_plot = np.vstack([x_hat_plot, x_hat.T])
            sig_x = np.vstack([sig_x, P_hat[0][0]])
            sig_y = np.vstack([sig_y, P_hat[1][1]])
            sig_th = np.vstack([sig_th, P_hat[2][2]])
        else:
            # Prediction Step
            P_hat, x_hat = Pcheck_k, xcheck_k
            x_hat_plot = np.vstack([x_hat_plot, x_hat.T])
            sig_x = np.vstack([sig_x, P_hat[0][0]])
            sig_y = np.vstack([sig_y, P_hat[1][1]])
            sig_th = np.vstack([sig_th, P_hat[2][2]])

        
        #animation
        plt.clf()
        ax = plt.gca()
        
        plt.suptitle('Animation')
        plt.title('rmax = 1m')
        plt.xlim(-3, 11)
        plt.ylim(-3, 4)
        plt.scatter(l[:, 0], l[:, 1], zorder=100, s=30, c='black', label='Landmarks')
        plt.scatter(l[idx_less[0]][:, 0], l[idx_less[0]][:, 1], zorder=100, s=30, c='orange', label='Landmarks Visible')
        plt.scatter(x_hat[0][0], x_hat[1][0], zorder=100, s=20, c='r', label='Estimated')
        plt.scatter(x_true[k][0], y_true[k][0], zorder=100, s=20, c='b', label='Ground Truth')

        # variance ellipse
        ra, rb, th = cov_ellipse(P_hat)
        ellipse = Ellipse(xy = (x_hat[0][0], x_hat[1][0]), zorder=0, width = 3*2*ra, height = 3*2*rb, angle=th*(180/np.pi), color='red', alpha=0.3)
        ax.add_artist(ellipse)

        # find end point with angle
        endx = 0.5*np.cos(wrap(x_hat[2][0]))
        endy = 0.5*np.sin(wrap(x_hat[2][0]))
        endx_true = 0.5*np.cos(wrap(th_true[k][0]))
        endy_true = 0.5*np.sin(wrap(th_true[k][0]))
        plt.arrow(x_hat[0][0], x_hat[1][0], endx, endy, color='r')
        plt.arrow(x_true[k][0], y_true[k][0], endx_true, endy_true, color='b')
        
        plt.legend()
        plt.xlabel('x(m)')
        plt.ylabel('y(m)')
        plt.draw()
        plt.pause(0.000000001)
    
    plotting(x_hat_plot, sig_x, sig_y, sig_th)

    return P_hat, x_hat


if __name__ == "__main__":
    ekf()