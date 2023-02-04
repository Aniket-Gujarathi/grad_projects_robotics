import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# from dcm_from_rpy import dcm_from_rpy
from triangulate import triangulate

def dcm_from_rpy(rpy):
    """
    Rotation matrix from roll, pitch, yaw Euler angles.

    The function produces a 3x3 orthonormal rotation matrix R
    from the vector rpy containing roll angle r, pitch angle p, and yaw angle
    y.  All angles are specified in radians.  We use the aerospace convention
    here (see descriptions below).  Note that roll, pitch and yaw angles are
    also often denoted by phi, theta, and psi (respectively).

    The angles are applied in the following order:

     1.  Yaw   -> by angle 'y' in the local (body-attached) frame.
     2.  Pitch -> by angle 'p' in the local frame.
     3.  Roll  -> by angle 'r' in the local frame.  

    Note that this is exactly equivalent to the following fixed-axis
    sequence:

     1.  Roll  -> by angle 'r' in the fixed frame.
     2.  Pitch -> by angle 'p' in the fixed frame.
     3.  Yaw   -> by angle 'y' in the fixed frame.

    Parameters:
    -----------
    rpy  - 3x1 np.array of roll, pitch, yaw Euler angles.

    Returns:
    --------
    R  - 3x3 np.array, orthonormal rotation matrix.
    """
    cr = np.cos(rpy[0]).item()
    sr = np.sin(rpy[0]).item()
    cp = np.cos(rpy[1]).item()
    sp = np.sin(rpy[1]).item()
    cy = np.cos(rpy[2]).item()
    sy = np.sin(rpy[2]).item()

    return np.array([[cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
                     [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
                     [  -sp,            cp*sr,            cp*cr]])

# Camera intrinsic matrices.
Kl = np.array([[500.0, 0.0, 320], [0.0, 500.0, 240.0], [0, 0, 1]])
Kr = Kl

# Camera poses (left, right).
Twl = np.eye(4)
Twl[:3, :3] = dcm_from_rpy([-np.pi/2, 0, 0])  # Tilt for visualization.
Twr = Twl.copy()
Twr[0, 3] = 1.0  # Baseline.

# Image plane points (left, right).
pl = np.array([[360], [237.0]])
pr = np.array([[240], [238.5]])

# Image plane uncertainties (covariances).
Sl = np.eye(2)
Sr = np.eye(2)

[Pl, Pr, P, S] = triangulate(Kl, Kr, Twl, Twr, pl, pr, Sl, Sr)

# Visualize - plot rays and the estimate of P...
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')

ax.plot(np.array([Twl[0, 3], Pl[0, 0]]), 
        np.array([Twl[1, 3], Pl[1, 0]]),
        np.array([Twl[2, 3], Pl[2, 0]]), 'b-')
ax.plot(np.array([Twr[0, 3], Pr[0, 0]]),
        np.array([Twr[1, 3], Pr[1, 0]]),
        np.array([Twr[2, 3], Pr[2, 0]]), 'r-')
ax.plot(np.array([Pl[0, 0], Pr[0, 0]]),
        np.array([Pl[1, 0], Pr[1, 0]]),
        np.array([Pl[2, 0], Pr[2, 0]]), 'g-')
ax.plot([P[0, 0]], [P[1, 0]], [P[2, 0]], 'bx', markersize = 8)
plt.show()