import numpy as np
from numpy.linalg import inv
# from dcm_from_rpy import dcm_from_rpy
from estimate_motion_ls import estimate_motion_ls
from estimate_motion_ils import estimate_motion_ils

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

# Generate initial and transformed points.
C  = dcm_from_rpy(np.array([10, -8, 12])*np.pi/180)
t  = np.array([[0.5], [-0.8], [1.7]])

#Pi = np.array([[1, 2, 3, 4], [7, 3, 4, 8], [9, 11, 6, 3]])
Pi = np.random.rand(3, 10)
Pf = C@Pi + t  # You may wish to add noise to the points.
Si = np.dstack((1*np.eye(3),)*10)
Sf = np.dstack((1*np.eye(3),)*10)

Tfi_est = estimate_motion_ls(Pi, Pf, Si, Sf)

# Check that the transforms match...
Tfi = np.vstack((np.hstack((C, t)), np.array([[0, 0, 0, 1]])))
print(Tfi - Tfi_est)

# Now try with iteration.
Tfi_est = estimate_motion_ils(Pi, Pf, Si, Sf, 10)
print(Tfi - Tfi_est)