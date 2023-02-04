import numpy as np
from numpy.linalg import inv
# from rpy_from_dcm import rpy_from_dcm
# from dcm_from_rpy import dcm_from_rpy
from estimate_motion_ls import estimate_motion_ls

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

def rpy_from_dcm(R):
    """
    Roll, pitch, yaw Euler angles from rotation matrix.

    The function computes roll, pitch and yaw angles from the
    rotation matrix R. The pitch angle p is constrained to the range
    (-pi/2, pi/2].  The returned angles are in radians.

    Inputs:
    -------
    R  - 3x3 orthonormal rotation matrix.

    Returns:
    --------
    rpy  - 3x1 np.array of roll, pitch, yaw Euler angles.
    """
    rpy = np.zeros((3, 1))

    # Roll.
    rpy[0] = np.arctan2(R[2, 1], R[2, 2])

    # Pitch.
    sp = -R[2, 0]
    cp = np.sqrt(R[0, 0]*R[0, 0] + R[1, 0]*R[1, 0])

    if np.abs(cp) > 1e-15:
      rpy[1] = np.arctan2(sp, cp)
    else:
      # Gimbal lock...
      rpy[1] = np.pi/2
  
      if sp < 0:
        rpy[1] = -rpy[1]

    # Yaw.
    rpy[2] = np.arctan2(R[1, 0], R[0, 0])

    return rpy

def estimate_motion_ils(Pi, Pf, Si, Sf, iters):
    """
    Estimate motion from 3D correspondences.
  
    The function estimates the 6-DOF motion of a body, given a series 
    of 3D point correspondences. This method relies on NLS.
    
    Arrays Pi and Pf store corresponding landmark points before and after 
    a change in pose.  Covariance matrices for the points are stored in Si 
    and Sf. All arrays should contain float64 values.

    Parameters:
    -----------
    Pi  - 3xn np.array of points (intial - before motion).
    Pf  - 3xn np.array of points (final - after motion).
    Si  - 3x3xn np.array of landmark covariance matrices.
    Sf  - 3x3xn np.array of landmark covariance matrices.

    Outputs:
    --------
    Tfi  - 4x4 np.array, homogeneous transform matrix, frame 'i' to frame 'f'.
    """
    # Initial guess...
    Tfi = estimate_motion_ls(Pi, Pf, Si, Sf)
    C = Tfi[:3, :3]
    I = np.eye(3)
    rpy = rpy_from_dcm(C).reshape(3, 1)
    
    # initial theta
    theta = rpy
    
    # Iterate.
    for j in np.arange(iters):
        A = np.zeros((6, 6))
        B = np.zeros((6, 1))

        #  Jacobian and cov init
        J = np.zeros((3, 3))
        
        #--- FILL ME IN ---

        for i in np.arange(Pi.shape[1]):
            # Partial derivatives wrt theta
            Rx, Ry, Rz = dcm_jacob_rpy(C)
            
            # Jacobian wrt theta
            J[:, 0] = (Rx @ (Pi[:, i]).reshape(3, 1)).reshape(3)
            J[:, 1] = (Ry @ (Pi[:, i]).reshape(3, 1)).reshape(3)
            J[:, 2] = (Rz @ (Pi[:, i]).reshape(3, 1)).reshape(3)
            
            H = np.hstack((J, np.identity(3)))
            
            # Weights
            W = inv(Sf[:, :, i] + C @ Si[:, :, i] @ C.T)
            
            # residual
            Q = Pf[:, i].reshape(3, 1) - C @ Pi[:, i].reshape(3, 1) + J @ theta[0:3].reshape((3, 1))
            
            # Sum all A and B for all the landmarks
            A += H.T @ W @ H            
            B += H.T @ W @ Q
            
        #------------------

        # Solve system and check stopping criteria if desired...
        theta = inv(A)@B
        rpy = theta[0:3].reshape(3, 1)
        C = dcm_from_rpy(rpy)
        t = theta[3:6].reshape(3, 1)

    Tfi = np.vstack((np.hstack((C, t)), np.array([[0, 0, 0, 1]])))

    # Check for correct outputs...
    correct = isinstance(Tfi, np.ndarray) and Tfi.shape == (4, 4)

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return Tfi

def dcm_jacob_rpy(C):
     # Rotation - convenient!
    cp = np.sqrt(1 - C[2, 0]*C[2, 0])
    cy = C[0, 0]/cp
    sy = C[1, 0]/cp

    dRdr = C@np.array([[ 0,   0,   0],
                       [ 0,   0,  -1],
                       [ 0,   1,   0]])

    dRdp = np.array([[ 0,    0, cy],
                     [ 0,    0, sy],
                     [-cy, -sy,  0]])@C

    dRdy = np.array([[ 0,  -1,  0],
                     [ 1,   0,  0],
                     [ 0,   0,  0]])@C
    
    return dRdr, dRdp, dRdy