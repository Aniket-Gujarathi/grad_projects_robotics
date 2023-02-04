import numpy as np
from numpy.linalg import inv

def get_cross(A):
    Ax = np.array([[0, -A[2], A[1]], [A[2], 0, -A[0]], [-A[1], A[0], 0]])
    return Ax

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

def find_jacobian(K, Twc, Wpt):
    """
    Determine the Jacobian for NLS camera pose optimization.

    The function computes the Jacobian of an image plane point with respect
    to the current camera pose estimate, given a landmark point. The 
    projection model is the simple pinhole model.

    Note that the homogeneous transformation matrix provided defines the
    transformation from the *camera frame* to the *world frame* (to 
    project into the image, you would need to invert this matrix).

    Parameters:
    -----------
    K    - 3x3 np.array, camera intrinsic calibration matrix.
    Twc  - 4x4 np.array, homogenous pose matrix, current guess for camera pose. 
    Wpt  - 3x1 world point on calibration target (one of n).

    Returns:
    --------
    J  - 2x6 np.array, Jacobian matrix (columns are tx, ty, tz, r, p, q).
         The Jacobian must contain float64 values.
    """
    #--- FILL ME IN ---
 
    # Code goes here...
    
    # C_init, t_init
    C_init = Twc[:3, :3]
    t_init = Twc[:3, -1].reshape((3, 1))
    
    # form the u vector
    u = (Wpt - t_init)
    
    # break the procedure in three parts - rigid transformation, augmenting to image plane, apply intrinsic. 
    # Now apply chain rule

    # f1 = (C_wc)_inv*u, f2 = f1 / z, f3 = Kf2
    f1 = C_init.T @ u
    f2 = f1 / f1[2]
    f3 = K @ f2
    
    # partial derivatives
    delta_f3f2 = K
    delta_f2f1 = np.array([[1/f1[2], 0, -f1[0]/(f1[2]**2)], [0, 1/f1[2], -f1[1]/(f1[2])**2], [0, 0, 0]], dtype='float32')
    #partial derivative wrt translation
    delta_f1t = -C_init.T

    #partial derivative wrt rotation
    
    # According to the Wahba problem the Jacobian for the rotations 
    # find Ix
    I3 = np.array([0, 0, 1])
    I2 = np.array([0, 1, 0])
    I1 = np.array([1, 0, 0])
    I3x = get_cross(I3)
    I2x = get_cross(I2)
    I1x = get_cross(I1)
    
    # rpy from dcm
    rpy = rpy_from_dcm(C_init)
    r = rpy[0]
    p = rpy[1]
    y = rpy[2]
    
    # Jacobian calculation
    C_y = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]], dtype=np.float32)
    C_p = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]], dtype=np.float32)
    C_r = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]], dtype=np.float32)
    
    delta_f1r = -(I1x @ C_r.T @ C_p.T @ C_y.T @ u)
    delta_f1p = -(C_r.T @ I2x @ C_p.T @ C_y.T @ u)
    delta_f1y = -(C_r.T @ C_p.T @ I3x @ C_y.T @ u)
    
    delta_theta = np.array([delta_f1r, delta_f1p, delta_f1y])
    
    J_t = delta_f3f2 @ delta_f2f1 @ delta_f1t
    J_theta = (delta_f3f2 @ delta_f2f1 @ delta_theta).T.reshape((3, 3))
    J = np.concatenate((J_t, J_theta), axis=1)
    
    J = J[:2]
    
    #------------------

    correct = isinstance(J, np.ndarray) and \
        J.dtype == np.float64 and J.shape == (2, 6)

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return J