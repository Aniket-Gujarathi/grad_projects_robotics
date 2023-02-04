import numpy as np
from numpy.linalg import inv, norm

def triangulate(Kl, Kr, Twl, Twr, pl, pr, Sl, Sr):
    """
    Triangulate 3D point position from camera projections.

    The function computes the 3D position of a point landmark from the 
    projection of the point into two camera images separated by a known
    baseline. All arrays should contain float64 values.

    Parameters:
    -----------
    Kl   - 3x3 np.array, left camera intrinsic calibration matrix.
    Kr   - 3x3 np.array, right camera intrinsic calibration matrix.
    Twl  - 4x4 np.array, homogeneous pose, left camera in world frame.
    Twr  - 4x4 np.array, homogeneous pose, right camera in world frame.
    pl   - 2x1 np.array, point in left camera image.
    pr   - 2x1 np.array, point in right camera image.
    Sl   - 2x2 np.array, left image point covariance matrix.
    Sr   - 2x2 np.array, right image point covariance matrix.

    Returns:
    --------
    Pl  - 3x1 np.array, closest point on ray from left camera  (in world frame).
    Pr  - 3x1 np.array, closest point on ray from right camera (in world frame).
    P   - 3x1 np.array, estimated 3D landmark position in the world frame.
    S   - 3x3 np.array, covariance matrix for estimated 3D point.
    """
    #--- FILL ME IN ---

    # Compute baseline (right camera translation minus left camera translation).
    b = (Twr[:3, 3] - Twl[:3, 3]).reshape((3, 1))
    
    # Unit vectors projecting from optical center to image plane points.
    # Use variables rayl and rayr for the rays.
    
    #  rotation matrix (3x3)
    C_wl = Twl[:3, :3]
    C_wr = Twr[:3, :3]
    # convert into homogenous representation
    pl = np.vstack((pl, 1))
    pr = np.vstack((pr, 1))

    # ray vectors - unnormalized
    rayl_check = C_wl @ inv(Kl) @ pl
    rayr_check = C_wr @ inv(Kr) @ pr

    # unit vectors
    rayl = rayl_check / norm(rayl_check)
    rayr = rayr_check / norm(rayr_check)
    
    # Projected segment lengths.
    # Use variables ml and mr for the segment lengths.
    ml = ((b.T @ rayl) - (b.T @ rayr)*(rayl.T @ rayr)) / (1 - (rayl.T @ rayr)**2)
    mr = (rayl.T @ rayr)*ml - (b.T @ rayr)

    # Segment endpoints.
    # User variables Pl and Pr for the segment endpoints.
    # camera optical centres from the intrinsic matrix
    cl = Kl[:, 2].reshape(3, 1)
    cr = Kr[:, 2].reshape(3, 1)
    Pl = Twl[:3, 3].reshape(3, 1) + rayl*ml
    Pr = Twr[:3, 3].reshape(3, 1) + rayr*mr

    # Now fill in with appropriate ray Jacobians. These are 
    # 3x4 matrices, but two columns are zeros (because the right
    # ray direction is not affected by the left image point and 
    # vice versa).
    drayl = np.zeros((3, 4))  # Jacobian left ray w.r.t. image points.
    drayr = np.zeros((3, 4))  # Jacobian right ray w.r.t. image points.

    # Add code here...
    # dray_hat / ||dray|| - derivative wrt norm
    drayl_norm = (np.identity(3) - rayl @ rayl.T) / norm(rayl_check)
    drayr_norm = (np.identity(3) - rayr @ rayr.T) / norm(rayr_check)

    # derivative wrt point vector
    d_vec_l = C_wl @ inv(Kl)
    d_vec_r = C_wr @ inv(Kr)

    # derivative of point vector wrt points
    d_uv_l = np.array([[1, 0], [0, 1], [0, 0]])
    d_uv_r = np.array([[1, 0], [0, 1], [0, 0]])

    #  Jacobians
    J_l = drayl_norm @ d_vec_l @ d_uv_l
    J_r = drayr_norm @ d_vec_r @ d_uv_r
    
    drayl = np.hstack((J_l, np.zeros((3, 2))))
    drayr = np.hstack((np.zeros((3, 2)), J_r))

    #------------------

    # Compute dml and dmr (partials wrt segment lengths).
    u = np.dot(b.T, rayl) - np.dot(b.T, rayr)*np.dot(rayl.T, rayr)
    v = 1 - np.dot(rayl.T, rayr)**2

    du = (b.T@drayl).reshape(1, 4) - \
         (b.T@drayr).reshape(1, 4)*np.dot(rayl.T, rayr) - \
         np.dot(b.T, rayr)*((rayr.T@drayl) + (rayl.T@drayr)).reshape(1, 4)
 
    dv = -2*np.dot(rayl.T, rayr)*((rayr.T@drayl).reshape(1, 4) + \
        (rayl.T@drayr).reshape(1, 4))

    m = np.dot(b.T, rayr) - np.dot(b.T, rayl)@np.dot(rayl.T, rayr)
    n = np.dot(rayl.T, rayr)**2 - 1

    dm = (b.T@drayr).reshape(1, 4) - \
         (b.T@drayl).reshape(1, 4)*np.dot(rayl.T, rayr) - \
         np.dot(b.T, rayl)@((rayr.T@drayl) + (rayl.T@drayr)).reshape(1, 4)
    dn = -dv

    dml = (du*v - u*dv)/v**2
    dmr = (dm*n - m*dn)/n**2

    # Finally, compute Jacobian for P w.r.t. image points.
    JP = (ml*drayl + rayl*dml + mr*drayr + rayr*dmr)/2

    #--- FILL ME IN ---

    # 3D point.
    P = (Pl + Pr) / 2
    

    # 3x3 landmark point covariance matrix (need to form
    # the 4x4 image plane covariance matrix first).

    # image plane covariance
    I_cov = np.zeros((4, 4))
    I_cov[:2, :2] = Sl
    I_cov[2:, 2:] = Sr
    
    # landmark point covariance
    S = JP @ I_cov @ JP.T
    
    #------------------

    # Check for correct outputs...
    correct = isinstance(Pl, np.ndarray) and Pl.shape == (3, 1) and \
              isinstance(Pr, np.ndarray) and Pr.shape == (3, 1) and \
              isinstance(P,  np.ndarray) and P.shape  == (3, 1) and \
              isinstance(S,  np.ndarray) and S.shape  == (3, 3)

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return Pl, Pr, P, S