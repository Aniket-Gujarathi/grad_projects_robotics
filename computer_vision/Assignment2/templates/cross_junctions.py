import numpy as np
from numpy.linalg import inv, lstsq
from scipy.linalg import null_space
from scipy.ndimage.filters import *
from matplotlib.path import Path

# You may add support functions here, if desired.

def saddle_point(I):
    """
    Locate saddle point in an image patch.

    The function identifies the subpixel centre of a cross-junction in the
    image patch I, by fitting a hyperbolic paraboloid to the patch, and then 
    finding the critical point of that paraboloid.

    Note that the location of 'pt' is relative to (-0.5, -0.5) at the upper
    left corner of the patch, i.e., the pixels are treated as covering an 
    area of one unit square.

    Parameters:
    -----------
    I  - Single-band (greyscale) image patch as np.array (e.g., uint8, float).

    Returns:
    --------
    pt  - 2x1 np.array (float64), subpixel location of saddle point in I (x, y).
    """
    #--- FILL ME IN ---
 
    # Code goes here.

    # error term
    h, w = I.shape
    # get the A and b matrices for linear least squares
    A = []
    b = []
    for x in range(w):
        for y in range(h): 
            A.append([x**2, x*y, y**2, x, y, 1])
            b.append([I[y, x]])
    #stack the matrices
    A = np.stack(A)
    b = np.stack(b)
    
    # linear least squares
    e = np.linalg.lstsq(A, b, rcond=None)
    
    # find the saddle point
    C = np.array([[2*e[0][0][0], e[0][1][0]], [e[0][1][0], 2*e[0][2][0]]])
    C_inv = np.linalg.inv(C)
    D = np.array([[e[0][3][0]], [e[0][4][0]]])
    pt = np.matmul(-C_inv,D)

    #------------------

    correct = isinstance(pt, np.ndarray) and \
        pt.dtype == np.float64 and pt.shape == (2, 1)

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return pt

def dlt_homography(I1pts, I2pts):
    # calculate A to find the solution Ah = 0
    A = []
    for i in range(I1pts.shape[1]):
        x = I1pts[0][i]
        y = I1pts[1][i]
        u = I2pts[0][i]
        v = I2pts[1][i]
        
        # formula for A in the thesis
        A.append(np.array([[-x, -y, -1, 0, 0, 0, u*x, u*y, u], [0, 0, 0, -x, -y, -1, v*x, v*y, v]]))
    
    # stack the (2x9) vectors to form the (8x9) vector A
    A = np.stack(A).reshape(8, 9)
    
    # solution for H is the null space of A
    H = null_space(A).reshape((3, 3))
    # normalize the H matrix  
    H = np.divide(H, H[-1][-1])     
    #------------------

    return H, A

def cross_junctions(I, bpoly, Wpts):
    """
    Find cross-junctions in image with subpixel accuracy.

    The function locates a series of cross-junction points on a planar 
    calibration target, where the target is bounded in the image by the 
    specified quadrilateral. The number of cross-junctions identified 
    should be equal to the number of world points.

    Note also that the world and image points must be in *correspondence*,
    that is, the first world point should map to the first image point, etc.

    Parameters:
    -----------
    I      - Single-band (greyscale) image as np.array (e.g., uint8, float).
    bpoly  - 2x4 np.array, bounding polygon (clockwise from upper left).
    Wpts   - 3xn np.array of world points (in 3D, on calibration target).

    Returns:
    --------
    Ipts  - 2xn np.array of cross-junctions (x, y), relative to the upper
            left corner of the target. The array must contain float64 values.
    """
    #--- FILL ME IN ---
    # Code goes here...
    # To account for the border (trial and error)
    delta_x = 1.3 * 0.0635
    delta_y = 1.1 * 0.0635

    # using known dimension of a square of checkerboard (8x6)
    w_board = 7 * 0.0635
    h_board = 5 * 0.0635
    

    # getting the destination points for the homography as we know that the object is a rectangle in 3D world with known dimension
    dst = np.array([[Wpts[0][0] - delta_x, w_board + delta_x, w_board + delta_x, Wpts[0][0] - delta_x], [Wpts[1][0] - delta_y, Wpts[1][0] - delta_y, h_board + delta_y, h_board + delta_y]]) 

    # homography calculation from target to the image plane
    H, _ = dlt_homography(dst, bpoly)

    # bring all Wpts on the target plane to the image plane using the inverse homography
    pts_img = []
    for i in range(Wpts[0].shape[0]):
        point = np.array([Wpts[0][i], Wpts[1][i], 1]) # (x, y) 
        pts_warp = H @ point
        pts_img.append([pts_warp[0] / pts_warp[2], pts_warp[1] / pts_warp[2]])
    pts_img = np.array(pts_img).T.astype('int32')
    

    # get the saddle points
    Ipts = np.zeros((2, Wpts[0].shape[0]))
    d = round(0.5*np.sqrt((pts_img[0][0] - pts_img[0][8])**2 + (pts_img[1][0] - pts_img[1][8])**2)) # window size for cropping
    
    for i in range(Wpts[0].shape[0]):
        x, y = pts_img[0][i], pts_img[1][i]
        local_saddle = saddle_point(I[y-d:y+d, x-d:x+d]).T[0]
        Ipts[0][i] = local_saddle[0] + x - d
        Ipts[1][i] = local_saddle[1] + y - d

 ### --- Trial --- ###
    # warping with known homography
    # A = cv2.warpPerspective(I, H, (round(w_board + delta_x), round(h_board + delta_y)))
    
    # # get saddle points in target frame
    # Ipts = np.ones((3, Wpts[0].shape[0]))
    # d = 63.5 # window size for cropping
    # Wpts_shift = (Wpts)*1000
    # for i in range(0, Wpts[0].shape[0]):
    #     try:
    #         local_saddle = saddle_point(A[int(Wpts_shift[1][i]-d):int(Wpts_shift[1][i]+d), int(Wpts_shift[0][i]-d):int(Wpts_shift[0][i]+d)])
    #         Ipts[0][i] = local_saddle[0] + Wpts_shift[0][i]
    #         Ipts[1][i] = local_saddle[1] + Wpts_shift[1][i]    
    #     except:
    #         continue
    # # Apply H inv to the saddle points
    # Ipts = np.linalg.inv(H) @ Ipts
    # Ipts[0] = Ipts[0] / Ipts[2]
    # Ipts[1] = Ipts[1] / Ipts[2]
    # Ipts = Ipts[0:2]
    
    ### ------ ###

    #------------------


    correct = isinstance(Ipts, np.ndarray) and \
        Ipts.dtype == np.float64 and Ipts.shape[0] == 2

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return Ipts