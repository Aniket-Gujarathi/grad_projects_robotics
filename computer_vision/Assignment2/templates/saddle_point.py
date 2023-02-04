import numpy as np
from numpy.linalg import inv, lstsq

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