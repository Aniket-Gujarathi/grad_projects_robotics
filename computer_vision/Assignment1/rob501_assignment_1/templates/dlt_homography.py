import numpy as np
from numpy.linalg import inv, norm
from scipy.linalg import null_space

def dlt_homography(I1pts, I2pts):
    """
    Find perspective Homography between two images.

    Given 4 points from 2 separate images, compute the perspective homography
    (warp) between these points using the DLT algorithm.

    Parameters:
    ----------- 
    I1pts  - 2x4 np.array of points from Image 1 (each column is x, y).
    I2pts  - 2x4 np.array of points from Image 2 (in 1-to-1 correspondence).

    Returns:
    --------
    H  - 3x3 np.array of perspective homography (matrix map) between image coordinates.
    A  - 8x9 np.array of DLT matrix used to determine homography.
    """
    #--- FILL ME IN ---
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

if __name__ == "__main__":
    I1 = np.array([[416, 485, 488, 410], [40,  61, 353, 349]])
    I2 = np.array([[2, 218, 218, 2], [2, 2, 409, 409]])
    dlt_homography(I1, I2)