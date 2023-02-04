# Billboard hack script file.
import numpy as np
from matplotlib.path import Path
from imageio import imread, imwrite

from dlt_homography import dlt_homography
from bilinear_interp import bilinear_interp
from histogram_eq import histogram_eq

# import matplotlib.pyplot as plt


def billboard_hack():
    """
    Hack and replace the billboard!

    Parameters:
    ----------- 

    Returns:
    --------
    Ihack  - Hacked RGB intensity image, 8-bit np.array (i.e., uint8).
    """
    # Bounding box in Y & D Square image.
    bbox = np.array([[404, 490, 404, 490], [38,  38, 354, 354]])

    # Point correspondences.
    Iyd_pts = np.array([[416, 485, 488, 410], [40,  61, 353, 349]])
    Ist_pts = np.array([[2, 218, 218, 2], [2, 2, 409, 409]])

    Iyd = imread('../billboard/yonge_dundas_square.jpg')
    Ist = imread('../billboard/uoft_soldiers_tower_dark.png')

    Ihack = np.asarray(Iyd)
    Ist = np.asarray(Ist)

    #--- FILL ME IN ---

    # Let's do the histogram equalization first.
    J = histogram_eq(Ist).astype('uint8')
    
    # Compute the perspective homography we need...
    H, A = dlt_homography(Ist_pts, Iyd_pts) 
    
    # Main 'for' loop to do the warp and insertion - 
    # this could be vectorized to be faster if needed!

    # You may wish to make use of the contains_points() method
    # available in the matplotlib.path.Path class!

    # get the Iyd points in shape (4x2) for the contains_points method
    Iyd_rearranged = np.vstack((Iyd_pts[0], Iyd_pts[1])).T
    
    # get the points within the Iyd_pts polygon in the image Ihack using contains_points
    path = Path(Iyd_rearranged)
    x, y = np.mgrid[:Ihack.shape[1], :Ihack.shape[0]]
    
    # make a grid of shape (Nx2)
    grid_points = np.vstack((x.flatten(), y.flatten())).T 
    
    # find a patch which contains the points bounded by the Iyd_pts
    patch = path.contains_points(grid_points)
    patch_points = grid_points[np.where(patch)]
    
    
    # find inverse mapping of each point in the patch_points
    for i in range(0, patch_points.shape[0]):
        point = np.array([patch_points[i][0], patch_points[i][1], 1])
        
        # inverse warping from YD image to ST
        new_point = np.matmul(np.linalg.inv(H), point)
    
        # reshaping the point for the bilinear_interp function and normalize by the z component
        new_point = np.array([new_point[0] / new_point[2], new_point[1] / new_point[2]]).reshape((2, 1))

        # bilinear interpolation
        b = bilinear_interp(J, new_point)
        
        if (b):
            Ihack[patch_points[i][1], patch_points[i][0]] =  [b, b, b]       
    
    #------------------

    # plt.imshow(Ihack)
    # plt.show()
    # imwrite(Ihack, 'billboard_hacked.png');

    return Ihack

    

if __name__ == "__main__":
    hack = billboard_hack()