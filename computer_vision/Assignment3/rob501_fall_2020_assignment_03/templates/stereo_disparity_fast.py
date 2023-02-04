import numpy as np
from scipy.ndimage.filters import *

import cv2

def stereo_disparity_fast(Il, Ir, bbox, maxd):
    """
    Fast stereo correspondence algorithm.

    This function computes a stereo disparity image from left stereo 
    image Il and right stereo image Ir. Only disparity values within
    the bounding box region (inclusive) are evaluated.

    Parameters:
    -----------
    Il    - Left stereo image, m x n pixel np.array, greyscale.
    Ir    - Right stereo image, m x n pixel np.array, greyscale.
    bbox  - 2x2 np.array, bounding box, relative to left image, from top left
            corner to bottom right corner (inclusive). (x, y) in columns.
    maxd  - Integer, maximum disparity value; disparities must be within zero
            to maxd inclusive (i.e., don't search beyond maxd).

    Returns:
    --------
    Id  - Disparity image (map) as np.array, same size as Il.
    """
    # Hints:
    #
    #  - Loop over each image row, computing the local similarity measure, then
    #    aggregate. At the border, you may replicate edge pixels, or just avoid
    #    using values outside of the image.
    #
    #  - You may hard-code any parameters you require in this function.
    #
    #  - Use whatever window size you think might be suitable.
    #
    #  - Optimize for runtime AND for clarity.

    #--- FILL ME IN ---

    # Code goes here...

########## BRUTE ############    
    # window size
    w_size = 11
    patch = w_size // 2

    # shapes
    h, w = Il.shape
    
    # initialize Id
    Id = np.zeros(Il.shape)
    
    # loop over the left image
    for y in range(bbox[1, 0], bbox[1, 1], patch):
        for xl in range(bbox[0, 0], bbox[0, 1], patch):
            # create a window of given size
            Cl = Il[y - patch:y + patch + 1, xl - patch:xl + patch + 1]
            
            # initialize scores
            min_score = np.inf
            disp = 0
            
            #loop over the epipolar line in right image
            for xr in range(xl, xl - maxd - 1, -1):
                # check if going over the limit
                if (xr - patch < 0 or xr + patch + 1 > Ir.shape[1]):
                    break

                # window in right image
                Cr = Ir[y - patch:y + patch + 1, xr - patch:xr + patch + 1]  
                
                # calculate sad
                sad = np.sum(np.abs(Cl.flatten() - Cr.flatten()))
                if (sad < min_score):
                    min_score = sad
                    disp = abs(xr - xl)
                    
            # #normalize disp
            # disp = 255*disp/maxd
            # set disparity value to the window
            Id[y-patch:y+patch, xl-patch:xl+patch] = disp

      
##############--------------#############################                    

                        
    #------------------

    correct = isinstance(Id, np.ndarray) and Id.shape == Il.shape

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return Id