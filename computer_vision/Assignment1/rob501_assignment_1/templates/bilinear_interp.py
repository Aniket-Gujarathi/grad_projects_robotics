import numpy as np
from numpy.linalg import inv

def bilinear_interp(I, pt):
    """
    Performs bilinear interpolation for a given image point.

    Given the (x, y) location of a point in an input image, use the surrounding
    4 pixels to compute the bilinearly-interpolated output pixel intensity.

    Note that images are (usually) integer-valued functions (in 2D), therefore
    the intensity value you return must be an integer (use round()).

    This function is for a *single* image band only - for RGB images, you will 
    need to call the function once for each colour channel.

    Parameters:
    -----------
    I   - Single-band (greyscale) intensity image, 8-bit np.array (i.e., uint8).
    pt  - 2x1 np.array of point in input image (x, y), with subpixel precision.

    Returns:
    --------
    b  - Interpolated brightness or intensity value (whole number >= 0).
    """
    #--- FILL ME IN ---

    if pt.shape != (2, 1):
        raise ValueError('Point size is incorrect.')

    # get the surrounding 4 pixels positions (x, y)
    x_left = max(int(pt[0][0]), 0)
    y_up = max(int(pt[1][0]), 0)
    x_right = min(x_left + 1, I.shape[1] - 1)
    y_down = min(y_up + 1, I.shape[0] - 1)

    print(pt[0][0])

    # value at the pixels
    Qru = I[y_up, x_right]
    Qlu = I[y_up, x_left]
    Qrd = I[y_down, x_right]
    Qld = I[y_down, x_left]
    

    # linear interpolation in x direction
    f_up = ((x_right - pt[0][0]) / (x_right - x_left)) * Qlu + ((pt[0][0] - x_left) / (x_right - x_left)) * Qru
    f_down = ((x_right - pt[0][0]) / (x_right - x_left)) * Qld + ((pt[0][0] - x_left) / (x_right - x_left)) * Qrd

    # linear interpolation in y direction
    fy = ((y_down - pt[1][0]) / (y_down - y_up)) * f_up + ((pt[1][0] - y_up) / (y_down - y_up)) * f_down

    b = np.round(fy)

    #------------------

    return b

