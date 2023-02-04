import numpy as np

def histogram_eq(I):
    """
    Histogram equalization for greyscale image.

    Perform histogram equalization on the 8-bit greyscale intensity image I
    to produce a contrast-enhanced image J. Full details of the algorithm are
    provided in the Szeliski text.

    Parameters:
    -----------
    I  - Single-band (greyscale) intensity image, 8-bit np.array (i.e., uint8).

    Returns:
    --------
    J  - Contrast-enhanced greyscale intensity image, 8-bit np.array (i.e., uint8).
    """
    #--- FILL ME IN ---

    # Verify I is grayscale.
    if I.dtype != np.uint8:
        raise ValueError('Incorrect image format!')

    # create a histogram
    histogram,_ = np.histogram(I.ravel(), 256, [0, 256])
    
    #create a pdf
    pdf = histogram / I.size
    
    # create a cdf
    cdf = np.cumsum(pdf)

    #rescale from [0-255]
    cdf_rescaled = (256*cdf)        
    
    # Get enhanced image back using the original image information
    orig_img = I.flatten()
    enhanced = [cdf_rescaled[i] for i in orig_img]
    J = np.reshape(enhanced, I.shape)

    #------------------

    return J
