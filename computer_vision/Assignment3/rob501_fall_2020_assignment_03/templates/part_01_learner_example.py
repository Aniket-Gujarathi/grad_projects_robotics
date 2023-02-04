import numpy as np
import matplotlib.pyplot as plt
from imageio import imread
# from stereo_disparity_score import stereo_disparity_score
from stereo_disparity_fast import stereo_disparity_fast

def stereo_disparity_score(It, Id, bbox):
    """
    Evaluate accuracy of disparity image.

    This function computes the RMS error between a true (known) disparity
    map and a map produced by a stereo matching algorithm. There are many
    possible metrics for stereo accuracy: we use the RMS error and the 
    percentage of incorrect disparity values (where we allow one unit
    of 'wiggle room').

    Note that pixels in the grouth truth disparity image with a value of
    zero are ignored (these are deemed to be invalid pixels).

    Parameters:
    -----------
    It    - Ground truth disparity image, m x n pixel np.array, greyscale.
    Id    - Computed disparity image, m x n pixel np.array, greyscale.
    bbox  - 2x2 np.array, bounding box, relative to left image, from top left
            corner to bottom right corner (inclusive).

    Returns:
    --------
    N     - Number of valid depth measurements in It image.
    rms   - Test score, RMS error between Id and It.
    pbad  - Percentage of incorrect depth values (for valid pixels).
    """
    # Ignore points where ground truth is unknown.
    mask = It != 0
    Id = Id.astype(np.float64)
    It = It.astype(np.float64)

    # Cut down the mask to only consider pixels in the box...
    mask[:, :bbox[0, 0]] = 0
    mask[:, bbox[0, 1] + 1:] = 0
    mask[:bbox[1, 0], :] = 0
    mask[bbox[1, 1] + 1:, :] = 0
    # plt.imshow(mask, cmap = "gray")
    # plt.show()

    N = np.sum(mask)  # Total number of valid pixels.
    rms = np.sqrt(np.sum(np.square(Id[mask] - It[mask]))/N)
    pbad = np.sum(np.abs(Id[mask] - It[mask]) > 2)/N

    return N, rms, pbad

# Load the stereo images and ground truth.
Il = imread("../stereo/cones/cones_image_02.png", as_gray = True)
Ir = imread("../stereo/cones/cones_image_06.png", as_gray = True)

# The cones and teddy datasets have ground truth that is 'stretched'
# to fill the range of available intensities - here we divide to undo.
It = imread("../stereo/cones/cones_disp_02.png",  as_gray = True)/4.0

# Load the appropriate bounding box.
bbox = np.load("../data/cones_02_bounds.npy")

Id = stereo_disparity_fast(Il, Ir, bbox, 55)
N, rms, pbad = stereo_disparity_score(It, Id, bbox)
print("Valid pixels: %d, RMS Error: %.2f, Percentage Bad: %.2f" % (N, rms, pbad))
plt.imshow(Id, cmap = "gray")
plt.show()