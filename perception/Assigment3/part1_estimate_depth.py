import os
import sys

import cv2
import numpy as np
import kitti_dataHandler


def main():

    ################
    # Options
    ################
    # Input dir and output dir
    # disp_dir = 'data/train/disparity'
    # output_dir = 'data/train/est_depth'
    # calib_dir = 'data/train/calib'
    # sample_list = ['000001', '000002', '000003', '000004', '000005']

    # test
    disp_dir = 'data/test/disparity'
    output_dir = 'data/test/est_depth'
    calib_dir = 'data/test/calib'
    sample_list = ['000011', '000012', '000013', '000014', '000015']
    ################

    for sample_name in (sample_list):
        # Read disparity map
        left_disp_path = disp_dir +'/' + sample_name + '.png'
        disp_left = cv2.imread(left_disp_path)

        # Read calibration info
        frame_calib = kitti_dataHandler.read_frame_calib(calib_dir + "/" + sample_name + ".txt")
        stereo_calib = kitti_dataHandler.get_stereo_calibration(frame_calib.p2, frame_calib.p3)

        # Calculate depth (z = f*B/disp)
        depth = stereo_calib.f * stereo_calib.baseline / np.array(disp_left)

        # Discard pixels past 80m, below 10cm and with disparity 0
        depth[depth==np.inf] = 0
        depth[depth>80.0] = 0
        depth[depth<0.1] = 0

        # Save depth map
        #train
        # cv2.imwrite(f'data/exp/depth/train/{sample_name}.png', depth)
        #test
        cv2.imwrite(f'data/exp/depth/test/{sample_name}.png', depth)


if __name__ == '__main__':
    main()
