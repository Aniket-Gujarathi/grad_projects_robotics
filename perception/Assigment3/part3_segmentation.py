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
    depth_dir = 'data/train/gt_depth'
    label_dir = 'data/train/gt_labels'
    output_dir = 'data/exp/est_segmentation'
    image_dir = 'data/train/left'
    sample_list = ['000001', '000002', '000003', '000004', '000005', '000006', '000007', '000008', '000009', '000010']

    # Input dir and output dir
    # depth_dir = 'data/exp/est_depth/test'
    # label_dir = 'data/exp/labels'
    # output_dir = 'data/exp/est_segmentation'
    # image_dir = 'data/test/left'
    # sample_list = ['000011', '000012', '000013', '000014', '000015']
    ################

    for sample_name in sample_list:
    	# Read depth map
        depth_path = depth_dir +'/' + sample_name + '.png'
        depth = cv2.imread(depth_path)

        left_image = cv2.imread(f'{image_dir}/{sample_name}.png')
        
        # Discard depths less than 10cm from the camera
        depth[depth<0.1] = 0.0

        # Read 2d bbox
        bbox_list = kitti_dataHandler.read_labels(label_dir, sample_name)

        # For each bbox
        dist_thresh = 4 
        mask = 255 * np.ones(left_image.shape)
        for bbox in bbox_list:
            if bbox.type == "Car":
                # Estimate the average depth of the objects
                roi = depth[int(bbox.y1):int(bbox.y2)+1, int(bbox.x1):int(bbox.x2)+1]
                avg_depth = np.mean(roi)
                
                # Find the pixels within a certain distance
                nearby_points = (roi <= avg_depth + dist_thresh) & (roi >= avg_depth - dist_thresh)
                roi[nearby_points] = 0 
                mask[int(bbox.y1):int(bbox.y2)+1, int(bbox.x1):int(bbox.x2)+1] = roi
                
        # Save the segmentation mask
        # cv2.imshow("Image", mask)
        # cv2.waitKey(0)
        cv2.imwrite(f'data/exp/instance_segmentation/{sample_name}.png', mask)

if __name__ == '__main__':
    main()
