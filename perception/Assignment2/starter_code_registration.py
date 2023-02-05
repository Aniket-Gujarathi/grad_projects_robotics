import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

def rpy_from_dcm(R):
    """
    Roll, pitch, yaw Euler angles from rotation matrix.
    The function computes roll, pitch and yaw angles from the
    rotation matrix R. The pitch angle p is constrained to the range
    (-pi/2, pi/2].  The returned angles are in radians.
    Inputs:
    -------
    R  - 3x3 orthonormal rotation matrix.
    Returns:
    --------
    rpy  - 3x1 np.array of roll, pitch, yaw Euler angles.
    --------
    Reference: ROB 501 Assignment 2
    """
    rpy = np.zeros((3, 1))

    # Roll.
    rpy[0] = np.arctan2(R[2, 1], R[2, 2])

    # Pitch.
    sp = -R[2, 0]
    cp = np.sqrt(R[0, 0]*R[0, 0] + R[1, 0]*R[1, 0])

    if np.abs(cp) > 1e-15:
      rpy[1] = np.arctan2(sp, cp)
    else:
      # Gimbal lock...
      rpy[1] = np.pi/2
  
      if sp < 0:
        rpy[1] = -rpy[1]

    # Yaw.
    rpy[2] = np.arctan2(R[1, 0], R[0, 0])

    return rpy

def load_point_cloud(path):
    # Load the point cloud data (do NOT change this function!)
    data = pd.read_csv(path, header=None)
    point_cloud = data.to_numpy()
    return point_cloud

def nearest_search(pcd_source, pcd_target):
    """
    Description: Find the nearest neighbour (brute force solution)
    Input: source point cloud [x, y, z]
           target point cloud
    Output: Corresponding source points list
            Corresponding target points list
            mean distance between the corresponding points
    """
    # TODO: Implement the nearest neighbour search
    # TODO: Compute the mean nearest euclidean distance between the source and target point cloud
    corr_target = []
    corr_source = []
    distances = []
    ec_dist_mean = 0

    for source_point in pcd_source:
        min_dist = np.inf
        corr_source.append(source_point)
        for target_point in pcd_target:
            distance = np.linalg.norm(source_point - target_point)
            if distance < min_dist:
                min_dist = distance
                corr_target_point = target_point
        corr_target.append(corr_target_point)
        distances.append(min_dist)
    
    ec_dist_mean = np.mean(np.array(distances))
    return corr_source, corr_target, ec_dist_mean

def find_centroid(points):
    """
    Description: To find the centroid of a pointcloud
    Input: pointcloud list
    Output: centroid [x_c, y_c, z_c]
    """
    points = np.array(points)
    length = points.shape[0]
    sum_x = np.sum(points[:, 0])
    sum_y = np.sum(points[:, 1])
    sum_z = np.sum(points[:, 2])

    return [sum_x/length, sum_y/length, sum_z/length]

def estimate_pose(corr_source, corr_target):
    """
    Description: Estimate the pose for one iteration of ICP
    Input: Corresponding source points list [x, y, z]
           Corresponding target points list [x, y, z]
    Output: Pose (4 x 4)
            translation x
            translation y
            translation z    
    """
    # TODO: Compute the 6D pose (4x4 transform matrix)
    # TODO: Get the 3D translation (3x1 vector)
    pose = np.identity(4)
    translation_x = 0
    translation_y = 0
    translation_z = 0

    # find centroids and weights
    p_M = find_centroid(corr_source)
    p_D = find_centroid(corr_target)
    weights_sum = len(corr_source) # taking w_j = 1

    # compute outer product matrix
    source_offset = np.array(corr_source) - np.array(p_M)
    target_offset = np.array(corr_target) - np.array(p_D)
    W_DM = (target_offset.T @ source_offset) / weights_sum

    # svd
    v, s, u_T = np.linalg.svd(W_DM)

    # compose final rotation and translation
    M = np.eye(3)
    M[-1, -1] = np.linalg.det(u_T.T) * np.linalg.det(v)
    C_DM = v @ M @ u_T
    t_DM = -C_DM.T @ p_D + p_M

    # pose
    translation = -C_DM @ t_DM
    pose[:3, :3] = C_DM
    pose[:3, -1] = translation
    
    return pose, translation[0], translation[1], translation[2]


def icp(pcd_source, pcd_target):
    """
    Description: Implement ICP
    Input: Source point cloud [x, y, z]
           Target point cloud
    Output: pose (4 x 4)
    """
    # TODO: Put all together, implement the ICP algorithm
    # TODO: Use your implemented functions "nearest_search" and "estimate_pose"
    # TODO: Run 30 iterations
    # TODO: Show the plot of mean euclidean distance (from function "nearest_search") for each iteration
    # TODO: Show the plot of pose translation (from function "estimate_pose") for each iteration
    # Initialize
    pose = np.identity(4)
    mean_dist = []
    x_trans = []
    y_trans = []
    z_trans = [] 
    prev_pose = np.eye(4)
    iterations = 30
    for iter in range(iterations):
        # nearest search
        corr_source, corr_target, ec_dist_mean = nearest_search(pcd_source, pcd_target)
        mean_dist.append(ec_dist_mean)
        
        # estimate pose
        pose, translation_x, translation_y, translation_z = estimate_pose(corr_source, corr_target)
        x_trans.append(translation_x)
        y_trans.append(translation_y)
        z_trans.append(translation_z)

        # apply the transformation to source pointcloud
        pts = np.vstack([np.transpose(pcd_source), np.ones(len(pcd_source))])
        cloud_registered = np.matmul(pose, pts)
        pcd_source = np.transpose(cloud_registered[0:3, :])

        # append the pose
        final_pose = pose @ prev_pose
        prev_pose = final_pose

    # plot the ICP loss
    plt.plot(range(iterations), mean_dist, color="blue", label="ICP Loss")
    plt.legend()
    plt.xlabel("Number of Iterations")
    plt.ylabel("ICP Loss (Mean Euclidean Distance)")
    plt.show()

    # plot the 3D translations
    plt.subplots(figsize=(6, 4))
    plt.plot(range(iterations), x_trans, color="red", label="translation x")
    plt.plot(range(iterations), y_trans, color="green", label="translation y")
    plt.plot(range(iterations), z_trans, color="blue", label="translation z")
    plt.legend()
    plt.xlabel("Number of Iterations")
    plt.ylabel("3D Translation (mm)")
    plt.show()

    return final_pose


def main():
    # Dataset and ground truth poses
    #########################################################################################
    # Training and test data (3 pairs in total)
    train_file = ['bunny', 'dragon']
    test_file = ['armadillo']

    # Ground truth pose (from training data only, used for validating your implementation)
    GT_poses = []
    gt_pose = [0.8738,-0.1128,-0.4731,24.7571,
            0.1099,0.9934,-0.0339,4.5644,
            0.4738,-0.0224,0.8804,10.8654,
            0.0,0.0,0.0,1.0]
    gt_pose = np.array(gt_pose).reshape([4,4])
    GT_poses.append(gt_pose)
    gt_pose = [0.7095,-0.3180,0.6289,46.3636,
               0.3194,0.9406,0.1153,3.3165,
               -0.6282,0.1191,0.7689,-6.4642,
               0.0,0.0,0.0,1.0]
    gt_pose = np.array(gt_pose).reshape([4,4])
    GT_poses.append(gt_pose)
    #########################################################################################



    # Training (validate your algorithm)
    ##########################################################################################################
    for i in range(2):
        # Load data
        path_source = './training/' + train_file[i] + '_source.csv'
        path_target = './training/' + train_file[i] + '_target.csv'
        pcd_source = load_point_cloud(path_source)
        pcd_target = load_point_cloud(path_target)
        gt_pose_i = GT_poses[i]

        # Visualize the point clouds before the registration
        ax = plt.axes(projection='3d')
        ax.scatter3D(pcd_source[:,0], pcd_source[:,1], pcd_source[:,2], cmap='Greens')
        ax.scatter3D(pcd_target[:,0], pcd_target[:,1], pcd_target[:,2], cmap='Reds')
        plt.legend(["Source Point Cloud" , "Target Point Cloud"])
        ax.set_title('Point Clouds Before Registration')
        plt.show()

        # TODO: Use your implemented ICP algorithm to get the estimated 6D pose (from source to target point cloud)
        pose = icp(pcd_source, pcd_target)

        # Transform the point cloud
        # TODO: Replace the ground truth pose with your computed pose and transform the source point cloud
        pts = np.vstack([np.transpose(pcd_source), np.ones(len(pcd_source))])
        # cloud_registered = np.matmul(gt_pose_i, pts)
        cloud_registered = np.matmul(pose, pts)
        cloud_registered = np.transpose(cloud_registered[:3, :])

        # TODO: Evaluate the rotation and translation error of your estimated 6D pose with the ground truth pose
        rotation = pose[:3, :3]
        translation = pose[:3, -1]

        rotation_gt = gt_pose_i[:3, :3]
        translation_gt = gt_pose_i[:3, -1]

        translation_error = translation_gt - translation
        rpy = rpy_from_dcm(rotation)
        rpy_gt = rpy_from_dcm(rotation_gt)
        rotation_error = rpy_gt - rpy

        print(f"Rotation Error: {rotation_error}, Translation Error: {translation_error}")

        # Visualize the point clouds after the registration
        ax = plt.axes(projection='3d')
        ax.scatter3D(cloud_registered[:,0], cloud_registered[:,1], cloud_registered[:,2], cmap='Greens')
        ax.scatter3D(pcd_target[:,0], pcd_target[:,1], pcd_target[:,2], cmap='Reds')
        plt.legend(["Transformed Source Point Cloud", "Target Point Cloud"])
        ax.set_title('Point Clouds After Registration')
        plt.show()
    ##########################################################################################################



    # Test
    ####################################################################################
    for i in range(1):
        # Load data
        path_source = './test/' + test_file[i] + '_source.csv'
        path_target = './test/' + test_file[i] + '_target.csv'
        pcd_source = load_point_cloud(path_source)
        pcd_target = load_point_cloud(path_target)

        # Visualize the point clouds before the registration
        ax = plt.axes(projection='3d')
        ax.scatter3D(pcd_source[:,0], pcd_source[:,1], pcd_source[:,2], cmap='Greens')
        ax.scatter3D(pcd_target[:,0], pcd_target[:,1], pcd_target[:,2], cmap='Reds')
        plt.legend(["Source Point Cloud" , "Target Point Cloud"])
        ax.set_title('Point Clouds Before Registration')
        plt.show()

        # TODO: Use your implemented ICP algorithm to get the estimated 6D pose (from source to target point cloud)
        pose = icp(pcd_source, pcd_target)

        # TODO: Show your outputs in the report
        # TODO: 1. Show your estimated 6D pose (4x4 transformation matrix)
        # TODO: 2. Visualize the registered point cloud and the target point cloud
        print(f"Test Pose: {pose}")

        # Transform the point cloud
        pts = np.vstack([np.transpose(pcd_source), np.ones(len(pcd_source))])
        cloud_registered = np.matmul(pose, pts)
        cloud_registered = np.transpose(cloud_registered[:3, :])

        ax = plt.axes(projection='3d')
        ax.scatter3D(cloud_registered[:,0], cloud_registered[:,1], cloud_registered[:,2], cmap='Greens')
        ax.scatter3D(pcd_target[:,0], pcd_target[:,1], pcd_target[:,2], cmap='Reds')
        plt.legend(["Transformed Source Point Cloud", "Target Point Cloud"])
        ax.set_title('Point Clouds After Registration')
        plt.show()

if __name__ == '__main__':
    main()