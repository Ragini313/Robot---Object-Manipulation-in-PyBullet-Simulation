import pybullet as p
import open3d as o3d
import numpy as np 

import cv2

from src.utils import pb_image_to_numpy, depth_to_point_cloud

from src.robot import Robot
from src.simulation import Simulation

# from src.utils import depth_to_point_cloud

# class Perception:
def process_camera_data(rgb, depth, seg, viewMat_inv, intrinsic, camera_name):
    """Process RGB, depth, and segmentation data from a single camera."""
    unique_ids = np.unique(seg)
    object_poses = {}

    for obj_id in unique_ids:
        if obj_id <= 0:  # Skip background
            continue

        # Create a binary mask for this object
        mask = (seg == obj_id)

        # Filter RGB and depth using the mask
        rgb_masked = np.where(mask[..., np.newaxis], rgb, 0)
        depth_masked = np.where(mask, depth, 0)

        # Create RGBD image
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color=o3d.geometry.Image(rgb_masked),
            depth=o3d.geometry.Image(depth_masked),
            depth_scale=1.0,
            depth_trunc=1000.0,
            convert_rgb_to_intensity=False
        )

        # Create point cloud
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)

        # Remove points with zero depth
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        mask = ~np.all(points == 0, axis=1)
        pcd.points = o3d.utility.Vector3dVector(points[mask])
        pcd.colors = o3d.utility.Vector3dVector(colors[mask])

        if len(pcd.points) < 10:
            continue  # Skip objects with too few points

        # Compute centroid and orientation
        points_array = np.asarray(pcd.points)
        centroid_camera = np.mean(points_array, axis=0)
        centroid_world = np.dot(viewMat_inv, np.append(centroid_camera, 1.0))[:3]

        obb = pcd.get_oriented_bounding_box()
        orientation_camera = obb.R
        orientation_world = np.dot(viewMat_inv[:3, :3], orientation_camera)

        o3d.visualization.draw_geometries([pcd, obb])
        
        ## From this visualisation, i note the following:
        # id_1 = background
        # id_2 = table
        # id_3 = robot
        # id_4 = goal_basket
        # id_5 = ycb_banana?
        # id_6 = obstacle_1
        # id_7 = obstacle_2

        object_poses[obj_id] = {
            'position_camera': centroid_camera,
            'orientation_camera': orientation_camera,
            'position_world': centroid_world,
            'orientation_world': orientation_world,
            'dimensions': obb.extent
        }

    return object_poses

def combine_poses(poses1, poses2):
    """Combine poses from two cameras."""
    combined_poses = {}
    for obj_id in set(poses1.keys()).union(set(poses2.keys())):
        if obj_id in poses1 and obj_id in poses2:
            # Average positions and orientations
            pos1 = poses1[obj_id]['position_world']
            pos2 = poses2[obj_id]['position_world']
            combined_pos = (pos1 + pos2) / 2

            ori1 = poses1[obj_id]['orientation_world']
            ori2 = poses2[obj_id]['orientation_world']
            combined_ori = (ori1 + ori2) / 2  # Simple averaging (can be improved)

            combined_poses[obj_id] = {
                'position_world': combined_pos,
                'orientation_world': combined_ori,
                'dimensions': poses1[obj_id]['dimensions']  # Use dimensions from one camera
            }
        elif obj_id in poses1:
            combined_poses[obj_id] = poses1[obj_id]
        else:
            combined_poses[obj_id] = poses2[obj_id]
    return combined_poses

def visualize_poses(poses):
    """Visualize object poses."""
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    for obj_id, pose in poses.items():
        # Create coordinate frame at the object's position
        obj_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.2, 
            origin=pose['position_world']
        )
        vis.add_geometry(obj_frame)

    vis.run()
    vis.destroy_window()


def estimate_obb_from_camera(rgb, depth, seg_mask, obj_id, intrinsic, view_matrix_inv):
    """Estimates OBB without ground truth, using only camera data."""
    # Create mask for the target object
    mask = (seg_mask == obj_id)
    if not np.any(mask):
        return None

    # Mask RGB and depth
    rgb_masked = np.where(mask[..., None], rgb, 0)
    depth_masked = np.where(mask, depth, 0)

    # Create Open3D RGBD image
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color=o3d.geometry.Image(rgb_masked.astype(np.uint8)),
        depth=o3d.geometry.Image(depth_masked.astype(np.float32)),
        depth_scale=1.0,
        depth_trunc=1.0,  # Adjust based on your scene
        convert_rgb_to_intensity=False
    )

    # Generate point cloud
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd, intrinsic, project_valid_depth_only=True
    )

    # Remove outliers and downsample
    pcd = pcd.voxel_down_sample(voxel_size=0.01)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    if len(pcd.points) < 10:
        return None

    # Compute OBB
    obb = pcd.get_oriented_bounding_box()

    # Transform from camera to world coordinates
    points = np.asarray(pcd.points)
    points_homog = np.hstack([points, np.ones((len(points), 1))])
    points_world = (view_matrix_inv @ points_homog.T).T[:, :3]

    # Recompute OBB in world coordinates
    pcd_world = o3d.geometry.PointCloud()
    pcd_world.points = o3d.utility.Vector3dVector(points_world)
    obb_world = pcd_world.get_oriented_bounding_box()

    return {
        'position': obb_world.center,
        'orientation': obb_world.R,
        'extents': obb_world.extent
    }


## For accessing the EE camera when on top of YCB object:
 # # Capture RGB, depth, and segmentation images from the EE camera
    # rgb_ee, depth_ee, seg_ee = sim.get_ee_renders()

    
    # # Get view matrix for EE camera
    # ee_pos, ee_rot = sim.robot.get_ee_pose()
    # rot_matrix = p.getMatrixFromQuaternion(ee_rot)
    # rot_matrix = np.array(rot_matrix).reshape(3, 3)
    # init_camera_vector = (0, 0, 1)  # z-axis
    # init_up_vector = (0, 1, 0)  # y-axis
    # camera_vector = rot_matrix.dot(init_camera_vector)
    # up_vector = rot_matrix.dot(init_up_vector)
    # ee_viewMat = p.computeViewMatrix(ee_pos, ee_pos + 0.1 * camera_vector, up_vector)
    # ee_viewMat_array = np.array(ee_viewMat).reshape(4, 4)
    # ee_viewMat_inv = np.linalg.inv(ee_viewMat_array)




# Define camera intrinsic parameters (same for both cameras)
    # intrinsic = o3d.camera.PinholeCameraIntrinsic(
    #     width=width,
    #     height=height,
    #     fx=fx,
    #     fy=fy,
    #     cx=cx,
    #     cy=cy
    # )
    
 # # Process EE camera data
# ee_object_poses = process_camera_data(rgb_ee, depth_ee, seg_ee, ee_viewMat_inv, intrinsic, "EE Camera")

