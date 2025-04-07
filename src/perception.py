
import pybullet as p
import open3d as o3d
import numpy as np 
import matplotlib.pyplot as plt

import cv2

from src.utils import pb_image_to_numpy, depth_to_point_cloud

from src.robot import Robot
from src.simulation import Simulation



def camera_to_world_transform(view_matrix, static: bool = True) -> np.ndarray:
    view_matrix = np.array(view_matrix).reshape(4, 4, order='F')  # Fortran-order reshape
    transform = np.linalg.inv(view_matrix)
    
    # Invert X and Z translation components
    transform[0, 3] = -transform[0, 3]  # X-axis
    transform[2, 3] = -transform[2, 3]  # Z-axis
    
    # Additional rotation matrix (flips X and Z axes)
    additional_rotation = np.array([
        [1, 0, 0, 0],
        [ 0, 1, 0, 0],
        [ 0, 0, -1, 0],
        [ 0, 0, 0, 1]
    ])
    
    return additional_rotation @ transform

def camera_to_world_transform_1(view_matrix, static: bool = True) -> np.ndarray:   ## this is for non-ycb_objects
    view_matrix = np.array(view_matrix).reshape(4, 4, order='F')  # Fortran-order reshape
    transform = np.linalg.inv(view_matrix)
    
    # Invert X and Z translation components
    transform[0, 3] = -transform[0, 3]  # X-axis
    transform[2, 3] = -transform[2, 3]  # Z-axis
    
    # Additional rotation matrix (flips X and Z axes)
    additional_rotation = np.array([
        [-1, 0, 0, 0],
        [ 0, 1, 0, 0],
        [ 0, 0, -1, 0],
        [ 0, 0, 0, 1]
    ])
    
    return additional_rotation @ transform


def estimate_ycb_object_pose(img, view_mat, object_id, width, height, far, near, focal_length, show_visualizations=False):
    """
    Estimate the pose of an object in the scene given its segmentation ID.
    
    Args:
        img: The camera image tuple from pybullet's getCameraImage
        view_mat: The view matrix used for rendering
        object_id: The segmentation ID of the object to track
        width, height: Image dimensions
        far, near: Camera clipping planes
        focal_length: Camera focal length
        show_visualizations: Whether to display debug visualizations
    
    Returns:
        center_world: The object's center position in world coordinates (3D)
        center_cam: The object's center position in camera coordinates (3D)
    """
    # Extract components from camera image
    rgb_image = img[2][:, :, :3]  # RGB
    depth_buffer = img[3]          # Depth buffer
    seg_mask = img[4]              # Segmentation mask
    
    # Convert depth buffer to actual distances
    depth = far * near / (far - (far - near) * depth_buffer)
    
    # Create binary mask for the object
    binary_mask = (seg_mask == object_id)
    
    if show_visualizations:
        # Visualize segmentation mask
        plt.imshow(binary_mask, cmap='gray')
        plt.title(f"Segmentation Mask (Object ID={object_id})")
        plt.show()
        
        # # Visualize masked depth
        # masked_depth = np.where(binary_mask, depth, np.nan)
        # plt.imshow(masked_depth, cmap='viridis')
        # plt.colorbar(label='Depth (meters)')
        # plt.title(f"Masked Depth (Object ID: {object_id})")
        # plt.axis('off')
        # plt.show()
    
    # Calculate camera parameters
    cx, cy = width / 2, height / 2
    fx, fy = focal_length, focal_length
    
    # Get all pixels belonging to the object
    y_indices, x_indices = np.where(binary_mask)
    
    if len(y_indices) == 0:
        print(f"No pixels found for object ID {object_id}")
        return None, None
    
    # Calculate 3D points in camera coordinates
    Z = depth[binary_mask]
    X = (x_indices - cx) * Z / fx
    Y = (y_indices - cy) * Z / fy
    points_cam = np.column_stack((X, Y, Z))
    
    # Center in Camera Coordinates
    center_cam = np.mean(points_cam, axis=0)
    
    # Compute the corrected camera-to-world transform
    cam_to_world = camera_to_world_transform(view_mat)
    
    # Transform the camera-space center to world coordinates
    center_cam_hom = np.append(center_cam, 1.0)  # Homogeneous coordinates [X, Y, Z, 1]
    center_world_hom = cam_to_world @ center_cam_hom
    center_world = center_world_hom[:3]  # Extract X, Y, Z
    
    # Get ground truth for comparison
    try:
        ground_truth_pos = p.getBasePositionAndOrientation(object_id)
    except:
        ground_truth_pos = ([-999, -999, -999], [0, 0, 0, 1])  # Dummy if object doesn't exist
    
    # Print debug info
    print(f"\nYCB Object ID: {object_id}")
    #print(f"Center (Camera): [{center_cam[0]:.2f}, {center_cam[1]:.2f}, {center_cam[2]:.2f}]")
    print(f"Derived World Pose: [{center_world[0]:.2f}, {center_world[1]:.2f}, {center_world[2]:.2f}]")
    print(f"Ground Truth Pose: [({ground_truth_pos[0][0]:.2f}, {ground_truth_pos[0][1]:.2f}, {ground_truth_pos[0][2]:.2f}), ...]")
    
    return center_world, center_cam

def estimate_obstacle_pose(img, view_mat, object_id, width, height, far, near, focal_length, show_visualizations=False):
    """
    Estimate the pose of an object in the scene given its segmentation ID.
    
    Args:
        img: The camera image tuple from pybullet's getCameraImage
        view_mat: The view matrix used for rendering
        object_id: The segmentation ID of the object to track
        width, height: Image dimensions
        far, near: Camera clipping planes
        focal_length: Camera focal length
        show_visualizations: Whether to display debug visualizations
    
    Returns:
        center_world: The object's center position in world coordinates (3D)
        center_cam: The object's center position in camera coordinates (3D)
    """
    # Extract components from camera image
    rgb_image = img[2][:, :, :3]  # RGB
    depth_buffer = img[3]          # Depth buffer
    seg_mask = img[4]              # Segmentation mask
    
    # Convert depth buffer to actual distances
    depth = far * near / (far - (far - near) * depth_buffer)
    
    # Create binary mask for the object
    binary_mask = (seg_mask == object_id)
    
    if show_visualizations:
        # Visualize segmentation mask
        plt.imshow(binary_mask, cmap='gray')
        plt.title(f"Segmentation Mask (Object ID={object_id})")
        plt.show()
        
        # # Visualize masked depth
        # masked_depth = np.where(binary_mask, depth, np.nan)
        # plt.imshow(masked_depth, cmap='viridis')
        # plt.colorbar(label='Depth (meters)')
        # plt.title(f"Masked Depth (Object ID: {object_id})")
        # plt.axis('off')
        # plt.show()
    
    # Calculate camera parameters
    cx, cy = width / 2, height / 2
    fx, fy = focal_length, focal_length
    
    # Get all pixels belonging to the object
    y_indices, x_indices = np.where(binary_mask)
    
    if len(y_indices) == 0:
        print(f"No pixels found for object ID {object_id}")
        return None, None
    
    # Calculate 3D points in camera coordinates
    Z = depth[binary_mask]
    X = (x_indices - cx) * Z / fx
    Y = (y_indices - cy) * Z / fy
    points_cam = np.column_stack((X, Y, Z))
    
    # Center in Camera Coordinates
    center_cam = np.mean(points_cam, axis=0)
    
    # Compute the corrected camera-to-world transform
    cam_to_world = camera_to_world_transform_1(view_mat)
    
    # Transform the camera-space center to world coordinates
    center_cam_hom = np.append(center_cam, 1.0)  # Homogeneous coordinates [X, Y, Z, 1]
    center_world_hom = cam_to_world @ center_cam_hom
    center_world = center_world_hom[:3]  # Extract X, Y, Z
    
    # Get ground truth for comparison
    try:
        ground_truth_pos = p.getBasePositionAndOrientation(object_id)
    except:
        ground_truth_pos = ([-999, -999, -999], [0, 0, 0, 1])  # Dummy if object doesn't exist
    
    # Print debug info
    print(f"\nObstacle ID: {object_id}")
    #print(f"Center (Camera): [{center_cam[0]:.2f}, {center_cam[1]:.2f}, {center_cam[2]:.2f}]")
    print(f"Derived World Pose: [{center_world[0]:.2f}, {center_world[1]:.2f}, {center_world[2]:.2f}]")
    print(f"Ground Truth Pose: [({ground_truth_pos[0][0]:.2f}, {ground_truth_pos[0][1]:.2f}, {ground_truth_pos[0][2]:.2f}), ...]")
    
    return center_world, center_cam

def estimate_goal_basket_pose(img, view_mat, object_id, width, height, far, near, focal_length, show_visualizations=False):
    """
    Estimate the pose of an object in the scene given its segmentation ID.
    
    Args:
        img: The camera image tuple from pybullet's getCameraImage
        view_mat: The view matrix used for rendering
        object_id: The segmentation ID of the object to track
        width, height: Image dimensions
        far, near: Camera clipping planes
        focal_length: Camera focal length
        show_visualizations: Whether to display debug visualizations
    
    Returns:
        center_world: The object's center position in world coordinates (3D)
        center_cam: The object's center position in camera coordinates (3D)
    """
    # Extract components from camera image
    rgb_image = img[2][:, :, :3]  # RGB
    depth_buffer = img[3]          # Depth buffer
    seg_mask = img[4]              # Segmentation mask
    
    # Convert depth buffer to actual distances
    depth = far * near / (far - (far - near) * depth_buffer)
    
    # Create binary mask for the object
    binary_mask = (seg_mask == object_id)
    
    if show_visualizations:
        # Visualize segmentation mask
        plt.imshow(binary_mask, cmap='gray')
        plt.title(f"Segmentation Mask (Object ID={object_id})")
        plt.show()
        
        # # Visualize masked depth
        # masked_depth = np.where(binary_mask, depth, np.nan)
        # plt.imshow(masked_depth, cmap='viridis')
        # plt.colorbar(label='Depth (meters)')
        # plt.title(f"Masked Depth (Object ID: {object_id})")
        # plt.axis('off')
        # plt.show()
    
    # Calculate camera parameters
    cx, cy = width / 2, height / 2
    fx, fy = focal_length, focal_length
    
    # Get all pixels belonging to the object
    y_indices, x_indices = np.where(binary_mask)
    
    if len(y_indices) == 0:
        print(f"No pixels found for object ID {object_id}")
        return None, None
    
    # Calculate 3D points in camera coordinates
    Z = depth[binary_mask]
    X = (x_indices - cx) * Z / fx
    Y = (y_indices - cy) * Z / fy
    points_cam = np.column_stack((X, Y, Z))
    
    # Center in Camera Coordinates
    center_cam = np.mean(points_cam, axis=0)
    
    # Compute the corrected camera-to-world transform
    cam_to_world = camera_to_world_transform_1(view_mat)
    
    # Transform the camera-space center to world coordinates
    center_cam_hom = np.append(center_cam, 1.0)  # Homogeneous coordinates [X, Y, Z, 1]
    center_world_hom = cam_to_world @ center_cam_hom
    center_world = center_world_hom[:3]  # Extract X, Y, Z
    
    # Get ground truth for comparison
    try:
        ground_truth_pos = p.getBasePositionAndOrientation(object_id)
    except:
        ground_truth_pos = ([-999, -999, -999], [0, 0, 0, 1])  # Dummy if object doesn't exist
    
    # Print debug info
    print(f"\nGoal Basket ID: {object_id}")
    #print(f"Center (Camera): [{center_cam[0]:.2f}, {center_cam[1]:.2f}, {center_cam[2]:.2f}]")
    print(f"Derived World Pose: [{center_world[0]:.2f}, {center_world[1]:.2f}, {center_world[2]:.2f}]")
    print(f"Ground Truth Pose: [({ground_truth_pos[0][0]:.2f}, {ground_truth_pos[0][1]:.2f}, {ground_truth_pos[0][2]:.2f}), ...]")
    
    return center_world, center_cam


def visualize_pose_results(results):
    """Visualize the pose estimation results."""
    if results is None:
        print("Object not visible in the image")
        return
    
    # Display binary mask
    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.imshow(results['binary_mask'], cmap='gray')
    plt.title(f"Segmentation Mask (Object ID={results['object_id']})")
    
    # Display masked depth
    plt.subplot(132)
    plt.imshow(results['masked_depth'], cmap='viridis')
    plt.colorbar(label='Depth (meters)')
    plt.title(f"Masked Depth (Object ID: {results['object_id']})")
    
    # Display 2D bounding box
    img = np.zeros((results['binary_mask'].shape[0], results['binary_mask'].shape[1], 3))
    x_min, y_min, x_max, y_max = results['bbox_2d']
    img[y_min:y_max, x_min:x_max] = [1, 0, 0]  # Red box
    
    plt.subplot(133)
    plt.imshow(img)
    plt.title("2D Bounding Box")
    plt.tight_layout()
    plt.show()
    
    # Print pose information
    print(f"\nObject ID: {results['object_id']}")
    print(f"Center (Camera): [{results['center_cam'][0]:.2f}, {results['center_cam'][1]:.2f}, {results['center_cam'][2]:.2f}]")
    print(f"Center (World): [{results['center_world'][0]:.2f}, {results['center_world'][1]:.2f}, {results['center_world'][2]:.2f}]")


