import os
import glob
import yaml
import numpy as np
import open3d as o3d
import pybullet as p

import pybullet_data

from scipy.spatial.transform import Rotation as R
from typing import List, Dict, Any, Optional



def check_grasp_collision_with_pcd(
    self,
    gripper_meshes: List[o3d.geometry.TriangleMesh],
    object_pcd: o3d.geometry.PointCloud,
    num_collisions: int = 10,
    tolerance: float = 0.002  # slightly relaxed tolerance for real-world depth noise
    ) -> List[bool]:
    """
    Checks for collision between each gripper mesh and the point cloud.

    Args:
        gripper_meshes: List of gripper TriangleMesh objects at sampled grasp poses
        object_pcd: Point cloud of the scene or object captured by camera
        num_collisions: Number of close point pairs to define collision
        tolerance: Distance threshold to count a point as colliding

    Returns:
        List of booleans: True if collision occurs for that mesh, False otherwise
    """
    results = []

    # Build KDTree once for the object point cloud
    object_kdtree = o3d.geometry.KDTreeFlann(object_pcd)
    object_points = np.asarray(object_pcd.points)

    for mesh in gripper_meshes:
        # Sample the surface of the mesh to get a point cloud representation
        sampled_pcd = mesh.sample_points_uniformly(number_of_points=1000)
        sampled_points = np.asarray(sampled_pcd.points)

        collision_count = 0
        for point in sampled_points:
            k, idx, _ = object_kdtree.search_knn_vector_3d(point, 1)
            if k > 0:
                nearest_point = object_points[idx[0]]
                dist = np.linalg.norm(point - nearest_point)
                if dist <= tolerance:
                    collision_count += 1
                    if collision_count >= num_collisions:
                        break

        results.append(collision_count >= num_collisions)

    return results

#Filter out gripper positions (grasps) that are too far from the mesh center.

#Use a filter function with a distance tolerance.
def grasp_dist_filter(self, center_grasp: np.ndarray,
                    object_position: np.ndarray,
                    tolerance: float = 0.15) -> bool:
    distance = np.linalg.norm(center_grasp - object_position)
    return distance <= tolerance

def grasp_with_pybullet(self, translation, quaternion):
    """
    Use PyBullet's PandaSim to grasp an object at a specified pose.
    """
    # # Initialize PandaSim
    # panda = panda_sim.PandaSimAuto(p, [0, 0, 0])
    # panda.control_dt = self.step()

    # Move to the grasp position
    print("Moving to grasp position...")
    p.bullet_client.resetBasePositionAndOrientation(self.robot_id, translation, quaternion)

    # Execute grasp
    for _ in range(200):  # Wait for the robot to grasp the object 
        self.step()

    # Close gripper
    print("Closing gripper...")
    self.robot.close_gripper()
    
    # Lift object slightly
    lift_position = translation + np.array([0, 0, 0.05])  # Move up by 10cm
    p.bullet_client.resetBasePositionAndOrientation(self.robot_id, lift_position, quaternion)
    
    print("Grasp completed.")

def force_control_gripper(self, target_force=10.0, velocity=0.01):
    left_finger_idx = self.robot.gripper_idx[0]
    right_finger_idx = self.robot.gripper_idx[1]
    
    p.setJointMotorControl2(
        self.robot.id, left_finger_idx,
        p.VELOCITY_CONTROL,
        targetVelocity=-velocity,
        force=target_force
    )
    
    p.setJointMotorControl2(
        self.robot.id, right_finger_idx,
        p.VELOCITY_CONTROL,
        targetVelocity=-velocity,
        force=target_force
    )
def grasp_until_force_threshold(self, desired_force=10.0, max_steps=200):
    for _ in range(max_steps):
        self.force_control_gripper(target_force=desired_force)
        p.stepSimulation()
        
        left_force = np.linalg.norm(p.getJointState(self.robot.id,
                                                    self.robot.gripper_idx[0]))
        right_force = np.linalg.norm(p.getJointState(self.robot.id,
                                                    self.robot.gripper_idx[1]))

        avg_force = (left_force + right_force) / 2.0
        print(f"Gripper contact force: {avg_force:.2f} N")

        if avg_force >= desired_force * 0.95:
            print("Target grasp force reached.")
            break

# def score_grasp(self, translation, rotation_matrix, object_position):
#     dist_score = np.linalg.norm(translation - object_position)

#     # Prefer grasps where Z-axis points down
#     z_axis = rotation_matrix[:, 2]
#     alignment = np.dot(z_axis, np.array([0, 0, -1]))  # 1 = perfect downward

#     distance_weight = 1.0
#     alignment_weight = 0.8

#     score = distance_weight * dist_score - alignment_weight * alignment
#     return score

def rate_grasps(self, grasp_poses_list, object_position, object_orientation):
    """
    Rates grasps based on distance to object_position and orientation similarity.
    
    Args:
        grasp_poses_list (list): List of dicts with 'translation' and 'rotation_matrix'.
        object_position (np.array): 3D position of the object.
        object_orientation (np.array or None): Optional target rotation matrix.

    Returns:
        List of tuples (index, score), sorted by score (lower is better).
    """
    scored_grasps = []

    for idx, grasp in enumerate(grasp_poses_list):
        pos, rot = grasp
        pos = np.array(pos)
        rot = np.array(rot).reshape((3, 3))  # Ensure proper shape

        # Position distance
        dist_score = np.linalg.norm(pos - object_position)

        # Orientation distance (optional)
        if object_orientation is not None:
            grasp_rot = R.from_matrix(rot)
            obj_rot = R.from_matrix(object_orientation)
            ori_score = grasp_rot.inv() * obj_rot
            ori_dist = ori_score.magnitude()  # Quaternion angle difference
        else:
            ori_dist = 0

        total_score = dist_score + ori_dist
        scored_grasps.append((idx, total_score))

    return sorted(scored_grasps, key=lambda x: x[1])