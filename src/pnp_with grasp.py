

import os
import glob
import yaml
import numpy as np
import pybullet as p
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from typing import Sequence, Tuple

import pybullet_robots.panda.panda_sim_grasp as panda_sim


from typing import Dict, Any, List
from pybullet_object_models import ycb_objects  # type:ignore

from src.simulation import Simulation as sim
from src.grasping import*


class Grasp:
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



#****** RAFEY's Pick and Place Code******#

# import numpy as np
# from scipy.spatial.transform import Rotation as R

# from control import Control 
# from robot import Robot

# class PickAndPlace:
#     def __init__(self):
#         # Define robot parameters (replace with your actual robot's parameters)
#         self.link_lengths = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]  # Example
#         self.joint_limits = [(-2.9671, 2.9671), (-1.8326, 1.8326), (-2.9671, 2.9671),
#                              (-3.1416, 0.0), (-2.9671, 2.9671), (-0.0873, 3.8223),
#                              (-2.9671, 2.9671)]  # Example

#         # Gripper parameters (replace with your actual gripper's parameters)
#         self.gripper_open_value = 0.04
#         self.gripper_closed_value = 0.0

#         # Define table and object dimensions (example values)
       
#         self.control = Control()
#         self.robot = Robot()



#     def pick(self, object_pose):
#         """
#         Placeholder: Implements the pick operation.
#         This function defines the sequence of actions needed to pick up the object.
#         """
#         print("Starting pick operation...")

#         # 1. Pre-grasp approach (move above the object)
#         pre_grasp_position = object_pose[:3] + [0, 0, 0.1]  # Approach from above (10cm)
#         pre_grasp_orientation = R.from_euler('z', 0)
#         initial_joints = self.robot.get_joint_positions()  # Initial guess for joint angles
#         joint_angles = self.control.ik_solver(pre_grasp_position, pre_grasp_orientation, initial_joints)
#         if joint_angles is None:
#             print("IK failed for pre-grasp position")
#             return False
#         self.robot.position_control(joint_angles) #Move robot to the joint angles

#         # 2. Open the gripper
#         self.robot.open_gripper()

#         # 3. Move to grasp position
#         grasp_position = object_pose[:3]
#         grasp_orientation = R.from_euler('z', 0) #Adjust rotation as needed
#         joint_angles = self.control.ik_solver(grasp_position, grasp_orientation, joint_angles)
#         if joint_angles is None:
#             print("IK failed for grasp position")
#             return False
#         self.robot.position_control(joint_angles) #Move robot to the joint angles

#         # 4. Close the gripper
#         self.robot.close_gripper()

#         # 5. Post-grasp lift (move up with the object)
#         post_grasp_position = object_pose[:3] + [0, 0, 0.1]
#         post_grasp_orientation = R.from_euler('z', 0)
#         joint_angles = self.control.ik_solver(post_grasp_position, post_grasp_orientation, joint_angles)
#         if joint_angles is None:
#             print("IK failed for post-grasp position")
#             return False
#         self.robot.position_control(joint_angles) #Move robot to the joint angles

#         print("Pick operation complete.")
#         return True

#     def place(self, place_pose):
#         """
#         Placeholder: Implements the place operation.
#         This function defines the sequence of actions needed to place the object at the goal location.
#         """
#         print("Starting place operation...")

#         # 1. Pre-place approach (move above the place location)
#         pre_place_position = place_pose[:3] + [0, 0, 0.1]  # Approach from above (10cm)
#         pre_place_orientation = R.from_euler('z', 0)
#         initial_joints = [0, 0, 0, 0, 0, 0, 0]  # Initial guess for joint angles
#         joint_angles = self.control.ik_solver(pre_place_position, pre_place_orientation, initial_joints)
#         if joint_angles is None:
#             print("IK failed for pre-place position")
#             return False
#         self.move_to_joint_angles(joint_angles) #Move robot to the joint angles

#         # 2. Move to place position
#         place_position = place_pose[:3]
#         place_orientation = R.from_euler('z', 0)
#         joint_angles = self.inverse_kinematics(place_position, place_orientation, joint_angles)
#         if joint_angles is None:
#             print("IK failed for place position")
#             return False
#         self.robot.position_control(joint_angles) #Move robot to the joint angles

#         # 3. Open the gripper
#         self.robot.open_gripper()

#         # 4. Post-place retreat (move up after releasing the object)
#         post_place_position = place_pose[:3] + [0, 0, 0.1]
#         post_place_orientation =  R.from_euler('z', 0)
#         joint_angles = self.inverse_kinematics(post_place_position, post_place_orientation, joint_angles)
#         if joint_angles is None:
#             print("IK failed for post-place position")
#             return False
#         self.robot.position_control(joint_angles) #Move robot to the joint angles

#         print("Place operation complete.")
#         return True

# def main():
#     # Initialize the PickAndPlace instance
#     pick_and_place = PickAndPlace()

#     # Define object and goal poses
#     object_pose = (-0.050183952461055004, -0.46971427743603356, 1.3231258620680433)
#     object_orientation = (0.0, 0.0, 0.0, 1.0)
#     # goal_pose = (0.65, 0.8, 1.24)
#     # goal_orientation = (-0.0, -0.0, 0.7071067811865475, 0.7071067811865476)

   

#     # # Perform pick operation
#     # print("\nExecuting pick operation...")
#     # pick_success = pick_and_place.pick(object_pose, object_orientation)

#     # if pick_success:
#     #     # Perform place operation
#     #     print("\nExecuting place operation...")
#     #     place_success = pick_and_place.place(goal_pose, goal_orientation)

#     #     if place_success:
#     #         print("\nPick and place task completed successfully!")
#     #     else:
#     #         print("\nPlace operation failed.")
#     # else:
#     #     print("\nPick operation failed.")

# if __name__ == "__main__":
#     main()

# object_pose = (296.22743805,  61.04207592,   0.99762923)

# object_orientation = (0.0, 0.0, 0.0, 1.0)
# # Convert quaternion to rotation matrix
# goal_orientation_matrix = sim.control.quaternion_to_rotation_matrix(object_orientation)
# print("Goal orientation roation matrix:", goal_orientation_matrix)

# # Get the initial joint angles from the robot
# initial_joint_angles = sim.robot.get_joint_positions()

# print("Inital joint angles of the robot:", initial_joint_angles)

# # calculating the new joint angles that the franka has to be in to get reach the object location 

# final_joint_angles = sim.control.ik_solver(
#     goal_position=np.array(object_pose),
#     goal_orientation=goal_orientation_matrix,
#     initial_joint_angles=initial_joint_angles,
#     max_iterations=100,
#     tolerance=1e-3,
#     learning_rate=0.1
#     )

# sim.robot.position_control(final_joint_angles)




def move_down_to_object(self, goal_position, initial_joint_angles, step_size=0.01, max_steps=100):
    """
    Move the end-effector downwards to the object's position.
    """
    current_joint_angles = initial_joint_angles.copy()
    for step in range(max_steps):
        # Get current end-effector position
        ee_pos, _ = self.get_ee_pose(current_joint_angles)

        # Check if the end-effector is close to the goal position
        if np.linalg.norm(ee_pos - goal_position) < step_size:
            print("Reached the object position.")
            break

        # Move downwards by a small step
        goal_position[2] -= step_size  # Move downwards in the z-axis

        # Solve IK for the new goal position
        current_joint_angles = self.ik_solver(
            goal_position=goal_position,
            initial_joint_angles=current_joint_angles,
            max_iterations=10,
            tolerance=1e-3,
            learning_rate=0.1
        )

        # Move the robot to the new joint angles
        self.robot.position_control(current_joint_angles)

        # Step the simulation
        for _ in range(10):  # Adjust the number of steps as needed
            p.stepSimulation()



def pick_object(self, object_position, initial_joint_angles):
    """
    Pick the object by opening the gripper, moving down, closing the gripper, and moving back up.
    """
    # Step 1: Open the gripper
    print("Opening the gripper...")
    self.robot.open_gripper()
    for _ in range(100):  # Wait for the gripper to open
        p.stepSimulation()

    # Step 2: Move to a position slightly above the object
    print("Moving to the object...")
    goal_position = np.array(object_position) + np.array([0, 0, 0.1])  # 10 cm above the object
    final_joint_angles = self.ik_solver(
        goal_position=goal_position,
        initial_joint_angles=initial_joint_angles,
        max_iterations=100,
        tolerance=1e-3,
        learning_rate=0.1
    )
    self.robot.position_control(final_joint_angles)
    for _ in range(100):  # Wait for the robot to reach the position
        p.stepSimulation()

    # Step 3: Move down to the object
    print("Moving down to the object...")
    self.move_down_to_object(
        goal_position=np.array(object_position),
        initial_joint_angles=final_joint_angles
    )

    # Step 4: Close the gripper
    print("Closing the gripper...")
    self.robot.close_gripper()
    for _ in range(100):  # Wait for the gripper to close
        p.stepSimulation()

    # Step 5: Move back up
    print("Moving back up...")
    goal_position = np.array(object_position) + np.array([0, 0, 0.1])  # 10 cm above the object
    final_joint_angles = self.ik_solver(
        goal_position=goal_position,
        initial_joint_angles=self.robot.get_joint_positions(),
        max_iterations=50,    ## reduced from 100 to 50 because program freezes after 47 iterations
        tolerance=1e-3,
        learning_rate=0.1
    )
   
    self.robot.position_control(final_joint_angles)
    for _ in range(100):  # Wait for the robot to move back up
        p.stepSimulation()

    print("Object picked successfully.")

def pick_and_place(sim, object_position, target_position):
    """
    Perform a pick-and-place operation:
    1. Open the gripper.
    2. Move to the object.
    3. Access the ee camera images and create point cloud images and sample the grasp on the point cloud 
    4. Detect the object using yolo11
    5. Move down to the object
    6. Grasping the object.
    7. Move back up.
    8. Move to the target position.
    9. Move down to the target position 
    10. Open the gripper to release the object.
    """
    # Define the desired orientation (identity quaternion: no rotation)
    desired_quaternion = np.array([1.0, 0.0, 0.0, 0.0])  # Quaternion (w, x, y, z)
    desired_orientation = sim.control.quaternion_to_rotation_matrix(desired_quaternion)  # Convert to rotation matrix

    # Step 1: Open the gripper
    print("Opening the gripper...")
    sim.robot.open_gripper()
    for _ in range(100):  # Wait for the gripper to open
        sim.step()

    # Step 2: Move to the object (slightly above it)
    print("Moving to the object...")
    goal_position = np.array(object_position) + np.array([0, 0, 0.1])  # 10 cm above the object
    initial_joint_angles = sim.robot.get_joint_positions()
    final_joint_angles = sim.control.ik_solver(
        goal_position=goal_position,
        goal_orientation=desired_orientation,  # Pass the rotation matrix
        initial_joint_angles=initial_joint_angles,
        max_iterations=100,
        tolerance=1e-3,
        learning_rate=0.1
    )
    sim.robot.position_control(final_joint_angles)
    for _ in range(200):  # Wait for the robot to reach the position
        sim.step()

    # Step 3: Access the ee camera images and create point cloud images and sample the grasp on the point cloud 
    print("Capturing EE camera images...")
    # _, depth_image, _ = self.sim.capture_ee_images()
    rgb_image, depth_image, segmentation_mask = sim.get_ee_renders()
    
    # print(depth_image)
    # # Camera parameters
    # fx = 640 / (2.0 * np.tan(np.radians(70 / 2.0)))  # Assume FOV = 70 degrees
    # fy = fx  # Assuming square pixels
    # cx, cy = 640 / 2, 480 / 2  # Principal point

    # # Create intrinsic matrix
    # intrinsic = o3d.camera.PinholeCameraIntrinsic()
    # intrinsic.set_intrinsics(640, 480, fx, fy, cx, cy)

    # # Convert depth image to Open3D format
    # depth_image_o3d = o3d.geometry.Image(depth_image)

    # print(depth_image_o3d)

    # # Generate point cloud from depth image
    # point_cloud = o3d.geometry.PointCloud.create_from_depth_image(
    #     depth_image_o3d,
    #     intrinsic,
    #     depth_scale=1000.0,  # Adjust based on your depth image scale
    #     depth_trunc=5.0   # Truncate distances beyond this value
    # )

    # print(point_cloud)
    # print(np.asarray(point_cloud.points))
    # point_cloud.paint_uniform_color([1, 0, 0])  # Red color
    # o3d.visualization.draw_geometries([point_cloud])
    # Camera parameters

    ee_pos, ee_ori = sim.robot.get_ee_pose() 

    # Apply offset to calculate camera position
    ee_cam_offset = [0.0, 0.0, 0.1]  # Offset from configuration
    cam_pos = [
        ee_pos[0] + ee_cam_offset[0],
        ee_pos[1] + ee_cam_offset[1],
        ee_pos[2] + ee_cam_offset[2]
    ]        
    
    # Use end-effector orientation for camera alignment
    rotation_matrix = np.array(p.getMatrixFromQuaternion(ee_ori)).reshape(3, 3)

    # Camera target direction (forward vector)
    cam_forward = rotation_matrix[:, 2]
    cam_target = [
        cam_pos[0] + cam_forward[0],
        cam_pos[1] + cam_forward[1],
        cam_pos[2] + cam_forward[2]
    ]
   
    # Camera up direction (Y-axis)
    cam_up = rotation_matrix[:, 1]

    fov = 70                       # Field of view (degrees)
    aspect = 640 / 480             # Aspect ratio (width/height)
    near = 0.01                     # Near clipping plane
    far = 5.0                      # Far clipping plane
    width, height = 640, 480
    focal_length = width / (2 * np.tan(np.deg2rad(fov) / 2))

    # Compute view and projection matrices
    view_mat = p.computeViewMatrix(cam_pos, cam_target, cam_up)
    proj_mat = p.computeProjectionMatrixFOV(fov, aspect, near, far)

    # Get camera image (returns RGB, depth, segmentation)
    img = p.getCameraImage(width, height, view_mat, proj_mat)

    # Extract depth buffer (img[3] = depth buffer)
    
    rgb_image = img[2]  # Shape: (height, width, 4) (RGBA format)
    
    # # Display the image using matplotlib
    # plt.imshow(seg_s)
    # plt.axis('off')  # Hide axes
    # plt.show()


    depth_buffer = img[3]  # Shape: (height, width)

    # Convert depth buffer to actual distances (linearize if needed)
    # PyBullet's depth buffer is in [0, 1], where 1 = far plane, 0 = near plane
    depth = far * near / (far - (far - near) * depth_buffer)
    print(f"Depth value: {np.min(depth_image):.3f} to {np.max(depth_image):.3f}")

    # For visualization, we invert and normalize
    depth_image = 1 - depth_buffer  # Invert for better visualization (optional)
    depth_image = (depth_image - np.min(depth_image)) / (np.max(depth_image) - np.min(depth_image))  # Normalize to [0, 1]
    print(f"Depth image value: {np.min(depth_image):.3f} to {np.max(depth_image):.3f}")

    # # Display the depth image
    # plt.imshow(depth_image, cmap='gray')  # Grayscale for depth
    # plt.colorbar(label='Depth (0=near, 1=far)')
    # plt.title("Depth Image")
    # plt.axis('off')
    # plt.show()

    ## To create a pointcloud from depth_s:
    rgb_image = img[2][:, :, :3]  # Extract RGB

    # Convert depth buffer to actual distances
    depth = far * near / (far - (far - near) * depth_buffer)

    # Generate 3D points
    fx, fy = focal_length, focal_length
    cx, cy = width / 2, height / 2

    points = []
    colors = []

    for v in range(height):
        for u in range(width):
            Z = depth[v, u]
            if Z > 0:  # Ignore invalid depths
                X = (u - cx) * Z / fx
                Y = (v - cy) * Z / fy
                points.append([X, Y, Z])

                # Attach RGB color (if needed)
                colors.append(rgb_image[v, u] / 255.0)  # Normalize RGB to [0,1]

    # Convert to NumPy array
    points = np.array(points)
    colors = np.array(colors)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    # o3d.visualization.draw_geometries([pcd])
    
    #***Sampling the grasp around the point cloud***#
    object_position = np.array([0.0, -0.425, 0.5])  # Object's position
    num_grasps = 10  # Number of grasps to sample
    offset = 0.1  # Maximum sampling distance around the object

    grasp_poses_list = []
    for _ in range(num_grasps):
        # Sample a random translation within the offset sphere
        random_direction = np.random.uniform(-1, 1, 3)
        random_direction /= np.linalg.norm(random_direction)  # Normalize
        random_magnitude = np.random.uniform(0, offset)
        translation = object_position + random_direction * random_magnitude

        # Generate a random rotation
        random_rotation = R.random().as_matrix()

        assert random_rotation.shape == (3, 3)
        assert translation.shape == (3,)
        grasp_poses_list.append((random_rotation, translation))
    
    for i, (rotation, translation) in enumerate(grasp_poses_list):
        print(f"Grasp {i+1}:")
        print(f"Translation: {translation}")
        print(f"Rotation Matrix:\n{rotation}\n")
    
    if grasp_poses_list is None:
        print("No valid grasp found!")
        return False
    
    
    gripper_size = (0.08, 0.01, 0.04)

    # Create gripper meshes
    gripper_meshes = []
    for rotation, translation in grasp_poses_list:
        gripper = o3d.geometry.TriangleMesh.create_box(*gripper_size)
        gripper.compute_vertex_normals()

        # Center the mesh at the origin before applying transforms
        gripper.translate(-np.array(gripper_size) / 2)

        # Apply rotation and translation
        gripper.rotate(rotation, center=(0, 0, 0))
        gripper.translate(translation)

        gripper_meshes.append(gripper)

    # # Assign a color to the grippers
    # for mesh in gripper_meshes:
    #     mesh.paint_uniform_color([1, 0, 0])  # Red for gripper

    # o3d.visualization.draw_geometries([pcd] + gripper_meshes)

    collisions = sim.check_grasp_collision_with_pcd(gripper_meshes, pcd)

    # # Visualize only collision-free grippers
    # vis_objs = [pcd]
    # for mesh, collides in zip(gripper_meshes, collisions):
    #     if not collides:
    #         mesh.paint_uniform_color([0, 1, 0])  # Green for non-colliding
    #         vis_objs.append(mesh)
    #     else:
    #         mesh.paint_uniform_color([1, 0, 0])  # Red for collision (optional)

    # o3d.visualization.draw_geometries(vis_objs)


    # #Filtering and Visualizing Grasps
    # vis_objs = [pcd]

    # for pose, grasp_mesh in zip(grasp_poses_list, gripper_meshes):
    #     center_grasp = pose[1]  # Assuming pose[1] is the grasp center (np.ndarray)
    #     not_in_collision = not sim.check_grasp_collision_with_pcd(gripper_meshes, pcd)
    #     in_distance_range = sim.grasp_dist_filter(center_grasp, object_position)

    #     if not_in_collision and in_distance_range:
    #         vis_objs.extend(grasp_mesh)

    # o3d.visualization.draw_geometries(vis_objs)

    # the most suitable loccation for grasping is at
    # Grasp 6 is the best:
    # Distance ~5 cm from object center.

    # Gripper Z-axis points downward (approach vector ≈ -Z).

    # Best total score ≈ -0.665.

    # Scoring the grasp 

    
    # Step 4: Detect the object using yolo11

    # Grasp 6 details
    grasp_pos = np.array([0.0178195, -0.42063593,  0.54653101])
    grasp_orn = np.array([
        [ 0.82477544, -0.40806756, -0.39144136],
        [-0.37792016,  0.11713692, -0.91839822],
        [ 0.42062075,  0.90540588, -0.05760531]
    ])

    scoreing_grasps = sim.rate_grasps(grasp_poses_list, grasp_pos, grasp_orn)
    print(scoreing_grasps)
    # Step 5: Move down to the object
    print("Moving down to the object...")
    goal_position = np.array(grasp_pos)  # Move to the graspping position
    final_joint_angles = sim.control.ik_solver(
        goal_position=goal_position,
        goal_orientation=grasp_orn,  # Pass the rotation matrix
        initial_joint_angles=final_joint_angles,
        max_iterations=100,
        tolerance=1e-3,
        learning_rate=0.1
    )
    sim.robot.position_control(final_joint_angles)
    for _ in range(200):  # Wait for the robot to reach the position
        sim.step()

    return
    # # Step 6: Grasping the object with force control 
    # # Grasp 4 details
    # translation = np.array([-0.00186397, -0.42075018, 0.50045635])
    # rotation_matrix = np.array([
    #     [-0.00179586,  0.55073196, -0.83468023],
    #     [-0.93066812,  0.30445672,  0.20288655],
    #     [ 0.36586011,  0.77717464,  0.51200192]
    # ])

    # # Convert to quaternion using scipy
    # r = R.from_matrix(rotation_matrix)
    # quaternion = r.as_quat()  # Returns (x, y, z, w)

    # sim.grasp_with_pybullet(translation, quaternion)
    sim.grasp_until_force_threshold()


    print("Back in the pick and place function.....")

    # Step 7: Move back up
    print("Moving back up...")
    goal_position = np.array(object_position) + np.array([0, 0, 0.1])  # 10 cm above the object
    final_joint_angles = sim.control.ik_solver(
        goal_position=goal_position,
        goal_orientation=desired_orientation,  # Pass the rotation matrix
        initial_joint_angles=sim.robot.get_joint_positions(),
        max_iterations=100,
        tolerance=1e-3,
        learning_rate=0.1
    )
    sim.robot.position_control(final_joint_angles)
    for _ in range(200):  # Wait for the robot to move back up
        sim.step()

    # Step 8: Move to the target position
    print("Moving to the target position...")
    goal_position = np.array(target_position) + np.array([0, 0, 0.1])  # 10 cm above the target
    final_joint_angles = sim.control.ik_solver(
        goal_position=goal_position,
        goal_orientation=desired_orientation,  # Pass the rotation matrix
        initial_joint_angles=sim.robot.get_joint_positions(),
        max_iterations=100,
        tolerance=1e-3,
        learning_rate=0.1
    )
    sim.robot.position_control(final_joint_angles)
    for _ in range(200):  # Wait for the robot to reach the target position
        sim.step()

    # Step 9: Move down to the target position
    print("Moving down to the target position...")
    goal_position = np.array(target_position)  # Move to the target's exact position
    final_joint_angles = sim.control.ik_solver(
        goal_position=goal_position,
        goal_orientation=desired_orientation,  # Pass the rotation matrix
        initial_joint_angles=final_joint_angles,
        max_iterations=100,
        tolerance=1e-3,
        learning_rate=0.1
    )
    sim.robot.position_control(final_joint_angles)
    for _ in range(200):  # Wait for the robot to reach the target position
        sim.step()

    # Step 10: Open the gripper to release the object
    print("Opening the gripper to release the object...")
    sim.robot.open_gripper()
    for _ in range(200):  # Wait for the gripper to open
        sim.step()

    print("Pick-and-place operation completed.")


