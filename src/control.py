# To handle the grasping of the object, we need to implement an IK-solver. There are 2 methods to implement that

# Transpose method if Jacobian is not singular (steps: 
# Compute the Jacobian matrix 𝐽 ( 𝜃 ) J(θ). 
# Compute the transpose 𝐽 ( 𝜃 ) 𝑇 J(θ) T of the Jacobian. 
# Multiply the transpose 𝐽 ( 𝜃 ) 𝑇 J(θ) T by the desired end-effector velocity 𝑥 ˙ x ˙ to find the joint velocities 𝜃 ˙ θ ˙ .


# Pseudo-inverse method is more robust and ideal for complex configurations (steps:
# Compute the Jacobian matrix 𝐽 ( 𝜃 ) J(θ) for the current joint configuration. 
# Calculate the pseudo-inverse 𝐽 + ( 𝜃 ) J + (θ) using SVD or other methods. 
# Multiply the pseudo-inverse by the desired end-effector velocity 𝑥 ˙ x ˙ to find the joint velocities 𝜃 ˙ θ ˙ 


import yaml
import numpy as np
import pybullet as p

from typing import Dict, Any, Optional


TABLE_SCALING = 2.0

from scipy.spatial.transform import Rotation as R

from typing import Tuple, List
# from src.simulation import Simulation as sim
from .robot import Robot

# Load the robot settings from the YAML configuration
def load_robot_settings(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)  # Load the YAML
        print("Loaded config:", config)  # Debug: print the loaded configuration
        return config.get('robot_settings', {})  # Extract robot settings

robot_settings = load_robot_settings('configs/test_config.yaml')


class Control:
    def __init__(self):
        self.robot = Robot(urdf=robot_settings.get("urdf", "default.urdf"),
            init_position=robot_settings.get("default_init_pos", [0, 0, 0]),
            orientation=robot_settings.get("default_init_ori", [0, 0, 0]),
            arm_index=robot_settings.get("arm_idx", []),
            gripper_index=robot_settings.get("gripper_idx", []),
            ee_index=robot_settings.get("ee_idx", 0),
            arm_default=robot_settings.get("default_arm", []),
            table_scaling=robot_settings.get("table_scaling", TABLE_SCALING))
        

    def quaternion_to_rotation_matrix(self, quaternion):
        """
        Converts a quaternion to a rotation matrix.

        Args:
            quaternion (np.ndarray): A numpy array representing the quaternion in (x, y, z, w) format.

        Returns:
            np.ndarray: A 3x3 rotation matrix.
        """
        return R.from_quat(quaternion).as_matrix()

    def get_tf_mat(self, i, dh):
        a, d, alpha, theta = dh[i]
        q = theta
        return np.array([[np.cos(q), -np.sin(q), 0, a],
                         [np.sin(q) * np.cos(alpha), np.cos(q) * np.cos(alpha), -np.sin(alpha), -np.sin(alpha) * d],
                         [np.sin(q) * np.sin(alpha), np.cos(q) * np.sin(alpha), np.cos(alpha), np.cos(alpha) * d],
                         [0, 0, 0, 1]])

    def get_ee_pose(self, joint_angles):
        dh_params = np.array([[0, 0.333, 0, joint_angles[0]],
                              [0, 0, -np.pi / 2, joint_angles[1]],
                              [0, 0.316, np.pi / 2, joint_angles[2]],
                              [0.0825, 0, np.pi / 2, joint_angles[3]],
                              [-0.0825, 0.384, -np.pi / 2, joint_angles[4]],
                              [0, 0, np.pi / 2, joint_angles[5]],
                              [0.088, 0, np.pi / 2, joint_angles[6]]], dtype=np.float64)
        
        T_ee = np.identity(4)
        for i in range(7):
            T_ee = T_ee @ self.get_tf_mat(i, dh_params)
        
        return T_ee[:3, 3], T_ee[:3, :3]

    def euler_to_rotation_matrix(self, euler_angles):
        return R.from_euler('xyz', euler_angles).as_matrix()

    def get_jacobian(self, joint_angles):
        dh_params = np.array([[0, 0.333, 0, joint_angles[0]],
                               [0, 0, -np.pi / 2, joint_angles[1]],
                               [0, 0.316, np.pi / 2, joint_angles[2]],
                               [0.0825, 0, np.pi / 2, joint_angles[3]],
                               [-0.0825, 0.384, -np.pi / 2, joint_angles[4]],
                               [0, 0, np.pi / 2, joint_angles[5]],
                               [0.088, 0, np.pi / 2, joint_angles[6]]], dtype=np.float64)
        
        EE_Transformation = np.identity(4)
        for i in range(7):
            EE_Transformation = EE_Transformation @ self.get_tf_mat(i, dh_params)

        J = np.zeros((6, 7))
        T = np.identity(4)
        for i in range(7):
            T = T @ self.get_tf_mat(i, dh_params)
            p = EE_Transformation[:3, 3] - T[:3, 3]
            z = T[:3, 2]
            J[:3, i] = np.cross(z, p)
            J[3:, i] = z
        return J

    def pseudo_inverse(self, J):
        U, S, Vt = np.linalg.svd(J)
        S_inv = np.zeros((J.shape[1], J.shape[0]))
        for i in range(len(S)):
            if S[i] > 1e-6:
                S_inv[i, i] = 1.0 / S[i]
        return Vt.T @ S_inv @ U.T
    
    def orientation_error(self, current_rot, goal_rot):
        Re = np.dot(goal_rot, current_rot.T)
        error = np.array([Re[2,1] - Re[1,2],
                        Re[0,2] - Re[2,0],
                        Re[1,0] - Re[0,1]])
        return error
    
    def compute_joint_velocities(self, J, x_dot):
        return self.pseudo_inverse(J) @ x_dot

    def forward_kinematics(self, joint_angles):
        """
        Placeholder for forward kinematics calculation.  Replace with your robot's FK.
        Given joint angles, return the end-effector pose (position and rotation).
        """
        # This is a simplified example.  In a real robot, you would use DH parameters
        # or other kinematic models to calculate the end-effector pose.
        # Assuming a simple serial chain robot
        x = np.sum(self.link_lengths * np.cos(np.cumsum(joint_angles)))
        y = np.sum(self.link_lengths * np.sin(np.cumsum(joint_angles)))
        z = 1.0  # Assume constant height
        rotation = R.from_euler('z', np.sum(joint_angles))
        return np.array([x, y, z]), rotation
    


    def ik_solver(self, goal_position, goal_orientation, initial_joint_angles, 
                max_iterations=100, tolerance=1e-4, learning_rate=0.05):
        joint_angles = initial_joint_angles.copy()
        
        for iteration in range(max_iterations):
            # Get current pose and errors
            ee_pos, ee_rot = self.get_ee_pose(joint_angles)
            pos_error = goal_position - ee_pos
            orn_error = self.orientation_error(ee_rot, goal_orientation)
            
            # Normalize and weight errors
            pos_error *= 0.5  # Reduce position dominance
            error = np.concatenate([pos_error, orn_error])
            
            # Check convergence
            if np.linalg.norm(error) < tolerance:
                return joint_angles
            
            # Compute Jacobian with damped least squares
            J = self.get_jacobian(joint_angles)
            damping = 0.1  # Critical for stability
            J_pinv = J.T @ np.linalg.inv(J @ J.T + damping**2 * np.eye(6))
            
            # Apply update with joint limits
            joint_angles += learning_rate * (J_pinv @ error)
            joint_angles = np.clip(joint_angles, 
                                self.robot.lower_limits, 
                                self.robot.upper_limits)
        
        print("IK Warning: Max iterations reached")
        return joint_angles

    def generate_trajectory(self, start_pose, end_pose, steps=10):
        """Generate Cartesian-space waypoints"""
        waypoints = []
        for t in np.linspace(0, 1, steps):
            pos = start_pose[0] + t*(end_pose[0] - start_pose[0])
            rot = R.from_matrix(start_pose[1]).slerp(t, R.from_matrix(end_pose[1]))
            waypoints.append((pos, rot.as_matrix()))
        return waypoints
    
    def execute_trajectory(self, waypoints, max_velocity=0.3):
        current_joints = self.robot.get_joint_positions()
        
        for pos, rot in waypoints:
            # Solve IK for waypoint
            target_joints = self.ik_solver(pos, rot, current_joints)
            
            # Move with velocity control
            error = target_joints - current_joints
            velocities = np.clip(error * 5.0, -max_velocity, max_velocity)
            
            for _ in range(10):  # Control loop
                self.robot.velocity_control(velocities)
                current_joints = self.robot.get_joint_positions()
                new_error = target_joints - current_joints
                velocities = np.clip(new_error * 5.0, -max_velocity, max_velocity)
                p.stepSimulation()
                
                # Visualize (debug)
                p.addUserDebugLine(
                    self.get_ee_pose(current_joints)[0],
                    pos, [0,1,0], lifeTime=0.2)
                
    def reinforce_grasp(self):
        """Increase grip force during motion"""
        self.robot.close_gripper(force=30)  # Normal force
        for _ in range(50):
            p.stepSimulation()  # Let physics settle
            
        # Add sideways stabilization force
        gripper_pos = self.robot.get_ee_pose()[0]
        p.applyExternalForce(
            objectUniqueId=self.gripper_id,
            linkIndex=self.robot.ee_idx,
            forceObj=[0, 0, -5],  # Small downward force
            posObj=gripper_pos,
            flags=p.WORLD_FRAME
        )






