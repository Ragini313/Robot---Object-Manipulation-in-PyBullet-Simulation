# To handle the grasping of the object, we need to implement an IK-solver. There are 2 methods to implement that

# Transpose method if Jacobian is not singular (steps: 
# Compute the Jacobian matrix 𝐽 ( 𝜃 ) J(θ). 
# Compute the transpose 𝐽 ( 𝜃 ) 𝑇 J(θ) T of the Jacobian. 
# Multiply the transpose 𝐽 ( 𝜃 ) 𝑇 J(θ) T by the desired end-effector velocity 𝑥 ˙ x ˙ to find the joint velocities 𝜃 ˙ θ ˙ .


# Pseudo-inverse method is more robust and ideal for complex configurations (steps:
# Compute the Jacobian matrix 𝐽 ( 𝜃 ) J(θ) for the current joint configuration. 
# Calculate the pseudo-inverse 𝐽 + ( 𝜃 ) J + (θ) using SVD or other methods. 
# Multiply the pseudo-inverse by the desired end-effector velocity 𝑥 ˙ x ˙ to find the joint velocities 𝜃 ˙ θ ˙ 

# Our choice of method = Pseudo-inverse
import yaml
import numpy as np
import pybullet as p
#import robotic as ry
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
        # Initialize robotic Config
        #self.C = ry.Config()
        #self.C.addFile(ry.raiPath(robot_settings.get("urdf", "default.urdf"))) # Load the URDF into the robotic Config

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
    
    def ik_solver(self, goal_position, goal_orientation, initial_joint_angles, max_iterations=10, tolerance=1e-3, learning_rate=0.1):
        joint_angles = initial_joint_angles.copy()
        for iteration in range(max_iterations):
            ee_pos, ee_rot = self.get_ee_pose(joint_angles)
            position_error = goal_position - ee_pos
        
            # Calculate orientation error using rotation matrices
            orientation_error = self.orientation_error(ee_rot, goal_orientation)
        
            # Combine position and orientation errors
            error = np.concatenate([position_error, orientation_error])
        
            if np.linalg.norm(error) < tolerance:
                print(f"Converged after {iteration} iterations.")
                return joint_angles
            J = self.get_jacobian(joint_angles)
            joint_velocities = self.compute_joint_velocities(J, error)
            joint_angles += learning_rate * joint_velocities
            print(f"Max iterations reached. Final joint angles: {joint_angles}")
        return joint_angles


def main():
    # Define the target position and orientation
    object_pose = (-0.050183952461055004, -0.46971427743603356, 1.3231258620680433)
    object_orientation = (0.0, 0.0, 0.0, 1.0)  # Quaternion (x, y, z, w)

    # Convert quaternion to rotation matrix
    goal_orientation_matrix = R.from_quat(object_orientation).as_matrix()

    # Initialize the Control class
    control = Control()
    robot = Robot()

    # Get the initial joint angles from the robot (assuming the robot has a method for this)
    initial_joint_angles = robot.get_joint_positions()  # Replace with actual initial joint angles if available
    print(initial_joint_angles)
    # # Solve for joint angles using inverse kinematics
    # final_joint_angles = control.ik_solver(
    #     goal_position=np.array(object_pose),
    #     goal_orientation=goal_orientation_matrix,
    #     initial_joint_angles=initial_joint_angles,
    #     max_iterations=100,
    #     tolerance=1e-3,
    #     learning_rate=0.1
    # )

    # print("Final joint angles to reach the target pose:", final_joint_angles)

if __name__ == "__main__":
    main()    




