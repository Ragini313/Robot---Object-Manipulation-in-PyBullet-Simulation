import pybullet as p
import numpy as np
from typing import Tuple, List


TABLE_SCALING = 2.0


class Robot:
    """Robot Class.

    The class initializes a franka robot.

    Args:
        urdf: Path to the robot URDF file
        init_position: Initial position (x,y,z) of the robot base
        orientation: Robot orientation in axis angle representation
                        (roll,pitch,yaw)
        arm_index: List of joint indices for the robot arm
        gripper_index: List of joint indices for the gripper
        ee_index: Index of the end effector link
        arm_default: List of default joint angles for the robot arm and gripper
        table_scaling: Scaling parameter for the table height ^ size
    """
    def __init__(self,
                 urdf: str,
                 init_position: Tuple[float, float, float],
                 orientation: Tuple[float, float, float],
                 arm_index: List[int],
                 gripper_index: List[int],
                 ee_index: int,
                 arm_default: List[float],
                 table_scaling: float = TABLE_SCALING):

        # load robot
        self.pos = init_position
        self.axis_angle = orientation
        self.tscale = table_scaling

        if self.tscale != 1.0:
            self.pos = [self.pos[0], self.pos[1], self.pos[2] * self.tscale]
        self.ori = p.getQuaternionFromEuler(self.axis_angle)

        self.arm_idx = arm_index
        self.default_arm = arm_default
        self.gripper_idx = gripper_index

        self.ee_idx = ee_index

        

        self.id = p.loadURDF(urdf, self.pos, self.ori,
                             useFixedBase=True)

        self.lower_limits, self.upper_limits = self.get_joint_limits()

        self.set_default_position()

        for j in range(p.getNumJoints(self.id)):
            p.changeDynamics(self.id, j, linearDamping=0, angularDamping=0)

    def set_default_position(self):
        for idx, pos in zip(self.arm_idx, self.default_arm[:-2]):
            p.resetJointState(self.id, idx, pos)

        for grp_idx, pos in zip(self.gripper_idx, self.default_arm[-2:]):
            p.resetJointState(self.id, grp_idx, pos)

    def get_joint_limits(self):
        lower = []
        upper = []
        for idx in self.arm_idx:
            joint_info = p.getJointInfo(self.id, idx)
            lower.append(joint_info[8])
            upper.append(joint_info[9])
        return lower, upper

    def print_joint_infos(self):
        num_joints = p.getNumJoints(self.id)
        print('number of joints are: {}'.format(num_joints))
        for i in range(0, num_joints):
            print('Index: {}'.format(p.getJointInfo(self.id, i)[0]))
            print('Name: {}'.format(p.getJointInfo(self.id, i)[1]))
            print('Typ: {}'.format(p.getJointInfo(self.id, i)[2]))

    def get_joint_positions(self):
        states = p.getJointStates(self.id, self.arm_idx)
        return np.array([state[0] for state in states])

    def get_joint_velocites(self):
        states = p.getJointStates(self.id, self.arm_idx)
        return np.array([state[1] for state in states])

    def get_ee_pose(self):
        ee_info = p.getLinkState(self.id, self.ee_idx)
        ee_pos = ee_info[0]
        ee_ori = ee_info[1]
        return np.asarray(ee_pos), np.asarray(ee_ori)

    def position_control(self, target_positions):
        p.setJointMotorControlArray(
            self.id,
            jointIndices=self.arm_idx,
            controlMode=p.POSITION_CONTROL,
            targetPositions=target_positions,
        )
       
    ### my added helper function:

    def open_gripper(self):
        """
        Open the gripper by setting the gripper joints to their maximum limits.
        """
        gripper_opening = []
        for idx in self.gripper_idx:
            joint_info = p.getJointInfo(self.id, idx)
            upper_limit = joint_info[9]  # Upper limit of the joint
            gripper_opening.append(upper_limit)
        
        p.setJointMotorControlArray(
            self.id,
            jointIndices=self.gripper_idx,
            controlMode=p.POSITION_CONTROL,
            targetPositions=gripper_opening,
            forces=[100] * len(self.gripper_idx)  # Adjust force as needed
        )

    def close_gripper(self):
        """
        Close the gripper by setting the gripper joints to their minimum limits.
        """
        gripper_closing = []
        for idx in self.gripper_idx:
            joint_info = p.getJointInfo(self.id, idx)
            lower_limit = joint_info[8]  # Lower limit of the joint
            gripper_closing.append(lower_limit)
        
        p.setJointMotorControlArray(
            self.id,
            jointIndices=self.gripper_idx,
            controlMode=p.POSITION_CONTROL,
            targetPositions=gripper_closing,
            forces=[200] * len(self.gripper_idx)  # Adjust force as needed
        )

    
    def get_link_positions(self):
        """Get positions of all robot links"""
        link_positions = []
        for i in range(p.getNumJoints(self.id)):
            link_state = p.getLinkState(self.id, i)
            link_positions.append(np.array(link_state[0]))  # [0] is link world position
        return link_positions




