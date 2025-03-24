import os
import glob
import yaml
import numpy as np
import pybullet as p

from typing import Dict, Any
from pybullet_object_models import ycb_objects  # type:ignore
#from src.simulation import Simulation

from src.simulation import Simulation as sim

def move_down_to_object(self, goal_position, initial_joint_angles, step_size=0.01, max_steps=70):
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
            max_iterations=30,
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
        max_iterations=70,
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
        max_iterations=60,    ## reduced from 100 to 50 because program freezes after 47 iterations
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
    3. Close the gripper.
    4. Move back up.
    5. Move to the target position.
    6. Open the gripper to release the object.
    """
    # Define the desired orientation (identity quaternion: no rotation)
    desired_quaternion = np.array([1.0, 0.0, 0.0, 0.0])  # Quaternion (w, x, y, z) - 1.0, 0.0, 0.0, 0.0
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

    # Step 3: Move down to the object
    print("Moving down to the object...")
    goal_position = np.array(object_position) - np.array([0, 0, 0.23]) # Move to the object's exact position but 20 cm below the object
    final_joint_angles = sim.control.ik_solver(
        goal_position=goal_position,
        goal_orientation=desired_orientation,  # Pass the rotation matrix
        initial_joint_angles=final_joint_angles,
        max_iterations=70,
        tolerance=1e-3,
        learning_rate=0.1
    )
    sim.robot.position_control(final_joint_angles)
    for _ in range(200):  # Wait for the robot to reach the position
        sim.step()

    # Step 4: Close the gripper
    print("Closing the gripper...")
    sim.robot.close_gripper()
    for _ in range(200):  # Wait for the gripper to close
        sim.step()

    # Step 5: Move back up
    print("Moving back up...")
    goal_position = np.array(object_position) + np.array([0, 0, 0.1])  # 10 cm above the object
    final_joint_angles = sim.control.ik_solver(
        goal_position=goal_position,
        goal_orientation=desired_orientation,  # Pass the rotation matrix
        initial_joint_angles=sim.robot.get_joint_positions(),
        max_iterations=70,
        tolerance=1e-3,
        learning_rate=0.1
    )
    sim.robot.position_control(final_joint_angles)
    for _ in range(200):  # Wait for the robot to move back up
        sim.step()

    # Step 6: Move to the target position
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

    # Step 7: Move down to the target position
    print("Moving down to the target position...")
    goal_position = np.array(target_position)  # Move to the target's exact position
    final_joint_angles = sim.control.ik_solver(
        goal_position=goal_position,
        goal_orientation=desired_orientation,  # Pass the rotation matrix
        initial_joint_angles=final_joint_angles,
        max_iterations=70,
        tolerance=1e-3,
        learning_rate=0.1
    )
    sim.robot.position_control(final_joint_angles)
    for _ in range(200):  # Wait for the robot to reach the target position
        sim.step()

    # Step 8: Open the gripper to release the object
    print("Opening the gripper to release the object...")
    sim.robot.open_gripper()
    for _ in range(200):  # Wait for the gripper to open
        sim.step()

    print("Pick-and-place operation completed.")

