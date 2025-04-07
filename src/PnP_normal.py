import os
import glob
import yaml
import numpy as np
import pybullet as p

from typing import Dict, Any
from pybullet_object_models import ycb_objects  # type:ignore

from src.simulation import Simulation as sim


#from src.robot import get_link_positions

from src.grasping import* ##check_grasp_collision_with_pcd, grasp_dist_filter, grasp_until_force_threshold, force_control_gripper, rate_grasps, grasp_with_pybullet


pick_and_place_completed = False

def wait_until_reached(sim, target_joints, position_tolerance=0.01, orientation_tolerance=0.1, max_steps=500):
    """Wait until the robot reaches the target joint positions within tolerance."""
    for _ in range(max_steps):
        current_joints = sim.robot.get_joint_positions()
        error = np.linalg.norm(current_joints - target_joints)
        if error < position_tolerance:
            return True
        sim.step()
    print(f"Warning: Reached max steps without converging (error: {error})")
    return False

def generate_trajectory(start_pos, end_pos, num_points=10):
    """Generate a linear trajectory between two points"""
    return np.linspace(start_pos, end_pos, num_points)




def pause_simulation(sim, steps=200):
    """Simple fixed-duration pause"""
    for _ in range(steps):
        sim.step()  # Just step the simulation without doing anything else

def pick_and_place(sim, object_position, target_position):
    """
    Perform a pick-and-place operation with explicit height transitions:
    ** Need to wait until YCB object finally lands on the ground, or else robot clashes with object and moves it further
    1. Move to 0.1m above object
    2. Move down 0.1m to grasp position
    3. Grasp object
    4. Lift back up 0.1m
    5. Move to 0.1m above target
    6. Lower to target position
    7. Release object
    8. Lift back up 0.1m
    """

    global pick_and_place_completed

    # Define the desired orientation (identity quaternion: no rotation)
    desired_quaternion = np.array([1.0, 0.0, 0.0, 0.0])
    #desired_quaternion = np.array([0.0, 0.0, 0.0, 1.0])
    desired_orientation = sim.control.quaternion_to_rotation_matrix(desired_quaternion)
    
    # Get initial joint positions
    initial_joints = sim.robot.get_joint_positions()

    # ----- Initial Pause -----
    print("Pausing for object to settle...")
    pause_simulation(sim, steps=200)  # Simple fixed-duration pause
    
    # ----- Approach Phase (0.1m above object) -----
    print("Moving to 0.1m above object...")
    approach_height = 0.18  # 10cm above object
    approach_position = np.array(object_position) + np.array([0, 0, approach_height])
    
    approach_joints = sim.control.ik_solver(
        goal_position=approach_position,
        goal_orientation=desired_orientation,
        initial_joint_angles=initial_joints,
        max_iterations=200,
        tolerance=1e-3,
        learning_rate=0.1
    )
    
    sim.robot.position_control(approach_joints)
    wait_until_reached(sim, approach_joints)
    
    # Brief pause at approach height
    print("Pausing at approach height...")
    for _ in range(5):  # Wait for 50 simulation steps
        sim.step()
    
    # ----- Pre-Grasp Phase (move down 0.1m to object) -----
    print("Moving down to grasp position...")
    grasp_position = np.array(object_position)  # Original object position
    
    grasp_joints = sim.control.ik_solver(
        goal_position=grasp_position,
        goal_orientation=desired_orientation,
        initial_joint_angles=approach_joints,
        max_iterations=200,
        tolerance=1e-3,
        learning_rate=0.1
    )
    
    # Open gripper before moving down
    sim.robot.open_gripper()
    sim.robot.position_control(grasp_joints)
    wait_until_reached(sim, grasp_joints, position_tolerance=0.005)
    
    # ----- Grasp Phase -----
    print("Closing gripper...")
    sim.robot.close_gripper()
    for _ in range(100):  # Wait for grasp to complete
        sim.step()

    ### ----Gripper control to avoid object being flung out during transport

    # ----- Grip Reinforcement -----
    def reinforce_grip():
        # Apply additional stabilizing forces
        gripper_pos = sim.robot.get_ee_pose()[0]
        p.applyExternalForce(
            objectUniqueId=sim.robot.ee_idx,
            linkIndex=-1,
            forceObj=[0, 0, -15],  # Small downward force
            posObj=gripper_pos,
            flags=p.WORLD_FRAME
        )
        sim.step()

    # ----- Lift Phase -----
    print("Lifting object carefully...")
    lift_position = np.array(object_position) + np.array([0, 0, 0.1])
    current_joints = sim.robot.get_joint_positions()
    
    lift_joints = sim.control.ik_solver(
        goal_position=lift_position,
        goal_orientation=desired_orientation,
        initial_joint_angles=current_joints,
        max_iterations=200,
        tolerance=1e-3,
        learning_rate=0.1
    )

    # Slower movement by breaking into smaller increments
    steps = 10  # More steps = slower movement
    for i in range(1, steps+1):
        # Interpolate between current and target joints
        intermediate_joints = current_joints + (lift_joints - current_joints) * (i/steps)
        sim.robot.position_control(intermediate_joints)
        reinforce_grip()
        for _ in range(5):  # Multiple simulation steps per control step
            sim.step()

    # ----- Transport Phase with Collision Avoidance -----
    print("Transporting object carefully...")
    transport_position = np.array(target_position) + np.array([0, 0, 0.1])
    current_joints = sim.robot.get_joint_positions()
    
    transport_joints = sim.control.ik_solver(
        goal_position=transport_position,
        goal_orientation=desired_orientation,
        initial_joint_angles=current_joints,
        max_iterations=1000,
        tolerance=1e-4,
        learning_rate=0.05
    )

    # Break transport into smaller segments
    steps = 15  # Even more steps for longer movement
    for i in range(1, steps+1):
        intermediate_joints = current_joints + (transport_joints - current_joints) * (i/steps)
        sim.robot.position_control(intermediate_joints)
        reinforce_grip()
        for _ in range(5):  # Multiple simulation steps per control step
            sim.step()

    ### --- End of Gripper control
    
    # Verify grasp (pseudo-code - implement based on your setup)
    if not verify_grasp(sim):
        print("Grasp verification failed!")
        return False
    
  
      
    # ----- Lower Phase (to target position) -----
    print("Carefully lowering to target position...")
    place_position = np.array(target_position) + np.array([0, 0, approach_height])  
    
    # Slow, controlled descent
    current_joints = sim.robot.get_joint_positions()
    place_joints = sim.control.ik_solver(
        goal_position=place_position,
        goal_orientation=desired_orientation,
        initial_joint_angles=current_joints,
        max_iterations=300,
        tolerance=1e-3,
        learning_rate=0.1
    )
    
    # Break into smaller steps
    for i in range(1, 3):
        intermediate_joints = current_joints + (place_joints - current_joints) * (i/5)
        sim.robot.position_control(intermediate_joints)
        reinforce_grip()  # Maintain grip during descent
        for _ in range(10):
            sim.step()


    # # Open gripper 
    sim.robot.open_gripper()
    pause_simulation(sim, steps=200)


    # # ----- Final Retreat to go back to initial position ----- def set_default_position
    # print("Slowly retreating...")
    # retreat_position = sim.robot.set_default_position
    # current_joints = sim.robot.get_joint_positions()
    
    # retreat_joints = sim.control.ik_solver(
    #     goal_position=retreat_position,
    #     goal_orientation=desired_orientation,
    #     initial_joint_angles=current_joints,
    #     max_iterations=100,
    #     tolerance=1e-3,
    #     learning_rate=0.1
    # )
    
    # # Slow retreat motion
    # for i in range(1, 11):
    #     intermediate_joints = current_joints + (retreat_joints - current_joints) * (i/10)
    #     sim.robot.position_control(intermediate_joints)
    #     for _ in range(10):
    #         sim.step()

    # # Brief pause at end 
    # print("Pausing at end of pick n place...")
    # for _ in range(5):  # Wait for 50 simulation steps
    #     sim.step()

    print("Pick n Place Operation completed successfully.")

    ## To ensure the pick n place only runs once
    pick_and_place_completed = True
    return True

def verify_grasp(sim):
    """
    Simple grasp verification - implement based on your setup
    Options:
    1. Check if gripper position indicates it closed properly
    2. Check if object is moving with gripper
    3. Use force/torque sensors if available
    """
    # Placeholder - implement rafey's grasping logic
    return True

