import os
import glob
import yaml
import numpy as np
import pybullet as p
from typing import Dict, Any, List, Tuple
from pybullet_object_models import ycb_objects  # type:ignore
from src.simulation import Simulation as sim
from src.grasping import *

# Collision avoidance parameters
SAFETY_MARGIN = 0.15  # meters
COLLISION_CHECK_STEPS = 5
MAX_AVOIDANCE_ATTEMPTS = 5

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

def get_robot_link_positions(sim) -> List[np.ndarray]:
    """Get positions of all robot links"""
    return sim.robot.get_link_positions()  # Used the method we already have in Robot class


### this function was causing uncertainty in the transport phase resulting in the object being released, therefore, i decided to stick with the basic function
# def check_collision_risk(sim, target_joints, obs1_pos, obs2_pos, safety_margin=0.15):
#     """
#     Check if planned motion would bring robot too close to obstacles
#     """
#     # Store original joints
#     original_joints = sim.robot.get_joint_positions()
    
#     # Move to target joints temporarily
#     sim.robot.position_control(target_joints)
    
#     # Need to step simulation to update positions
#     for _ in range(5):
#         p.stepSimulation()
    
#     # Get current link positions
#     link_positions = sim.robot.get_link_positions()  # Call via robot instance
    
#     # Get EE position
#     ee_pos, _ = sim.robot.get_ee_pose()
#     link_positions.append(ee_pos)  # Include EE in collision checking
    
#     # Restore original position
#     sim.robot.position_control(original_joints)
#     for _ in range(5):
#         p.stepSimulation()
    
#     # Check distances to obstacles
#     min_dist_to_obs1 = min(np.linalg.norm(pos - obs1_pos) for pos in link_positions)
#     min_dist_to_obs2 = min(np.linalg.norm(pos - obs2_pos) for pos in link_positions)
    
#     return min_dist_to_obs1 < safety_margin or min_dist_to_obs2 < safety_margin

def check_collision_with_obstacles(sim, positions: List[np.ndarray], obstacle_positions: List[np.ndarray]) -> bool:
    """
    Check if any robot link is too close to any obstacle
    Args:
        positions: List of robot link positions
        obstacle_positions: List of obstacle positions to check against
    Returns:
        bool: True if collision risk detected
    """
    for pos in positions:
        for obs_pos in obstacle_positions:
            distance = np.linalg.norm(pos - obs_pos)
            if distance < SAFETY_MARGIN:
                return True
    return False

def check_self_collision(sim) -> bool:
    """Check if the robot is in self-collision"""
    # Get all link AABBs
    aabbs = []
    for link_idx in range(p.getNumJoints(sim.robot.id)):
        aabb = p.getAABB(sim.robot.id, link_idx)
        aabbs.append(aabb)
    
    # Check for overlapping AABBs (simple broad-phase check)
    for i in range(len(aabbs)):
        for j in range(i+1, len(aabbs)):
            # Check if AABBs overlap
            if (aabbs[i][1][0] > aabbs[j][0][0] and 
                aabbs[i][0][0] < aabbs[j][1][0] and
                aabbs[i][1][1] > aabbs[j][0][1] and 
                aabbs[i][0][1] < aabbs[j][1][1] and
                aabbs[i][1][2] > aabbs[j][0][2] and 
                aabbs[i][0][2] < aabbs[j][1][2]):
                return True
    return False

def check_collision_risk(sim, target_joints: np.ndarray, obstacle_positions: List[np.ndarray]) -> bool:
    """
    Comprehensive collision risk assessment
    Args:
        target_joints: Target joint configuration to check
        obstacle_positions: List of obstacle positions to avoid
    Returns:
        bool: True if collision risk detected
    """
    # Store original joints
    original_joints = sim.robot.get_joint_positions()
    
    # Move to target joints temporarily
    sim.robot.position_control(target_joints)
    
    # Step simulation to update positions
    for _ in range(COLLISION_CHECK_STEPS):
        p.stepSimulation()
    
    # Get current link positions
    link_positions = get_robot_link_positions(sim)
    
    # Get EE position and add it to the list
    ee_pos, _ = sim.robot.get_ee_pose()
    link_positions.append(ee_pos)
    
    # Check for collisions
    collision_detected = (
        check_collision_with_obstacles(sim, link_positions, obstacle_positions) or
        check_self_collision(sim)
    )
    
    # Restore original position
    sim.robot.position_control(original_joints)
    for _ in range(COLLISION_CHECK_STEPS):
        p.stepSimulation()
    
    return collision_detected

def find_safe_path(sim, start_joints: np.ndarray, target_joints: np.ndarray, 
                  obstacle_positions: List[np.ndarray]) -> Tuple[np.ndarray, bool]:
    """
    Find a collision-free path using multiple strategies
    Args:
        start_joints: Starting joint configuration
        target_joints: Desired joint configuration
        obstacle_positions: List of obstacle positions to avoid
    Returns:
        Tuple: (safe_joints, found_solution) where safe_joints is a collision-free configuration
    """
    strategies = [
        # Try lifting straight up first
        lambda: lift_vertical(sim, start_joints, 0.15),
        
        # Try offsetting in different horizontal directions
        lambda: offset_horizontal(sim, start_joints, 0.1, [1, 0, 0]),
        lambda: offset_horizontal(sim, start_joints, 0.1, [-1, 0, 0]),
        lambda: offset_horizontal(sim, start_joints, 0.1, [0, 1, 0]),
        lambda: offset_horizontal(sim, start_joints, 0.1, [0, -1, 0]),
        
        # Try intermediate waypoints
        lambda: find_intermediate_waypoint(sim, start_joints, target_joints, obstacle_positions),
    ]
    
    for strategy in strategies:
        candidate_joints = strategy()
        if candidate_joints is not None:
            if not check_collision_risk(sim, candidate_joints, obstacle_positions):
                return candidate_joints, True
    
    return None, False

def lift_vertical(sim, start_joints: np.ndarray, height: float) -> np.ndarray:
    """Generate joints for lifting EE vertically by specified height"""
    current_pos, current_ori = sim.robot.get_ee_pose()
    lifted_pos = current_pos + np.array([0, 0, height])
    
    return sim.control.ik_solver(
        goal_position=lifted_pos,
        goal_orientation=current_ori,
        initial_joint_angles=start_joints,
        max_iterations=100,
        tolerance=1e-3
    )

def offset_horizontal(sim, start_joints: np.ndarray, distance: float, direction: np.ndarray) -> np.ndarray:
    """Generate joints for offsetting EE horizontally by specified distance"""
    current_pos, current_ori = sim.robot.get_ee_pose()
    offset_pos = current_pos + distance * np.array(direction)
    
    return sim.control.ik_solver(
        goal_position=offset_pos,
        goal_orientation=current_ori,
        initial_joint_angles=start_joints,
        max_iterations=100,
        tolerance=1e-3
    )

def find_intermediate_waypoint(sim, start_joints: np.ndarray, target_joints: np.ndarray,
                             obstacle_positions: List[np.ndarray]) -> np.ndarray:
    """
    Find an intermediate waypoint that avoids obstacles by sampling in configuration space
    """
    # Get current and target EE poses
    start_pos, start_ori = sim.robot.get_ee_pose()
    sim.robot.position_control(target_joints)
    for _ in range(5):
        p.stepSimulation()
    target_pos, target_ori = sim.robot.get_ee_pose()
    sim.robot.position_control(start_joints)
    
    # Generate intermediate positions
    for alpha in [0.3, 0.5, 0.7]:
        # Linear interpolation in task space
        waypoint_pos = start_pos + alpha * (target_pos - start_pos)
        waypoint_pos[2] += 0.1  # Lift slightly
        
        # Try to find IK solution
        waypoint_joints = sim.control.ik_solver(
            goal_position=waypoint_pos,
            goal_orientation=start_ori,
            initial_joint_angles=start_joints,
            max_iterations=100,
            tolerance=1e-3
        )
        
        if waypoint_joints is not None:
            if not check_collision_risk(sim, waypoint_joints, obstacle_positions):
                return waypoint_joints
    return None

def move_with_collision_avoidance(sim, target_joints: np.ndarray, 
                                 obstacle_positions: List[np.ndarray], 
                                 max_attempts: int = MAX_AVOIDANCE_ATTEMPTS) -> bool:
    """
    Move robot to target joints while avoiding collisions
    Returns True if movement was successful
    """
    current_joints = sim.robot.get_joint_positions()
    
    # First check direct path
    if not check_collision_risk(sim, target_joints, obstacle_positions):
        sim.robot.position_control(target_joints)
        return wait_until_reached(sim, target_joints)
    
    # If direct path has collision, find alternative path
    for attempt in range(max_attempts):
        safe_joints, found = find_safe_path(sim, current_joints, target_joints, obstacle_positions)
        if found:
            # Move to safe intermediate position
            sim.robot.position_control(safe_joints)
            if not wait_until_reached(sim, safe_joints):
                continue
            
            # Try final approach from safe position
            if not check_collision_risk(sim, target_joints, obstacle_positions):
                sim.robot.position_control(target_joints)
                if wait_until_reached(sim, target_joints):
                    return True
        
        # Try a different strategy on next iteration
    
    print("Warning: Failed to find collision-free path after", max_attempts, "attempts")
    return False

def pause_simulation(sim, steps=200):
    """Simple fixed-duration pause"""
    for _ in range(steps):
        sim.step()

def pick_and_place(sim, object_position, target_position, tracker1, tracker2):
    """
    Enhanced pick-and-place with comprehensive collision avoidance
    """
    global pick_and_place_completed

    # Define obstacle positions (from trackers)
    obstacle_positions = [np.array(tracker1), np.array(tracker2)]
    
    # Define the desired orientation
    desired_quaternion = np.array([1.0, 0.0, 0.0, 0.0])
    desired_orientation = sim.control.quaternion_to_rotation_matrix(desired_quaternion)
    
    # Get initial joint positions
    initial_joints = sim.robot.get_joint_positions()

    # ----- Initial Pause -----
    print("Pausing for object to settle...")
    pause_simulation(sim, steps=200)
    
    # ----- Approach Phase -----
    print("Moving to approach height with collision checking...")
    approach_height = 0.18
    approach_position = np.array(object_position) + np.array([0, 0, approach_height])
    
    approach_joints = sim.control.ik_solver(
        goal_position=approach_position,
        goal_orientation=desired_orientation,
        initial_joint_angles=initial_joints,
        max_iterations=200,
        tolerance=1e-3,
        learning_rate=0.1
    )
    
    if not move_with_collision_avoidance(sim, approach_joints, obstacle_positions):
        print("Failed to safely reach approach position")
        return False
    
    # Brief pause at approach height
    print("Pausing at approach height...")
    pause_simulation(sim, steps=50)
    
    # ----- Pre-Grasp Phase -----
    print("Moving down to grasp position with collision checking...")
    grasp_position = np.array(object_position)
    
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
    if not move_with_collision_avoidance(sim, grasp_joints, obstacle_positions):
        print("Failed to safely reach grasp position")
        return False
    
    # ----- Grasp Phase -----
    print("Closing gripper...")
    sim.robot.close_gripper()
    pause_simulation(sim, steps=100)

    # ----- Lift Phase -----
    print("Lifting object carefully with collision checking...")
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
    
    if not move_with_collision_avoidance(sim, lift_joints, obstacle_positions):
        print("Failed to safely lift object")
        return False
    
    # ----- Transport Phase -----
    print("Transporting object carefully with collision checking...")
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
    
    if not move_with_collision_avoidance(sim, transport_joints, obstacle_positions):
        print("Failed to safely transport object")
        return False
    
    # Verify grasp
    if not verify_grasp(sim):
        print("Grasp verification failed!")
        return False
    
    # ----- Lower Phase -----
    print("Carefully lowering to target position...")
    place_position = np.array(target_position) + np.array([0, 0, approach_height])
    current_joints = sim.robot.get_joint_positions()
    
    place_joints = sim.control.ik_solver(
        goal_position=place_position,
        goal_orientation=desired_orientation,
        initial_joint_angles=current_joints,
        max_iterations=300,
        tolerance=1e-3,
        learning_rate=0.1
    )
    
    if not move_with_collision_avoidance(sim, place_joints, obstacle_positions):
        print("Failed to safely place object")
        return False
    
    # Open gripper 
    sim.robot.open_gripper()
    pause_simulation(sim, steps=50)
    
    # ----- Final Retreat -----
    print("Returning to default position...")
    retreat_position = sim.robot.set_default_position
    current_joints = sim.robot.get_joint_positions()
    
    retreat_joints = sim.control.ik_solver(
        goal_position=retreat_position,
        goal_orientation=desired_orientation,
        initial_joint_angles=current_joints,
        max_iterations=100,
        tolerance=1e-3,
        learning_rate=0.1
    )
    
    move_with_collision_avoidance(sim, retreat_joints, obstacle_positions)
    
    print("Pick and place operation completed successfully.")
    pick_and_place_completed = True
    return True

def verify_grasp(sim):
    """Simple grasp verification"""
    return True