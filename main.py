import os
import glob
import yaml
import cv2
import numpy as np
import pybullet as p
import open3d as o3d
from typing import Dict, Any

from pybullet_object_models import ycb_objects
from src.simulation import Simulation
from src.PnP_normal import pick_and_place, pick_and_place_completed
#from src.PnP_obstacle_avoidance import pick_and_place
from src.Kalman_Filter import ObstacleTracker
from src.objects import Obstacle
from src.perception import estimate_ycb_object_pose, visualize_pose_results, estimate_goal_basket_pose, estimate_obstacle_pose

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D




def run_exp(config: Dict[str, Any]):
    # Example Experiment Runner File
    print("Simulation Start:")
    print(config['world_settings'], config['robot_settings'])
    object_root_path = ycb_objects.getDataPath()
    files = glob.glob(os.path.join(object_root_path, "YcbFoamBrick"))
    obj_names = [file.split('/')[-1] for file in files]
    sim = Simulation(config)

    # Camera parameters
    cam_pos = [1.5, 0, 3.0]          # Camera position [x, y, z]
    cam_target = [0, 0, 0.7]       # Camera looks at this point
    cam_up = [0, 0, 1]             # Up direction (Z-axis)
    fov = 70                       # Field of view (degrees)
    aspect = 640 / 480             # Aspect ratio (width/height)
    near = 0.01                     # Near clipping plane
    far = 5.0                      # Far clipping plane
    width, height = 640, 480
    focal_length = width / (2 * np.tan(np.deg2rad(fov) / 2))

    # At the start of simulation:
    real_trajectory1 = []  # To store obstacle 1 positions
    real_trajectory2 = []  # To store obstacle 2 positions
    predicted_trajectory1 = []  # To store obstacle 1 positions
    predicted_trajectory2 = []  # To store obstacle 2 positions

 
    for obj_name in obj_names:
        for tstep in range(10):
            sim.reset(obj_name)
            print((f"Object: {obj_name}, Timestep: {tstep},"
                   f" pose: {sim.get_ground_tuth_position_object}"))
            pos, ori = sim.robot.pos, sim.robot.ori
            print(f"Robot inital pos: {pos} orientation: {ori}")
            l_lim, u_lim = sim.robot.lower_limits, sim.robot.upper_limits
            print(f"Robot Joint Range {l_lim} -> {u_lim}")
            sim.robot.print_joint_infos()
            jpos = sim.robot.get_joint_positions()
            print(f"Robot current Joint Positions: {jpos}")
            jvel = sim.robot.get_joint_velocites()
            print(f"Robot current Joint Velocites: {jvel}")
            ee_pos, ee_ori = sim.robot.get_ee_pose()
            print(f"Robot End Effector Position: {ee_pos}")
            print(f"Robot End Effector Orientation: {ee_ori}")



            for i in range(10000):
                
                sim.step()

                # PERCEPTION PIPELINE 
                
                # Capture RGB, depth, and segmentation images
                rgb_s, depth_s, seg_s = sim.get_static_renders()

                ## Debug what camera sees: -----------------------------
                
                # Compute view and projection matrices
                view_mat = p.computeViewMatrix(cam_pos, cam_target, cam_up)
                proj_mat = p.computeProjectionMatrixFOV(fov, aspect, near, far)

                # Get camera image (returns RGB, depth, segmentation)
                img = p.getCameraImage(width, height, view_mat, proj_mat)

                # Estimate pose for object ID 4
                goal_basket_pos,_ = estimate_goal_basket_pose(img, view_mat, object_id=4, width=width, height=height, 
                                     far=far, near=near, focal_length=focal_length, show_visualizations=True)
                print(f"Goal position: {goal_basket_pos}")
                
                              
                # Estimate pose for object ID 5
                ycb_object_pos,_ = estimate_ycb_object_pose(img, view_mat, object_id=5, width=width, height=height, 
                                     far=far, near=near, focal_length=focal_length, show_visualizations=True)
                print(f"Object position: {ycb_object_pos}")
                             
                
                # Estimate pose for object ID 6 & 7
                obstacle_object_ids = [6, 7]
                for obj_id in obstacle_object_ids:
                    obstacle_pos = estimate_obstacle_pose(img, view_mat, object_id=obj_id, width=width, height=height,
                                           far=far, near=near, focal_length=focal_length, show_visualizations=True)
                    
                    # Store poses based on object ID
                    if obj_id == 6:
                        obstacle1_pos,_ = obstacle_pos
                    elif obj_id == 7:
                        obstacle2_pos,_ = obstacle_pos

                # Verify we got both poses
                if obstacle1_pos is None or obstacle2_pos is None:
                    print("Warning: Failed to detect one or both obstacles")
                else:
                    print(f"Obstacle 1 pose: {obstacle1_pos}")
                    print(f"Obstacle 2 pose: {obstacle2_pos}")

                # Obstacle Tracking Pipeline

                ## in cases where the collision avoidance is failing due to the wrong obstacle poses, 
                ## ...use the ground truth values to accurately locate the obstacles

                # init_pos1,_ = p.getBasePositionAndOrientation(obstacle_1_pos)
                # init_pos2,_ = p.getBasePositionAndOrientation(obstacle_2_pos)
                init_pos1 = obstacle1_pos
                init_pos2 = obstacle2_pos
            

                print("\n--- Real Obstacle Pose ---")
                print(f"Obs.1 - G.Truth Pose: {init_pos1[0]:.3f}, {init_pos1[1]:.3f}, {init_pos1[2]:.3f})")
                print(f"Obs.2 - G.Truth Pose: {init_pos2[0]:.3f}, {init_pos2[1]:.3f}, {init_pos2[2]:.3f})")

                if i == 0:
                    # Get initial positions properly
                    tracker1 = ObstacleTracker(6, init_pos1)
                    tracker2 = ObstacleTracker(7, init_pos2)

                # Update trackers every frame
                filtered_pos1, vel1 = tracker1.update()
                filtered_pos2, vel2 = tracker2.update()

                # Print the "filtered" predicted positions
                # print("\n--- New Frame Obstacle Pose ---")
                print(f"Obs.1 - Predicted Position: {filtered_pos1[0]:.3f}, {filtered_pos1[1]:.3f}, {filtered_pos1[2]:.3f},  Velocity: {vel1}")
                print(f"Obs.2 - Predicted Position: {filtered_pos2[0]:.3f}, {filtered_pos2[1]:.3f}, {filtered_pos2[2]:.3f},  Velocity: {vel2}")

               
                # ## --- Now i try to visualise just the trajectory of obstacle 1
                ### To see the tracking visualisations, uncomment the below section

                # ## Save the real and predicted positions in a list
                # real_trajectory1.append(init_pos1)
                # real_trajectory2.append(init_pos2)
                # predicted_trajectory1.append(filtered_pos1)
                # predicted_trajectory2.append(filtered_pos2)

                # #print(f"Real_trajectory vs Predicted_trajectory_obs.1: {real_trajectory1}, {predicted_trajectory1}")

                # # Convert to numpy arrays with consistent format - to ensure the plot is "fair"
                # real_pos_1 = np.array(real_trajectory1)
                # predicted_pos_1 = np.array(predicted_trajectory1) 

                # # Create time steps (assuming 1 per measurement)
                # time_steps = np.arange(len(real_pos_1))

                ## TO plot the tracking per frame:

                # # Create figure with 3 subplots
                # plt.figure(figsize=(12, 8))

                # # X Position
                # plt.subplot(3, 1, 1)
                # plt.plot(time_steps, real_pos_1[:, 0], 'go-', label='Real X', markersize=4)
                # plt.plot(time_steps, predicted_pos_1[:, 0], 'rx--', label='Predicted X', markersize=4)
                # plt.ylabel('X Position')
                # plt.title('Real vs Kalman Filter Prediction')
                # plt.legend()

                # # Y Position
                # plt.subplot(3, 1, 2)
                # plt.plot(time_steps, real_pos_1[:, 1], 'go-', label='Real Y', markersize=4)
                # plt.plot(time_steps, predicted_pos_1[:, 1], 'rx--', label='Predicted Y', markersize=4)
                # plt.ylabel('Y Position')
                # plt.legend()

                # # Z Position
                # plt.subplot(3, 1, 3)
                # plt.plot(time_steps, real_pos_1[:, 2], 'go-', label='Real Z', markersize=4)
                # plt.plot(time_steps, predicted_pos_1[:, 2], 'rx--', label='Predicted Z', markersize=4)
                # plt.xlabel('Time Step')
                # plt.ylabel('Z Position')
                # plt.legend()

                # plt.tight_layout()
                # plt.show()

         
                ## --------------------------------
           
                # Skip visualization for most frames to speed up simulation
                if i % 1000 != 0:  # Only visualize every 100th frame
                    continue


                ## For other objects, especially the obstacles, the ycb object perception pipeline is not accurate!
                ## Therefore, create special perception pipelines for goal basket and moving obstacles
                ## ****** Therefore, it makes sense for now to use ground truth values for the obstacle tracking


                # # Testing the pick n place pipeline using our derived object pose:
                # if not pick_and_place_completed:
                #     object_position = np.array(ycb_object_pos)                
                #     target_position = np.array (goal_basket_pos)             
                #     pick_and_place(sim, object_position, target_position)
                # else:
                #     print("Pick-and-place already completed")

                # Testing the pick n place pipeline using trial and error values for pick n place:
                if not pick_and_place_completed:
                    object_position = np.array([-0.06, -0.46, 0.25])                
                    target_position = np.array([0.55, 0.58, 0.5])             
                    pick_and_place(sim, object_position, target_position)   ### using the standard PnP pipeine which does avoid obstacles without need for tracking
                    #pick_and_place(sim, object_position, target_position, tracker1, tracker2)   ### using the PnP_obstacle_avoidance pipeline

                else:
                    print("Pick-and-place already completed")

                ## CONTROL
                obs_position_guess = np.zeros((2, 3))
                print(f"-----------------")
                print((f"[{i}] Obstacle Position-Diff: "
                       f"{sim.check_obstacle_position(obs_position_guess)}"))
                goal_guess = np.zeros((7,))
                print((f"[{i}] Goal Obj Pos-Diff: "
                       f"{sim.check_goal_obj_pos(goal_guess)}"))
                print(f"[{i}] Goal Satisfied: {sim.check_goal()}")

                                
    sim.close()

if __name__ == "__main__":
    with open("configs/test_config.yaml", "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    run_exp(config)