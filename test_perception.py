# import os
# import glob
# import yaml
# import time
# import cv2

# import pybullet as p
# import open3d as o3d
# import numpy as np
# import matplotlib.pyplot as plt

# from typing import Dict, Any

# from pybullet_object_models import ycb_objects  # type:ignore

# from src.simulation import Simulation

# from src.PnP import pick_and_place

# from src.perception import process_camera_data, combine_poses, visualize_poses

# #from src.control import Control



# def run_exp(config: Dict[str, Any]):
#     # Example Experiment Runner File
#     print("Simulation Start:")
#     print(config['world_settings'], config['robot_settings'])
#     object_root_path = ycb_objects.getDataPath()
#     files = glob.glob(os.path.join(object_root_path, "YcbChipsCan"))
#     obj_names = [file.split('/')[-1] for file in files]
#     sim = Simulation(config)

#     # Camera info
#     projection_matrix = np.array(sim.projection_matrix).reshape(4, 4)
#     width, height = 640, 480  # from image dimensions
#     fx = projection_matrix[0, 0]
#     fy = projection_matrix[1, 1]
#     cx = projection_matrix[0, 2]
#     cy = projection_matrix[1, 2]
    
#     all_object_poses = {}  # Store all object poses throughout simulation
    
#     for obj_name in obj_names:
#         for tstep in range(10):
#             sim.reset(obj_name)
#             print((f"Object: {obj_name}, Timestep: {tstep},"
#                    f" pose: {sim.get_ground_tuth_position_object}"))
#             pos, ori = sim.robot.pos, sim.robot.ori
#             print(f"Robot inital pos: {pos} orientation: {ori}")
#             l_lim, u_lim = sim.robot.lower_limits, sim.robot.upper_limits
#             print(f"Robot Joint Range {l_lim} -> {u_lim}")
#             sim.robot.print_joint_infos()
#             jpos = sim.robot.get_joint_positions()
#             print(f"Robot current Joint Positions: {jpos}")
#             jvel = sim.robot.get_joint_velocites()
#             print(f"Robot current Joint Velocites: {jvel}")
#             ee_pos, ee_ori = sim.robot.get_ee_pose()
#             print(f"Robot End Effector Position: {ee_pos}")
#             print(f"Robot End Effector Orientation: {ee_ori}")

#             for i in range(10000):
                
#                 sim.step()

#                 # Skip visualization for most frames to speed up simulation
#                 if i % 100 != 0:  # Only visualize every 100th frame
#                     continue

#                 # PERCEPTION PIPELINE 
                
#                 # Capture RGB, depth, and segmentation images
#                 rgb_s, depth_s, seg_s = sim.get_static_renders()

#                 # Get view matrix for coordinate transformation
#                 ## stat_viewM shows the transformation from world coordinates to camera coordinates
#                 stat_viewMat_array = np.array(sim.stat_viewMat).reshape(4, 4)
#                 stat_viewMat_inv = np.linalg.inv(stat_viewMat_array)

#                 ## For Debugging and checking if my transformation from camera -> world are
#                 print("View matrix:")
#                 print(stat_viewMat_array)
#                 print("Inverse view matrix:")
#                 print(stat_viewMat_inv)

#                 # Print analysis of the view matrix components
#                 print("\n=== Static Camera View Matrix Analysis ===")
#                 print(f"View matrix inverse shape: {stat_viewMat_inv.shape}")
#                 rotation = stat_viewMat_inv[:3, :3]
#                 translation = stat_viewMat_inv[:3, 3]
#                 print(f"Rotation component:\n{rotation}")
#                 print(f"Translation component:\n{translation}")
#                 print(f"Determinant of rotation: {np.linalg.det(rotation)}")

#                 # Define camera intrinsic parameters (same for both cameras)
#                 intrinsic = o3d.camera.PinholeCameraIntrinsic(
#                     width=width,
#                     height=height,
#                     fx=fx,
#                     fy=fy,
#                     cx=cx,
#                     cy=cy
#                 )

#                 # Print unique object IDs in the segmentation mask
#                 unique_ids = np.unique(seg_s)
#                 print(f"Unique object IDs: {unique_ids}")
                
#                 # Process each unique object ID
#                 for obj_id in sorted(unique_ids):
#                     if obj_id == 0:  # Skip background (ID 0)
#                         continue
#                     print(f"Processing object ID: {obj_id}")

#                 # Dictionary to store object poses for current timestep
#                 timestep_object_poses = {}
                
#                 # Create a visualization list to collect all point clouds and geometries
#                 visualization_geometries = []
                
#                 # Create coordinate frame for world origin
#                 world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
#                 visualization_geometries.append(world_frame)

#                 # Process each unique object ID
#                 debug_dir = "debug_images"
#                 os.makedirs(debug_dir, exist_ok=True)
                
#                 # Print unique object IDs in the segmentation mask
#                 unique_ids = np.unique(seg_s)
#                 print(f"Unique object IDs in segmentation mask: {unique_ids}")

#                 for obj_id in unique_ids:
#                     if obj_id <= 0:  # Skip background (ID 0 or -1)
#                         continue
        
#                     print(f"Processing object with ID: {obj_id}")
    
#                     # Create a binary mask for this object
#                     mask = (seg_s == obj_id)

#                     mask_image = (mask * 255).astype(np.uint8)  # Convert mask to 0-255 range
#                     mask_filename = os.path.join(debug_dir, f"mask_obj_{obj_id}_timestep_{tstep}_frame_{i}.png")
#                     cv2.imwrite(mask_filename, mask_image)

#                     # Save the masked RGB image
#                     rgb_masked = np.copy(rgb_s)
#                     for c in range(3):  # For each color channel
#                         rgb_masked[:, :, c] = np.where(mask, rgb_s[:, :, c], 0)
#                     rgb_masked_filename = os.path.join(debug_dir, f"debug_frame_{obj_name}_{tstep}_{i}_obj{obj_id}_masked_rgb.png")
#                     cv2.imwrite(rgb_masked_filename, cv2.cvtColor(rgb_masked, cv2.COLOR_RGB2BGR))  # OpenCV uses BGR format
                    
#                     # Check if mask is empty
#                     if not np.any(mask):
#                         print(f"No pixels for object ID {obj_id}")
#                         continue
        
#                     # Filter RGB and depth using the mask
#                     rgb_masked = np.copy(rgb_s)
#                     depth_masked = np.copy(depth_s)
    
#                     # Set non-object pixels to 0 in both RGB and depth
#                     for c in range(3):  # For each color channel
#                         rgb_masked[:,:,c] = np.where(mask, rgb_s[:,:,c], 0)
#                     depth_masked = np.where(mask, depth_s, 0)
    
#                     # Create RGBD image for this object
#                     rgbd_object = o3d.geometry.RGBDImage.create_from_color_and_depth(
#                         color=o3d.geometry.Image(rgb_masked),
#                         depth=o3d.geometry.Image(depth_masked),
#                         depth_scale=100,   ## initial value of 1.0
#                         depth_trunc=1000.0,
#                         convert_rgb_to_intensity=False
#                     )
    
#                     # Create point cloud for this object
#                     pcd_object = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_object, intrinsic)
    
#                     # Remove points with zero depth (background)
#                     points = np.asarray(pcd_object.points)
#                     colors = np.asarray(pcd_object.colors)
    
#                     # Filter out points at origin (these are typically from masked-out regions)
#                     mask = ~np.all(points == 0, axis=1)
#                     pcd_object.points = o3d.utility.Vector3dVector(points[mask])
#                     pcd_object.colors = o3d.utility.Vector3dVector(colors[mask])
    
#                     # Check if we have enough points
#                     if len(pcd_object.points) < 10:
#                         print(f"Too few points for object ID {obj_id}: {len(pcd_object.points)}")
#                         continue
        
#                     print(f"Object {obj_id} point cloud has {len(pcd_object.points)} points")
    
#                     # Downsample and remove noise
#                     try:
#                         pcd_object = pcd_object.voxel_down_sample(voxel_size=0.01)
#                         pcd_object, _ = pcd_object.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
#                     except Exception as e:
#                         print(f"Error in processing point cloud for object {obj_id}: {e}")
#                         continue
    
#                     # Compute object pose (centroid and orientation)
#                     if len(pcd_object.points) > 0:
#                         # Compute centroid in camera coordinates
#                         points_array = np.asarray(pcd_object.points)
#                         centroid_camera = np.mean(points_array, axis=0)
                        
#                         # Transform centroid to world coordinates using homogeneous coordinates
#                         centroid_homogeneous = np.append(centroid_camera, 1.0)
#                         centroid_world = np.dot(stat_viewMat_inv, centroid_homogeneous)[:3]
                        
#                         # Get oriented bounding box
#                         try:
#                             obb = pcd_object.get_oriented_bounding_box()
#                             orientation_camera = obb.R  # Rotation matrix in camera coordinates
                            
#                             # Transform orientation to world coordinates
#                             # For rotation matrices, we need to use conjugation: R_world = R_cam_to_world * R_camera * R_world_to_cam
#                             # The simpler approach is to just use the world from camera rotation:
#                             orientation_world = np.dot(stat_viewMat_inv[:3, :3], orientation_camera)
            
#                             # Store both camera and world coordinates
#                             timestep_object_poses[obj_id] = {
#                                 'position_camera': centroid_camera,
#                                 'orientation_camera': orientation_camera,
#                                 'position_world': centroid_world,
#                                 'orientation_world': orientation_world,
#                                 'dimensions': obb.extent
#                             }
                            
#                             print(f"Object {obj_id}:")
#                             print(f"  - Position (Camera): {centroid_camera}")
#                             print(f"  - Position (World): {centroid_world}")
#                             print(f"  - Orientation (World):\n{orientation_world}")
#                             print(f"  - Dimensions: {obb.extent}")
                            
#                             # Transform point cloud to world coordinates for visualization
#                             pcd_world = o3d.geometry.PointCloud()
#                             transformed_points = []
                            
#                             for point in np.asarray(pcd_object.points):
#                                 # Convert to homogeneous coordinates
#                                 point_homogeneous = np.append(point, 1.0)
#                                 # Transform to world coordinates
#                                 point_world = np.dot(stat_viewMat_inv, point_homogeneous)[:3]
#                                 transformed_points.append(point_world)
                            
#                             pcd_world.points = o3d.utility.Vector3dVector(np.array(transformed_points))
                            
#                             # Assign a unique color to each object for better visualization
#                             color = [obj_id * 0.1 % 1.0, (obj_id * 0.3) % 1.0, (obj_id * 0.7) % 1.0]
#                             pcd_world.paint_uniform_color(color)
                            
#                             # Create a coordinate frame at the object's position
#                             obj_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
#                                 size=0.2, 
#                                 origin=centroid_world
#                             )
                            
#                             # Add to visualization list
#                             visualization_geometries.append(pcd_world)
#                             visualization_geometries.append(obj_frame)
                            
#                             # Create a transformed OBB in world coordinates
#                             obb_world = o3d.geometry.OrientedBoundingBox(
#                                 center=centroid_world,
#                                 R=orientation_world,
#                                 extent=obb.extent
#                             )
#                             visualization_geometries.append(obb_world)
            
#                         except Exception as e:
#                             print(f"Error computing OBB for object {obj_id}: {e}")
#                             # Fallback to just position without orientation
#                             timestep_object_poses[obj_id] = {
#                                 'position_camera': centroid_camera,
#                                 'position_world': centroid_world,
#                                 'orientation_camera': np.eye(3),  # Identity rotation
#                                 'orientation_world': np.dot(stat_viewMat_inv[:3, :3], np.eye(3)),
#                                 'dimensions': None
#                             }
    
#                 # Print comparison of estimated positions with ground truth
#                 for obj_id, pose in timestep_object_poses.items():
#                     print(f"Object {obj_id} at position (world): {pose['position_world']}")
#                     try:
#                         obj_pos, obj_orn = p.getBasePositionAndOrientation(obj_id)
#                         print(f"Ground truth: {obj_pos}")
#                         # Calculate and print the error
#                         error = np.linalg.norm(np.array(pose['position_world']) - np.array(obj_pos))
#                         print(f"Position error: {error:.4f} units")
#                     except p.error:
#                         print("Could not get ground truth position for this object")
                
#                 # Store poses for this timestep in the main dictionary
#                 all_object_poses[f"{obj_name}_{tstep}_{i}"] = timestep_object_poses

#                 # Optionally uncomment to visualize the point clouds
                
#                 # Visualize all objects in world coordinates
#                 if visualization_geometries:
#                     # Create a better visualization window with useful options
#                     vis = o3d.visualization.VisualizerWithKeyCallback()
#                     vis.create_window(window_name=f"Objects in World Coordinates - Frame {i}")
                    
#                     # Add all geometries
#                     for geom in visualization_geometries:
#                         vis.add_geometry(geom)
                    
#                     # Set view control options for better viewing
#                     view_control = vis.get_view_control()
#                     view_control.set_zoom(0.8)
#                     view_control.set_front([0, -1, 0])  # Look along -Y axis
#                     view_control.set_up([0, 0, 1])      # Z is up
                    
#                     # Add rendering options
#                     render_option = vis.get_render_option()
#                     render_option.point_size = 3
#                     render_option.background_color = np.array([0.1, 0.1, 0.1])  # Dark background
#                     render_option.show_coordinate_frame = True
                    
#                     # Run visualization
#                     vis.run()
#                     vis.destroy_window()
                

#                 ## PLANNING & CONTROL - placeholder for future implementation
#                 obs_position_guess = np.zeros((2, 3))
#                 print((f"[{i}] Obstacle Position-Diff: "
#                        f"{sim.check_obstacle_position(obs_position_guess)}"))
#                 goal_guess = np.zeros((7,))
#                 print((f"[{i}] Goal Obj Pos-Diff: "
#                        f"{sim.check_goal_obj_pos(goal_guess)}"))
#                 print(f"[{i}] Goal Satisfied: {sim.check_goal()}")
    
#     sim.close()
#     return all_object_poses


# if __name__ == "__main__":
#     with open("configs/test_config.yaml", "r") as stream:
#         try:
#             config = yaml.safe_load(stream)
#             print(config)
#         except yaml.YAMLError as exc:
#             print(exc)
#     run_exp(config)


import os
import glob
import yaml
import time
import cv2

import pybullet as p
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

from typing import Dict, Any

from pybullet_object_models import ycb_objects  # type:ignore

from src.simulation import Simulation

from src.PnP import pick_and_place

from src.perception import estimate_obb_from_camera

#from src.control import Control

def run_exp(config: Dict[str, Any]):
    print("Simulation Start:")
    print(config['world_settings'], config['robot_settings'])
    object_root_path = ycb_objects.getDataPath()
    files = glob.glob(os.path.join(object_root_path, "YcbChipsCan"))
    obj_names = [file.split('/')[-1] for file in files]
    sim = Simulation(config)

    # Camera info
    projection_matrix = np.array(sim.projection_matrix).reshape(4, 4)
    width, height = 640, 480
    fx = projection_matrix[0, 0]
    fy = projection_matrix[1, 1]
    cx = projection_matrix[0, 2]
    cy = projection_matrix[1, 2]
    
    all_object_poses = {}
    
    for obj_name in obj_names:
        for tstep in range(10):
            sim.reset(obj_name)
            print(f"Object: {obj_name}, Timestep: {tstep}, pose: {sim.get_ground_tuth_position_object}")
            
            # Get robot info (unchanged)
            pos, ori = sim.robot.pos, sim.robot.ori
            print(f"Robot initial pos: {pos} orientation: {ori}")
            l_lim, u_lim = sim.robot.lower_limits, sim.robot.upper_limits
            print(f"Robot Joint Range {l_lim} -> {u_lim}")
            sim.robot.print_joint_infos()
            jpos = sim.robot.get_joint_positions()
            print(f"Robot current Joint Positions: {jpos}")
            jvel = sim.robot.get_joint_velocites()
            print(f"Robot current Joint Velocities: {jvel}")
            ee_pos, ee_ori = sim.robot.get_ee_pose()
            print(f"Robot End Effector Position: {ee_pos}")
            print(f"Robot End Effector Orientation: {ee_ori}")

            for i in range(10000):
                sim.step()

                if i % 100 != 0:
                    continue

                # PERCEPTION PIPELINE 
                rgb_s, depth_s, seg_s = sim.get_static_renders()
                stat_viewMat_array = np.array(sim.stat_viewMat).reshape(4, 4)
                stat_viewMat_inv = np.linalg.inv(stat_viewMat_array)

                intrinsic = o3d.camera.PinholeCameraIntrinsic(
                    width=width,
                    height=height,
                    fx=fx,
                    fy=fy,
                    cx=cx,
                    cy=cy
                )

                unique_ids = np.unique(seg_s)
                print(f"Unique object IDs: {unique_ids}")
                
                timestep_object_poses = {}
                visualization_geometries = []
                world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
                visualization_geometries.append(world_frame)

                debug_dir = "debug_images"
                os.makedirs(debug_dir, exist_ok=True)
                
                for obj_id in unique_ids:
                    if obj_id <= 0:
                        continue
        
                    print(f"Processing object with ID: {obj_id}")
    
                    # Create mask and debug images (unchanged)
                    mask = (seg_s == obj_id)
                    mask_image = (mask * 255).astype(np.uint8)
                    mask_filename = os.path.join(debug_dir, f"mask_obj_{obj_id}_timestep_{tstep}_frame_{i}.png")
                    cv2.imwrite(mask_filename, mask_image)

                    rgb_masked = np.copy(rgb_s)
                    for c in range(3):
                        rgb_masked[:, :, c] = np.where(mask, rgb_s[:, :, c], 0)
                    rgb_masked_filename = os.path.join(debug_dir, f"debug_frame_{obj_name}_{tstep}_{i}_obj{obj_id}_masked_rgb.png")
                    cv2.imwrite(rgb_masked_filename, cv2.cvtColor(rgb_masked, cv2.COLOR_RGB2BGR))
                    
                    if not np.any(mask):
                        print(f"No pixels for object ID {obj_id}")
                        continue
        
                    # Get PyBullet OBB data
                    try:
                        obb_data = estimate_obb_from_camera(
                            rgb=rgb_s, 
                            depth=depth_s, 
                            seg_mask=seg_s, 
                            obj_id=obj_id,
                            intrinsic=intrinsic,
                            view_matrix_inv=stat_viewMat_inv
                        )
                        
                        if obb_data:
                            # Visualize
                            obb = o3d.geometry.OrientedBoundingBox()
                            obb.center = obb_data['position']
                            obb.R = obb_data['orientation']
                            obb.extent = obb_data['extents']
                            obb.color = [0, 1, 0]  # Green for perception-estimated
                            visualization_geometries.append(obb)
                        
                    
                        
                        # Compare with ground truth
                        print(f"Object {obj_id} PyBullet OBB position:", obb_data['position'])
                        actual_pos, actual_orn = p.getBasePositionAndOrientation(obj_id)
                        print("Actual PyBullet position:", actual_pos)
                        
                        # Create coordinate frame at object position
                        obj_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                            size=0.2, 
                            origin=obb_data['position']
                        )
                        visualization_geometries.append(obj_frame)
                        
                    except Exception as e:
                        print(f"Error processing object {obj_id}: {e}")
                        continue

                # Store poses for this timestep
                all_object_poses[f"{obj_name}_{tstep}_{i}"] = timestep_object_poses


                # Visualization
                if visualization_geometries:
                    vis = o3d.visualization.VisualizerWithKeyCallback()
                    vis.create_window(window_name=f"PyBullet OBBs - Frame {i}")
                    
                    for geom in visualization_geometries:
                        vis.add_geometry(geom)
                    
                    view_control = vis.get_view_control()
                    view_control.set_zoom(0.8)
                    view_control.set_front([0, -1, 0])
                    view_control.set_up([0, 0, 1])
                    
                    render_option = vis.get_render_option()
                    render_option.point_size = 3
                    render_option.background_color = np.array([0.1, 0.1, 0.1])
                    render_option.show_coordinate_frame = True
                    
                    vis.run()
                    vis.destroy_window()

                # PLANNING & CONTROL (unchanged)
                obs_position_guess = np.zeros((2, 3))
                print(f"[{i}] Obstacle Position-Diff: {sim.check_obstacle_position(obs_position_guess)}")
                goal_guess = np.zeros((7,))
                print(f"[{i}] Goal Obj Pos-Diff: {sim.check_goal_obj_pos(goal_guess)}")
                print(f"[{i}] Goal Satisfied: {sim.check_goal()}")
    
    sim.close()
    return all_object_poses

if __name__ == "__main__":
    with open("configs/test_config.yaml", "r") as stream:
        try:
            config = yaml.safe_load(stream)
            print(config)
        except yaml.YAMLError as exc:
            print(exc)
    run_exp(config)