# 🧠 Intelligent Robotic Manipulation

## 🚀 Project Overview

The objective of this project is to develop an intelligent robotic system capable of **grasping a YCB object** and **placing it in a designated goal basket**, all while **avoiding dynamic and static obstacles**. The challenge integrates elements from perception, control, motion planning, and manipulation in a simulated environment.

> 🔒 Note: Solutions must not use any privileged information from the simulation environment.

---

## 🧩 Task Breakdown (Guidance Only)

The following are the implemented subtasks;:

1. **Perception**  
   Detect and identify graspable YCB objects in the scene.

2. **Controller**  
   Move the robot arm in response to planned trajectories and control commands.

3. **Grasp Execution**  
   Sample grasp poses and execute reliable grasps on the identified object.

4. **Obstacle Localization & Tracking**  
   Detect, localize, and track obstacles in the environment to ensure safe planning.

5. **Trajectory Planning**  
   Plan and execute a collision-free trajectory to transport the object to the goal basket.

---

## 🛠 Technologies Used

- Robot simulation platform (e.g., PyBullet / MuJoCo / Isaac Gym)
- Computer vision (e.g., OpenCV, PyTorch, YOLO, or custom networks)
- Motion planning (e.g., RRT*, A*, CHOMP, or MoveIt)
- Control (e.g., PID, impedance control, inverse kinematics)

---

## 📁 Project Structure

