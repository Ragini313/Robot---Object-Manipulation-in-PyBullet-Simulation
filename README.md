# 🧠 Intelligent Robotic Manipulation
---

## 🚀 Project Overview


Welcome to the **Intelligent Robotic Manipulation** project! The objective of this project is to design and implement a robotic system that can **grasp a YCB object** and **place it in a goal basket** while effectively **avoiding obstacles**. This repository includes guidance, code, and configurations to accomplish this.

The objective of this project is to develop an intelligent robotic system capable of **grasping a YCB object** and **placing it in a designated goal basket**, all while **avoiding dynamic and static obstacles**. The challenge integrates elements from perception, control, motion planning, and manipulation in a simulated environment.


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



## 🔧 Installation \& Setup

```bash
git clone https://github.com/rafey1104/adaptive_pick_and_place_franka_robot.git
cd adaptive_pick_and_place_franka_robot

conda create -n irobman python=3.8
conda activate irobman
conda install pybullet
pip install matplotlib pyyaml

git clone https://github.com/eleramp/pybullet-object-models.git  # inside the irobman_project folder
pip install -e pybullet-object-models/
```

Make sure `pybullet.isNumpyEnabled()` returns True (optional but recommended).

---

## Running the project
```bash
python3 main.py
```


## 📌 Project Overview

The project is divided into 5 major tasks:

1. **Perception** - Detect the graspable object using a dual-camera setup.
2. **Control** - Move the robot arm using a custom IK-solver.
3. **Grasping** - Design and execute a grasp strategy.
4. **Localization \& Tracking** - Track and avoid dynamic/static obstacles.
5. **Planning** - Plan a safe trajectory to place the object into a goal receptacle.

---

## 🗂️ Codebase Structure

```
├── configs
│   └── test_config.yaml         # Experiment configurations
├── main.py                      # Example runner script
├── README.md
└── src
    ├── objects.py               # Object and obstacle definitions
    ├── robot.py                 # Robot class
    ├── simulation.py            # Simulation environment
    └── control.py               # Conrol class 
    ├──perception.py             # perception class
    ├── utility.py               # utility functions
    ├── pickandplace.py          # pick and place function


---

## 🧠 Tasks Breakdown

### ✅ Task 1: Perception (6D Pose Estimation)

Use a static camera for coarse object detection and an end-effector-mounted camera for fine pose estimation.

![Perception](images/perception_view.jpg)
🎥 [Watch Perception in Action](videos/perception_demo.mp4)

Used:

- Global \& ICP Registration
- MegaPose(explored)
- Synthetic masks from PyBullet

---

### ✅ Task 2: Controller (Inverse Kinematics)

Implemented an IK-solver (e.g., pseudo-inverse) for the Franka Panda robot to reach target positions.

![Controller](images/controller_view.jpg)
🎥 [Watch Controller in Action](videos/controller_demo.mp4)

---

### ✅ Task 3: Grasping

Used our IK-based controller and camera to execute object grasps. Begin with fixed objects (e.g., foam brick) and extend to random YCB items.

![Grasping](images/grasping_view.jpg)
🎥 [Watch Grasping in Action](videos/grasping_demo.mp4)

---

### ✅ Task 4: Localization \& Tracking

Tracks red sphere obstacles using the static camera setup and visualize obstacle motion.  Used Kalman Filter.

![Tracking](images/tracking_view.jpg)
🎥 [Watch Tracking in Action](videos/tracking_demo.mp4)

---

### ✅ Task 5: Planning

Planned a trajectory to the goal while avoiding obstacles. 

![Planning](images/planning_view.jpg)
🎥 [Watch Planning in Action](videos/planning_demo.mp4)

Example (no obstacle avoidance):

## 📎 Submission Format

1. GitHub repository with a **clear README** and **runnable scripts**
2. Final **report (PDF)** in **TU-Darmstadt format**

---

## 📝 Tips

- Use `W` to toggle wireframe mode
- Press `J` to display axes in GUI
- Disable cam output to speed up simulation
- Use debug GUI and log intermediate outputs
