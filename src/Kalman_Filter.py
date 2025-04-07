import os
import glob
import yaml
import time

import pybullet as p
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

from typing import Dict, Any

from pybullet_object_models import ycb_objects  # type:ignore

from src.simulation import Simulation

from filterpy.kalman import KalmanFilter

import numpy as np

class ObstacleTracker:
    def __init__(self, obstacle_id, initial_pos, dt=1/240.):  # Assuming 240Hz simulation
        self.kf = KalmanFilter(dim_x=6, dim_z=3)
        
        # State transition matrix (constant velocity model)
        self.kf.F = np.array([[1,0,0,dt,0,0],
                             [0,1,0,0,dt,0],
                             [0,0,1,0,0,dt],
                             [0,0,0,1,0,0],
                             [0,0,0,0,1,0],
                             [0,0,0,0,0,1]])
        
        # Measurement function
        self.kf.H = np.array([[1,0,0,0,0,0],
                            [0,1,0,0,0,0],
                            [0,0,1,0,0,0]])
        
        # Initial state (position and velocity)
        self.kf.x = np.array([*initial_pos, 0, 0, 0])
        
        # Covariance matrices (tune these)
        self.kf.P *= 10  # Initial uncertainty
        self.kf.R = np.eye(3) * 0.1  # Measurement noise
        self.kf.Q = np.eye(6) * 0.01  # Process noise
        
        self.obstacle_id = obstacle_id
        self.last_pos = initial_pos
        self.dt = dt

    def update(self):
        # Get new measurement
        z = np.array(p.getBasePositionAndOrientation(self.obstacle_id)[0])
        
        # Predict
        self.kf.predict()
        
        # Update
        self.kf.update(z)
        
        # Store filtered state
        self.last_pos = self.kf.x[:3]
        self.velocity = self.kf.x[3:]
        
        return self.last_pos, self.velocity




