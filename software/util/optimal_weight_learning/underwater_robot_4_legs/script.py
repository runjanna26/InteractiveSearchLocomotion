import time
import numpy as np
import mujoco
import mujoco.viewer
import rclpy
import math

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Assuming these are available in your workspace
from ros2_node import StickInsectNode
from muscle_model import MuscleModel
from hydrodynamic import Hydrodynamics
from generate_terrain import TerrainGenerator

FOOT_NAMES = ['FOOT_FR', 'FOOT_BR', 'FOOT_FL', 'FOOT_BL']
JOINT_GROUPS = ['FR', 'BR', 'FL', 'BL']
NUM_JOINTS_PER_GROUP = 4

class StickInsectEnv:
    def __init__(self, start_time=1.0, enable_ros=False, render=False):
        """
        Object-oriented wrapper for the MuJoCo simulation to allow for rapid 
        episodic resets during PIBB optimization.
        """
        self.start_time = start_time
        self.enable_ros = enable_ros
        self.has_viewer = render
        self.robot_fixed = False

        self.frame_skip = 1
        
        # 1. Generate Terrain and Load Model
        self._build_environment()
        
        # 2. Initialize Custom Physics and Controllers
        self.hydro = Hydrodynamics(self.model, water_level=0.8)
        self._init_controllers()
        self.foot_geom_ids = self._init_foot_sensors()
        
        # 3. Optional ROS Initialization (Best kept OFF during fast training)
        if self.enable_ros:
            if not rclpy.ok():
                rclpy.init()
            self.ros_node = StickInsectNode()
            
        # 4. Viewer Initialization
        if self.has_viewer:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data, show_left_ui=False, show_right_ui=False)
            self._setup_camera()

    def _build_environment(self):
        x_start = -20
        tg = TerrainGenerator()
        # terrain_xml             = tg.generate_rough_terrain(    name = 'rough_terrain_1',   n_rows=100, n_cols=100, start_pos=(x_start+15, 0, 1.5))
        # terrain_xml                 = tg.generate_flat_terrain(     name = 'flat_terrain_1',    n_rows=500, n_cols=500, start_pos=(x_start, 0, 1.5))
        # terrain_xml                 = tg.generate_soft_terrain(     name = 'soft_terrain_1',    n_rows=500, n_cols=500, start_pos=(x_start, 0, 1.5))
        # terrain_xml                 = tg.generate_muddy_terrain(     name = 'muddy_terrain_1',    n_rows=500, n_cols=500, start_pos=(x_start, 0, 1.5))
        # terrain_xml                 = tg.generate_slippery_terrain (     name = 'slippery_terrain_1',    n_rows=500, n_cols=500, start_pos=(x_start, 0, 1.5))
        
        terrain_xml                 = tg.generate_flat_terrain(     name = 'water_terrain_1',    n_rows=500, n_cols=500, start_pos=(x_start, 0, -1.5))

        # rough_water_terrain_xml       = tg.generate_rough_terrain(    name = 'rough_water_terrain_1',   n_rows=36, n_cols=36, start_pos=(-16.25 + x_start, 0, 0.25), h_dev=0.05)
        # flat_water_terrain_xml        = tg.generate_flat_terrain(     name = 'flat_water_terrain_1',    n_rows=36, n_cols=36, start_pos=(-16.25 + x_start - 9, 0, 0.25))
        # sponge_water_terrain_xml      = tg.generate_sponge_terrain(   name = 'sponge_water_terrain_1',  n_rows=36, n_cols=36, start_pos=(-16.25 + x_start - 18, 0, 0.25))
        # sandy_water_terrain_xml       = tg.generate_sandy_terrain(    name = 'sandy_water_terrain_1',   n_rows=36, n_cols=36, start_pos=(-16.25 + x_start - 27, 0, 0.25))
        # muddy_water_terrain_xml       = tg.generate_muddy_terrain(    name = 'muddy_water_terrain_1',   n_rows=36, n_cols=36, start_pos=(-16.25 + x_start - 36, 0, 0.25))


        with open("main_scene.xml", "r") as f:
            base_xml = f.read()
        complete_xml = base_xml.replace("INCLUDE_TERRAIN", terrain_xml, 1)
        self.model = mujoco.MjModel.from_xml_string(complete_xml)

        # self.model = mujoco.MjModel.from_binary_path("fast_rough_scene.mjb")


        self.data = mujoco.MjData(self.model)

        # mujoco.mj_saveModel(self.model, "fast_rough_scene.mjb", None)
        # print("Binary model saved!")

    def _get_actuator_name(self, idx):
        addr = self.model.name_actuatoradr[idx]
        return self.model.names[addr:].decode().split('\x00', 1)[0]

    def _init_controllers(self):
        self.controllers = {}
        self.actuator_ids = {}
        self.qpos_ids = {}
        self.qvel_ids = {}

        for i in range(self.model.nu):
            name = self._get_actuator_name(i)
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            
            if joint_id == -1:
                continue

            self.actuator_ids[name] = i
            self.qpos_ids[name] = self.model.jnt_qposadr[joint_id]
            self.qvel_ids[name] = self.model.jnt_dofadr[joint_id]

            self.controllers[name] = MuscleModel(_a=0.5, _b=10.0, _beta=0.0, _init_pos=0.0)

    def _init_foot_sensors(self):
        foot_geom_ids = {}
        for name in FOOT_NAMES:
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
            if gid == -1:
                gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid == -1:
                gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name.upper())
            
            if gid != -1:
                foot_geom_ids[name] = gid
        return foot_geom_ids

    def _setup_camera(self):
        robot_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot")
        
        # 2. Change the default Free Camera into a Tracking Camera
        self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self.viewer.cam.trackbodyid = robot_body_id
        
        # 3. Set the default starting angle (Optional but highly recommended)
        self.viewer.cam.distance = 3.0    # How far away to start (meters)
        self.viewer.cam.azimuth = 135      # Rotate left/right (degrees)
        self.viewer.cam.elevation = -20   # Tilt up/down (degrees)

    def get_grf(self):
        grf = {name: 0.0 for name in FOOT_NAMES}
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            for name, gid in self.foot_geom_ids.items():
                if contact.geom1 == gid or contact.geom2 == gid:
                    force = np.zeros(6)
                    mujoco.mj_contactForce(self.model, self.data, i, force)
                    grf[name] += force[0]
        return [grf[name] for name in FOOT_NAMES]
    
    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        
        # --- NEW: LET THE ROBOT SETTLE IN THE WATER BEFORE THE EPISODE STARTS ---
        # --- LET THE ROBOT SETTLE IN THE WATER ---
        for _ in range(int(1.0 / self.model.opt.timestep)):
            self.data.xfrc_applied[:] = 0.0  # Clear forces here too!
            self.hydro.apply(self.data)
            mujoco.mj_step(self.model, self.data)

        # --- NEW: WARM-UP SETTINGS ---
        # 1. Define how long to hold the robot while the CPG starts (e.g., 1.5 seconds)
        self.warmup_steps = int(1.5 / self.model.opt.timestep) 
        
        # 2. Save the exact 6-DOF state of the torso (7 position values, 6 velocity values)
        self.fixed_qpos = np.copy(self.data.qpos[:7])


        self.start_pos = np.copy(self.data.qpos[:3])

        self.step_count = 0
        
        # --- History Tracking for Fitness ---
        self.history_z = []
        self.history_roll = []
        self.history_pitch = []
        self.history_yaw = []
        
        self.accumulated_collision = 0.0
        self.contact_steps = 0
        self.slipping_steps = 0

        # New trackers for smoothness
        self.accumulated_torque_ripple = 0.0
        self.accumulated_joint_accel = 0.0
        self.step_count = 0

        # Array to hold the torques from the previous timestep
        self.prev_ctrl = np.zeros(self.model.nu)

    def step(self, targets):
        """
        Takes target joint positions (e.g. from CPG-RBF network), 
        applies MuscleModel math, handles hydrodynamics, and steps physics.
        """
        # ==========================================
        # 1. APPLY CONTROL & PHYSICS
        # ==========================================
        for name, ctrl in self.controllers.items():
            q = self.data.qpos[self.qpos_ids[name]]
            dq = self.data.qvel[self.qvel_ids[name]]
            
            # Prevent exploding targets
            target = np.clip(targets.get(name, 0.0), -3.14, 3.14) 
            ctrl.calculate(target, q, dq, self.model.opt.timestep)

    
            # Prevent exploding torques
            torque = np.clip(ctrl.get_torque(), -15.0, 15.0) 
            self.data.ctrl[self.actuator_ids[name]] = torque

        # ==========================================
        # 🚨 THE GOLD STANDARD PHYSICS STEP
        # ==========================================
        # Step 1 computes kinematics and WIPES any ghost viewer forces.
        mujoco.mj_step1(self.model, self.data)
        
        # ==========================================
        # 🚨 MANUAL FORCE CLEARING (Guarantees zero ghost forces)
        # ==========================================
        self.data.xfrc_applied[:] = 0.0
        self.data.qfrc_applied[:] = 0.0

        # Apply our water forces cleanly
        self.hydro.apply(self.data)
        
        # Step 2 integrates the actual movement
        mujoco.mj_step2(self.model, self.data)
        
        self.step_count += 1


        # Apply Custom Hydrodynamics
        self.hydro.apply(self.data)
        # for _ in range(self.frame_skip):
        #     # GOOD: Apply hydrodynamics right before EVERY physics step
        #     self.hydro.apply(self.data)
        #     mujoco.mj_step(self.model, self.data)

        # Step MuJoCo
        mujoco.mj_step(self.model, self.data)
        
        # Track total steps taken in this rollout (Crucial for fitness averages)
        self.step_count += 1


        if self.step_count < self.warmup_steps:
            # Overwrite the torso's position and rotation back to the starting point
            self.data.qpos[:7] = self.fixed_qpos
            # Force the torso's linear and angular velocity to strictly zero
            self.data.qvel[:6] = 0.0
            
            # 🚨 THE MAGIC FIX: Manually update the spatial kinematics!
            # This does exactly what the viewer was doing in the background.
            mujoco.mj_forward(self.model, self.data)
            
            # Constantly reset the start position so distance tracking ONLY begins upon release
            self.start_pos = np.copy(self.data.qpos[:3])
            
            # Render visualizer if enabled, but SKIP all collision/instability math below!
            if self.has_viewer: self.render()
            if self.enable_ros: self._handle_ros_feedback()
            return

        # ==========================================
        # 2. TRACK INSTABILITY METRICS
        # ==========================================
        self.history_z.append(self.data.qpos[2])
        roll, pitch, yaw = self.quat_to_euler(self.data.qpos[3:7])
        
        self.history_roll.append(abs(roll))
        self.history_pitch.append(abs(pitch))
        self.history_yaw.append(abs(yaw))
        
        # ==========================================
        # 3. TRACK COLLISION (Exponential Inverse Distance)
        # ==========================================
        # foot_positions = [self.data.geom(name).xpos for name in FOOT_NAMES]
        # num_feet = len(foot_positions)
        
        # for i in range(num_feet):
        #     for j in range(i + 1, num_feet):
        #         dist = np.linalg.norm(foot_positions[i] - foot_positions[j])
        #         if dist < 0.15: # maxdist threshold
        #             self.accumulated_collision += np.exp(-dist * 10) 
        
        step_collision = 0.0
        
        # Loop through every physical contact detected by MuJoCo in this timestep
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            
            # Get the ID of the main body that each colliding geometry belongs to
            body1_id = self.model.geom_bodyid[contact.geom1]
            body2_id = self.model.geom_bodyid[contact.geom2]
            
            # Convert those IDs into string names (e.g., 'FR_calf', 'FR_thigh')
            body1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body1_id)
            body2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body2_id)
            
            # Safely check if they have names
            if body1_name and body2_name:
                # List your leg prefixes
                for prefix in ['FR_', 'FL_', 'BR_', 'BL_']:
                    # If BOTH colliding bodies start with the exact same prefix...
                    if body1_name.startswith(prefix) and body2_name.startswith(prefix):
                        # ...it is an intra-leg collision! 
                        # Add a flat penalty for this frame.
                        step_collision += 1.0
                        break # Stop checking prefixes
                        
        self.accumulated_collision += step_collision
        # print(self.accumulated_collision)
        # ==========================================
        # 4. TRACK SLIPPAGE (Optimized to prevent double-counting)
        # ==========================================
        # First, gather all unique feet touching the ground this step
        contacting_feet_gids = set()
        foot_gids_set = set(self.foot_geom_ids.values())
        
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if contact.geom1 in foot_gids_set:
                contacting_feet_gids.add(contact.geom1)
            if contact.geom2 in foot_gids_set:
                contacting_feet_gids.add(contact.geom2)

        # Now, only calculate velocity ONCE per contacting foot
        foot_vel = np.zeros(6)
        for gid in contacting_feet_gids:
            self.contact_steps += 1
            mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_GEOM, gid, foot_vel, 0)
            
            # If XY velocity is high during contact, it is slipping
            horizontal_speed = np.linalg.norm(foot_vel[:2])
            if horizontal_speed > 0.05:
                self.slipping_steps += 1

        
        # Difference between current motor commands and previous commands
        torque_diff = self.data.ctrl - self.prev_ctrl
        self.accumulated_torque_ripple += np.sum(np.square(torque_diff))
        
        # Save current control for the next step
        self.prev_ctrl = np.copy(self.data.ctrl)

        # 2. Joint Acceleration (Shakiness)
        step_accel = 0.0
        for name in self.controllers.keys():
            # self.data.qacc holds the accelerations computed by MuJoCo
            step_accel += abs(self.data.qacc[self.qvel_ids[name]])
        self.accumulated_joint_accel += step_accel


        # ==========================================
        # 6. RENDERING & ROS
        # ==========================================
        if self.has_viewer:
            self.render()
        if self.enable_ros:
            self._handle_ros_feedback()

    def render(self):
        """Handles GUI updates and drawing custom hydrodynamics arrows."""
        if not self.viewer.is_running():
            return
            
        self.viewer.sync()
        with self.viewer.lock():
            self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT]     = True
            self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE]        = True
            self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE]     = True
            self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_COM]              = True
            self.viewer.opt.frame = mujoco.mjtFrame.mjFRAME_WORLD
            
            self.viewer.user_scn.ngeom = 0 
            
            # Draw Hydrodynamic arrows
            for i in self.hydro.body_ids:
                pos = self.data.xpos[i]
                
                # Linear Force
                force = np.array([self.data.xfrc_applied[i][0], self.data.xfrc_applied[i][1], self.data.xfrc_applied[i][2]])
                if np.linalg.norm(force) > 1e-3 and self.viewer.user_scn.ngeom < self.viewer.user_scn.maxgeom:
                    pt2_force = pos + (force * 100) + np.array([0, 0, 10])
                    mujoco.mjv_connector(self.viewer.user_scn.geoms[self.viewer.user_scn.ngeom], mujoco.mjtGeom.mjGEOM_ARROW, 0.005, pos, pt2_force)
                    self.viewer.user_scn.geoms[self.viewer.user_scn.ngeom].rgba = np.array([0.0, 0.5, 1.0, 1.0])
                    self.viewer.user_scn.ngeom += 1

                # Torque
                torque = np.array([self.data.xfrc_applied[i][3], self.data.xfrc_applied[i][4], self.data.xfrc_applied[i][5]])
                if np.linalg.norm(torque) > 1e-3 and self.viewer.user_scn.ngeom < self.viewer.user_scn.maxgeom:
                    pt2_torque = pos + (torque * 100) + np.array([0, 0, 10])
                    mujoco.mjv_connector(self.viewer.user_scn.geoms[self.viewer.user_scn.ngeom], mujoco.mjtGeom.mjGEOM_ARROW, 0.008, pos, pt2_torque)
                    self.viewer.user_scn.geoms[self.viewer.user_scn.ngeom].rgba = np.array([1.0, 0.0, 0.0, 1.0])
                    self.viewer.user_scn.ngeom += 1

    def _handle_ros_feedback(self):
        """Keep your ROS logic cleanly separated for when you need telemetry."""
        rclpy.spin_once(self.ros_node, timeout_sec=0.0)
        # Add your telemetry logic here (joint_angle_fb, GRF arrays, etc.) 
        # identical to your original script...

    def calculate_fitness(self):
        # ==========================================
        # 1. EARLY SAFEGUARD (Must execute first)
        # ==========================================
        # roll, pitch, yaw = self.quat_to_euler(self.data.qpos[3:7])
        
        # # Catch explosions or flips instantly
        # if (np.isnan(self.data.qpos).any() or 
        #     np.isnan(self.data.qvel).any() or 
        #     self.data.qpos[2] > 3.0 or 
        #     abs(roll) > 1.5 or 
        #     abs(pitch) > 1.5):
            
        #     return {
        #         "distance_walked": 0.0,
        #         "instability": 0.0,
        #         "collision": 0.0,
        #         "slippage": 0.0,
        #         "total_reward": -1000.0
        #     }

        # ==========================================
        # 2. CALCULATE METRICS
        # ==========================================
        # Forward Distance & Y-Axis Drift
        distance_walked = (self.data.qpos[0] - self.start_pos[0])
        drift_y = abs(self.data.qpos[1] - self.start_pos[1])
        
        # Instability 
        std_z = np.std(self.history_z) if len(self.history_z) > 0 else 0
        
        mean_roll = np.mean(self.history_roll) if len(self.history_roll) > 0 else 0
        mean_pitch = np.mean(self.history_pitch) if len(self.history_pitch) > 0 else 0
        mean_yaw = np.mean(self.history_yaw) if len(self.history_yaw) > 0 else 0
        
        # Now std_z will only be ~50 instead of ~5000 if it sinks
        instability = std_z + mean_roll + mean_pitch + mean_yaw
        
        # Averages for penalties
        if self.step_count > 0:
            avg_torque_ripple = self.accumulated_torque_ripple / self.step_count
            avg_joint_accel   = self.accumulated_joint_accel / self.step_count
            
            # --- NEW: Calculate Average Collision ---
            avg_collision     = self.accumulated_collision / self.step_count
        else:
            avg_torque_ripple = 0.0
            avg_joint_accel   = 0.0
            avg_collision     = 0.0

        # ==========================================
        # 3. TOTAL REWARD COMPUTATION
        # ==========================================
        w1 = 5.0      # Distance weight
        w2 = 3.0      # Instability weight
        w4 = 5.0      # Y-Drift penalty weight
        
        w_collision = 1000.0  # NEW: Collision penalty weight
        w_ripple    = 0.0 
        w_accel     = 0.0  
        

        total_reward = (w1 * distance_walked) - (w2 * instability) - (w4 * drift_y) - (w_collision * avg_collision) - (w_ripple * avg_torque_ripple) - (w_accel * avg_joint_accel)
        
        return {
            "distance_walked":  (w1 * distance_walked),
            "instability":      -(w2 * instability),
            "collision":        -(w_collision * avg_collision),        
            "slippage":         -(w4 * drift_y) - (w_ripple * avg_torque_ripple) - (w_accel * avg_joint_accel),  
            "total_reward":     total_reward
        }
    

    def quat_to_euler(self, quat):
        """Convert MuJoCo quaternion (w, x, y, z) to Euler angles (roll, pitch, yaw)."""
        w, x, y, z = quat
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
        
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
        
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
        
        return roll_x, pitch_y, yaw_z