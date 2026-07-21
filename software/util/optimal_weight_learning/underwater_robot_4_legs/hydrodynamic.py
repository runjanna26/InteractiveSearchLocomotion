import numpy as np
import mujoco

class Hydrodynamics:
    def __init__(self, model, water_level=0.0):
        self.water_level = water_level
        self.density = 1000.0  
        self.model = model
        self.gravity = abs(self.model.opt.gravity[2])
        
        self.Cd_side = 10.0 
        self.Cd_end = 0.1  
        self.linear_drag_coeff = 40.0 
        self.angular_drag_coeff = 10.0 

        target_density = 900 

        self.body_props = {}
        self.body_ids = [i for i in range(1, self.model.nbody)]
        self.torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot")
        
        for bid in self.body_ids:
            mass = self.model.body_mass[bid]
            
            geom_id = -1
            for g in range(self.model.ngeom):
                if self.model.geom_bodyid[g] == bid:
                    geom_id = g
                    break
            
            radius = 0.01 
            half_length = 0.05  # Store half-length instead of full length

            if geom_id != -1:
                g_size = self.model.geom_size[geom_id]
                g_type = self.model.geom_type[geom_id]
                
                if g_type in [5, 6]: # Cylinder or Capsule
                    radius = g_size[0]
                    if len(g_size) > 1:
                        half_length = g_size[1]
            
            area_side = (half_length * 2.0) * (radius * 2)
            area_end = np.pi * (radius ** 2)
            
            self.body_props[bid] = {
                'mass': mass,
                'vol': mass / target_density,
                'area_side': area_side,
                'area_end': area_end,
                'radius': radius,
                'half_length': half_length,
                'geom_id': geom_id  # <--- 🚨 ADD THIS LINE
            }

    def apply(self, data):
        for i in self.body_ids:
            props = self.body_props[i]
            
            # 1. GET GLOBAL ROTATION MATRIX
            rot = data.xmat[i].reshape(3, 3)

            # 🚨 NEW: GET GLOBAL ROTATION MATRIX OF THE CAPSULE GEOM ITSELF
            g_id = props['geom_id']
            if g_id != -1:
                geom_rot = data.geom_xmat[g_id].reshape(3, 3)
            else:
                geom_rot = rot # Fallback
            
            # --- FIXED SUBMERSION CHECK ---
            z_pos = data.xpos[i][2]
            z_extent_from_length = abs(geom_rot[2, 2] * props['half_length'])
            
            # In MuJoCo, cylinders/capsules are aligned along their local Z-axis.
            # rot[2, 2] gives the global Z-component of the local Z-axis.
            z_extent_from_length = abs(rot[2, 2] * props['half_length'])
            true_vertical_extent = z_extent_from_length + props['radius']
            
            bottom_z = z_pos - true_vertical_extent
            top_z = z_pos + true_vertical_extent
            
            if bottom_z >= self.water_level:
                continue # Fully out of water
                
            if top_z <= self.water_level:
                submerged_ratio = 1.0 # Fully submerged
            else:
                # Smooth transition as the leg exits/enters the water
                submerged_height = self.water_level - bottom_z
                submerged_ratio = np.clip(submerged_height / (true_vertical_extent * 2.0), 0.0, 1.0)
            
            # 2. APPLY BUOYANCY
            f_buoy_mag = self.density * self.gravity * (props['vol'] * submerged_ratio)
            
            # Apply upward linear force
            data.xfrc_applied[i][2] += f_buoy_mag

            if i == self.torso_id:
                local_up = rot[:, 2] # The robot's Z-axis (assuming Z is UP for the torso)
                global_up = np.array([0.0, 0.0, 1.0])
                
                # Cross product calculates the exact axis and amount of tilt
                tilt_axis = np.cross(local_up, global_up)
                
                # 'metacentric_height' is the simulated distance between CoM and Buoyancy. 
                # Increase this number (e.g., to 0.1) if it still rolls. Decrease if it twitches.
                metacentric_height = 0.05 
                
                restoring_torque = tilt_axis * f_buoy_mag * metacentric_height * submerged_ratio
                
                # Apply the stabilizing twist
                data.xfrc_applied[i][3] += restoring_torque[0]
                data.xfrc_applied[i][4] += restoring_torque[1]
                data.xfrc_applied[i][5] += restoring_torque[2]
            # --- 3. APPLY LINEAR DRAG ---
            local_lin_vel = data.cvel[i][3:6]
            vel = rot @ local_lin_vel 
            speed = np.linalg.norm(vel)
            
            # 🚨 NEW: Prevent speed spikes from exploding the v^2 calculation
            speed = min(speed, 10.0) # Cap hydro calculation to max 10 m/s
            
            if speed > 1e-4:
                vel_dir = vel / speed
                axis_z = geom_rot[:, 2]
                
                alignment = abs(np.dot(axis_z, vel_dir))
                area = (alignment * props['area_end']) + ((1.0 - alignment) * props['area_side'])
                cd   = (alignment * self.Cd_end)       + ((1.0 - alignment) * self.Cd_side)
                
                quadratic_drag = 0.5 * self.density * (speed**2) * cd * area
                linear_drag = self.linear_drag_coeff * speed * area 
                
                drag_mag = (quadratic_drag + linear_drag) * submerged_ratio
                
                # 🚨 THE FIX: Clamp the maximum drag force relative to the part's mass!
                # Do not allow water to apply an acceleration greater than 50 Gs
                max_safe_force = props['mass'] * self.gravity * 5.0 
                drag_mag = min(drag_mag, max_safe_force)
                
                f_drag = -drag_mag * vel_dir
                
                data.xfrc_applied[i][0] += f_drag[0]
                data.xfrc_applied[i][1] += f_drag[1]
                data.xfrc_applied[i][2] += f_drag[2]

            # 4. APPLY ANGULAR DRAG
            local_ang_vel = data.cvel[i][0:3] 
            ang_vel = rot @ local_ang_vel
            ang_speed = np.linalg.norm(ang_vel)
            
            if ang_speed > 1e-4:
                # Submerged ratio ensures angular drag fades smoothly too
                torque_drag = -self.angular_drag_coeff * ang_vel * props['vol'] * submerged_ratio
                
                data.xfrc_applied[i][3] += torque_drag[0]
                data.xfrc_applied[i][4] += torque_drag[1]
                data.xfrc_applied[i][5] += torque_drag[2]