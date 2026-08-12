import numpy as np
import mujoco

class Hydrodynamics:
    def __init__(self, model, water_level=0.0):
        self.water_level = water_level
        self.density = 1000.0  
        self.model = model
        self.gravity = abs(self.model.opt.gravity[2])
        
        # Base drag coefficients
        self.Cd_side = 10.0 
        self.Cd_end = 0.01  
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
                # Find the primary collision geom for this body
                if self.model.geom_bodyid[g] == bid and self.model.geom_contype[g] != 0:
                    geom_id = g
                    break
            
            # If no collision geom is found, grab ANY geom attached to the body
            if geom_id == -1:
                for g in range(self.model.ngeom):
                    if self.model.geom_bodyid[g] == bid:
                        geom_id = g
                        break

            # 🚨 THE FIX: Store the raw size and type instead of hardcoding radius
            if geom_id != -1:
                g_size = self.model.geom_size[geom_id]
                g_type = self.model.geom_type[geom_id]
            else:
                g_size = np.array([0.01, 0.01, 0.01])
                g_type = -1 # Unknown

            self.body_props[bid] = {
                'mass': mass,
                'vol': mass / target_density,
                'g_size': np.copy(g_size),
                'g_type': g_type,
                'geom_id': geom_id
            }

    def apply(self, data):
        for i in self.body_ids:
            props = self.body_props[i]
            g_size = props['g_size']
            g_type = props['g_type']
            
            # 1. GET GLOBAL ROTATION MATRIX
            rot = data.xmat[i].reshape(3, 3)

            g_id = props['geom_id']
            if g_id != -1:
                geom_rot = data.geom_xmat[g_id].reshape(3, 3)
                cop_pos = data.geom_xpos[g_id]  # Center of Pressure (the geom center)
            else:
                geom_rot = rot # Fallback
                cop_pos = data.xipos[i]         # Fallback to Center of Mass
            
            # Distance vector from Body Center of Mass to the Geometry Center (Paddle)
            r_vector = cop_pos - data.xipos[i]

            # ==========================================
            # 2. DYNAMIC SUBMERSION HEIGHT
            # ==========================================
            z_pos = cop_pos[2] # Use the actual geom center height, not the body joint
            
            # Check if BOX (Type 2)
            if g_type == mujoco.mjtGeom.mjGEOM_BOX: 
                # Box exact vertical extent based on its 3D rotation
                true_vertical_extent = (abs(geom_rot[2, 0] * g_size[0]) + 
                                        abs(geom_rot[2, 1] * g_size[1]) + 
                                        abs(geom_rot[2, 2] * g_size[2]))
            else:
                # Fallback for Capsules (Type 7) and Cylinders (Type 5)
                radius = g_size[0]
                half_length = g_size[1] if len(g_size) > 1 else 0.0
                true_vertical_extent = abs(geom_rot[2, 2] * half_length) + radius
            
            bottom_z = z_pos - true_vertical_extent
            top_z = z_pos + true_vertical_extent
            
            if bottom_z >= self.water_level:
                continue # Fully out of water
                
            if top_z <= self.water_level:
                submerged_ratio = 1.0 # Fully submerged
            else:
                submerged_height = self.water_level - bottom_z
                submerged_ratio = np.clip(submerged_height / (true_vertical_extent * 2.0), 0.0, 1.0)
            
            # ==========================================
            # 3. APPLY BUOYANCY
            # ==========================================
            f_buoy_mag = self.density * self.gravity * (props['vol'] * submerged_ratio)
            data.xfrc_applied[i][2] += f_buoy_mag

            # # ==========================================
            # # 4. GET EXACT VELOCITY AT CENTER OF PRESSURE
            # # ==========================================
            # obj_vel = np.zeros(6)
            # mujoco.mj_objectVelocity(self.model, data, mujoco.mjtObj.mjOBJ_BODY, i, obj_vel, 0)
            
            # ang_vel = obj_vel[0:3]
            # com_vel = obj_vel[3:6]
            
            # # Velocity at the Paddle = CoM Velocity + (Angular Velocity × Lever Arm)
            # vel = com_vel + np.cross(ang_vel, r_vector)
            # speed = np.linalg.norm(vel)
            
            # ==========================================
            # 4. GET EXACT VELOCITY AT CENTER OF PRESSURE
            # ==========================================
            obj_vel = np.zeros(6, dtype=np.float64)
            mujoco.mj_objectVelocity(self.model, data, mujoco.mjtObj.mjOBJ_BODY, i, obj_vel, 0)
            
            # Explicitly cast to standard numpy arrays to prevent 'moveaxis' dimension crashes
            ang_vel = np.array(obj_vel[0:3], dtype=np.float64)
            com_vel = np.array(obj_vel[3:6], dtype=np.float64)
            r_vector = np.array(cop_pos, dtype=np.float64) - np.array(data.xipos[i], dtype=np.float64)
            
            # Velocity at the Paddle = CoM Velocity + (Angular Velocity × Lever Arm)
            vel = com_vel + np.cross(ang_vel, r_vector)
            speed = np.linalg.norm(vel)
            
            # ==========================================
            # 5. DYNAMIC DRAG CALCULATION
            # ==========================================
            if speed > 1e-4:
                vel_dir = vel / speed
                
                # IF THE SHAPE IS A BOX (PADDLE)
                if g_type == mujoco.mjtGeom.mjGEOM_BOX:
                    axis_x = geom_rot[:, 0]
                    axis_y = geom_rot[:, 1]
                    axis_z = geom_rot[:, 2]
                    
                    align_x = abs(np.dot(axis_x, vel_dir))
                    align_y = abs(np.dot(axis_y, vel_dir))
                    align_z = abs(np.dot(axis_z, vel_dir))
                    
                    area_x = 4.0 * g_size[1] * g_size[2]
                    area_y = 4.0 * g_size[0] * g_size[2]
                    area_z = 4.0 * g_size[0] * g_size[1]
                    
                    area = (align_x * area_x) + (align_y * area_y) + (align_z * area_z)
                    cd = self.Cd_side # Box uses uniform high drag for flat surfaces
                    
                # IF THE SHAPE IS A CAPSULE/CYLINDER (THIGH/HIP)
                else:
                    axis_z = geom_rot[:, 2]
                    alignment = abs(np.dot(axis_z, vel_dir))
                    
                    radius = g_size[0]
                    half_length = g_size[1] if len(g_size) > 1 else radius
                    
                    area_side = (half_length * 2.0) * (radius * 2.0)
                    area_end = np.pi * (radius ** 2)
                    
                    area = (alignment * area_end) + ((1.0 - alignment) * area_side)
                    cd   = (alignment * self.Cd_end) + ((1.0 - alignment) * self.Cd_side)
                
                # Apply Final Drag Forces
                quadratic_drag = 0.5 * self.density * (speed**2) * cd * area
                linear_drag = self.linear_drag_coeff * speed * area 
                
                drag_mag = (quadratic_drag + linear_drag) * submerged_ratio
                
                max_safe_force = props['mass'] * self.gravity * 50.0 
                drag_mag = min(drag_mag, max_safe_force)
                
                f_drag = -drag_mag * vel_dir
                
                # 🚨 APPLY FORCE + INDUCED TORQUE FROM LEVER ARM
                induced_torque = np.cross(r_vector, f_drag)
                
                # Add Drag Force
                data.xfrc_applied[i][0] += f_drag[0]
                data.xfrc_applied[i][1] += f_drag[1]
                data.xfrc_applied[i][2] += f_drag[2]
                
                # Add Torque caused by pushing water with a long leg
                data.xfrc_applied[i][3] += induced_torque[0]
                data.xfrc_applied[i][4] += induced_torque[1]
                data.xfrc_applied[i][5] += induced_torque[2]

            # ==========================================
            # 6. APPLY ANGULAR DAMPING
            # ==========================================
            ang_speed = np.linalg.norm(ang_vel)
            if ang_speed > 1e-4:
                torque_drag = -self.angular_drag_coeff * ang_vel * props['vol'] * submerged_ratio
                data.xfrc_applied[i][3] += torque_drag[0]
                data.xfrc_applied[i][4] += torque_drag[1]
                data.xfrc_applied[i][5] += torque_drag[2]