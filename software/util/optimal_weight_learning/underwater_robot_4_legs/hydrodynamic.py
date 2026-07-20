import numpy as np
import mujoco

class Hydrodynamics:
    def __init__(self, model, water_level=0.0):
        self.water_level = water_level
        self.density = 1000.0  # Water density (kg/m^3)
        self.model = model
        self.gravity = abs(self.model.opt.gravity[2])
        
        # Simulation parameters (tune these for more/less drag and buoyancy)
        # DRAG COEFFICIENTS
        # Cylinder moving sideways (flat against flow)
        self.Cd_side = 6.0 # <--- INCREASE this to make it "thicker" 1.2
        # Cylinder moving end-on (like a needle)
        self.Cd_end = 0.5  # <--- INCREASE this to make it "thicker" 0.8
        
        self.linear_drag_coeff = 40.0 
        self.angular_drag_coeff = 10.0 # Tune this if it spins too much or too little

        # Robot density to ensure robot floats (500 kg/m^3 = Light wood/plastic)
        target_density = 900 


        # Pre-calculate Body Properties
        self.body_props = {}
        # Skip the world body (index 0)
        self.body_ids = [i for i in range(1, self.model.nbody)]
        self.torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot")
        


        for bid in self.body_ids:
            # 1. Get Mass
            mass = self.model.body_mass[bid]
            
            # 2. Estimate Shape Dimensions (Radius & Length)
            # We look for the first geom attached to this body
            geom_id = -1
            for g in range(self.model.ngeom):
                if self.model.geom_bodyid[g] == bid:
                    geom_id = g
                    break
            
            radius = 0.01 # Default fallback
            length = 0.1  # Default fallback

            if geom_id != -1:
                # Get size: [radius, half_length, ...] for cylinder/capsule
                g_size = self.model.geom_size[geom_id]
                g_type = self.model.geom_type[geom_id]
                
                # Type 5=Cylinder, 6=Capsule
                if g_type in [5, 6]:
                    radius = g_size[0]
                    # Check if length is defined in size (index 1)
                    if len(g_size) > 1:
                        length = g_size[1] * 2.0 
            
            # 3. Calculate Projected Areas
            # Area when moving sideways (Rectangle: L * D)
            area_side = length * (radius * 2)
            # Area when moving end-on (Circle: pi * r^2)
            area_end = np.pi * (radius ** 2)
            
            self.body_props[bid] = {
                'mass': mass,
                'vol': mass / target_density, # Volume for buoyancy
                'area_side': area_side,
                'area_end': area_end,
                'radius': radius
            }

    def apply(self, data):
        for i in self.body_ids:
            pos = data.xpos[i]
            z_pos = pos[2]
            props = self.body_props[i]
            
            # --- 1. CORRECTED SUBMERSION CHECK ---
            bottom_z = z_pos - props['radius']
            if bottom_z >= self.water_level:
                continue
                
            submerged_height = self.water_level - bottom_z
            submerged_ratio = np.clip(submerged_height / (props['radius'] * 2.0), 0.0, 1.0)
            
            # --- 2. APPLY BUOYANCY ---
            f_buoy = self.density * self.gravity * (props['vol'] * submerged_ratio)
            data.xfrc_applied[i][2] += f_buoy
            
            
            if i == self.torso_id:
                continue # Skip the rest of the loop for the torso!

            # ==========================================
            # GET GLOBAL ROTATION MATRIX
            # ==========================================
            rot = data.xmat[i].reshape(3, 3)

            # --- 3. APPLY LINEAR DRAG ---
            local_lin_vel = data.cvel[i][3:6]
            
            # 🚨 THE FIX: Rotate local velocity into the global world frame!
            vel = rot @ local_lin_vel 
            speed = np.linalg.norm(vel)
            
            if speed > 1e-4:
                vel_dir = vel / speed
                axis_z = rot[:, 2] # Global direction of the part's length
                
                alignment = abs(np.dot(axis_z, vel_dir))
                area = (alignment * props['area_end']) + ((1.0 - alignment) * props['area_side'])
                cd   = (alignment * self.Cd_end)       + ((1.0 - alignment) * self.Cd_side)
                
                quadratic_drag = 0.5 * self.density * (speed**2) * cd * area
                linear_drag = self.linear_drag_coeff * speed * area 
                
                drag_mag = (quadratic_drag + linear_drag) * submerged_ratio
                f_drag = -drag_mag * vel_dir
                
                # Apply global drag to global xfrc_applied
                data.xfrc_applied[i][0] += f_drag[0]
                data.xfrc_applied[i][1] += f_drag[1]
                data.xfrc_applied[i][2] += f_drag[2]

            # --- 4. APPLY ANGULAR DRAG ---
            local_ang_vel = data.cvel[i][0:3] 
            
            # 🚨 THE FIX: Rotate local angular velocity into global frame!
            ang_vel = rot @ local_ang_vel
            ang_speed = np.linalg.norm(ang_vel)
            
            if ang_speed > 1e-4:
                torque_drag = -self.angular_drag_coeff * ang_vel * props['vol'] * submerged_ratio
                
                data.xfrc_applied[i][3] += torque_drag[0]
                data.xfrc_applied[i][4] += torque_drag[1]
                data.xfrc_applied[i][5] += torque_drag[2]