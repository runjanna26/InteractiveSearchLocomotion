import numpy as np
import mujoco

class Hydrodynamics:
    def __init__(self, model, water_level=0.0):
        self.water_level = water_level
        self.density = 1000.0  # Water density (kg/m^3)
        self.model = model
        self.gravity = abs(self.model.opt.gravity[2])
        
        # DRAG COEFFICIENTS
        # Cylinder moving sideways (flat against flow)
        self.Cd_side = 1.2 # <--- INCREASE this to make it "thicker"
        # Cylinder moving end-on (like a needle)
        self.Cd_end = 0.8  # <--- INCREASE this to make it "thicker"
        
        
        # Pre-calculate Body Properties
        self.body_props = {}
        # Skip the world body (index 0)
        self.body_ids = [i for i in range(1, self.model.nbody)]
        
        # Target density to ensure robot floats (500 kg/m^3 = Light wood/plastic)
        target_density = 900.0 

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
            # --- 1. CHECK SUBMERSION ---
            pos = data.xpos[i]
            z_pos = pos[2]
            self.model.opt.viscosity = 0.0000089
            # If above water, skip
            if z_pos >= self.water_level:
                continue
                
            self.model.opt.viscosity = 0.089
            props = self.body_props[i]
            
            # Calculate partial submersion (simple linear fade)
            depth = self.water_level - z_pos
            # Fully submerged if depth > 2x radius
            submerged_ratio = np.clip(depth / (props['radius'] * 2), 0.0, 1.0)
            
            # --- 2. APPLY BUOYANCY (Archimedes) ---
            # F_buoy = rho * g * V_submerged
            f_buoy = self.density * self.gravity * (props['vol'] * submerged_ratio)
            data.xfrc_applied[i][2] += f_buoy
            
            # --- 3. APPLY DRAG (Quadratic) ---
            # Get linear velocity of the body
            vel = data.cvel[i][3:6]
            speed = np.linalg.norm(vel)
            
            if speed < 1e-4:
                continue
                
            # Direction of flow relative to body
            vel_dir = vel / speed
            
            # Determine Area based on orientation
            # Get body orientation matrix (local Z usually points along the limb length)
            rot = data.xmat[i].reshape(3, 3)
            axis_z = rot[:, 2] # Axis of the cylinder
            
            # Dot product: 1.0 = Moving End-on, 0.0 = Moving Sideways
            alignment = abs(np.dot(axis_z, vel_dir))
            
            # Interpolate Area and Cd
            area = (alignment * props['area_end']) + ((1.0 - alignment) * props['area_side'])
            cd   = (alignment * self.Cd_end)       + ((1.0 - alignment) * self.Cd_side)
            
            # Calculate Drag Force Magnitude: F = 0.5 * rho * v^2 * Cd * A
            drag_mag = 0.5 * self.density * (speed**2) * cd * area
            
            # Scale by submersion (no drag on parts in air)
            drag_mag *= submerged_ratio
            
            # Apply Force (Opposite to velocity)
            f_drag = -drag_mag * vel_dir
            
            data.xfrc_applied[i][0] += f_drag[0]
            data.xfrc_applied[i][1] += f_drag[1]
            data.xfrc_applied[i][2] += f_drag[2]