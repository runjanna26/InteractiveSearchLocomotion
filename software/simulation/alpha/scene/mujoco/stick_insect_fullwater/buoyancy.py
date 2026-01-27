
import numpy as np

class BuoyancyPhysics:
    def __init__(self, model, water_level=0.0, fluid_density=1000.0):
        self.water_level = water_level
        self.density = fluid_density
        self.gravity = abs(model.opt.gravity[2])
        
        # Stability factors
        self.viscosity_linear = 2.0      # Reduced from 20.0
        self.viscosity_angular = 0.1     # Reduced significantly from 5.0

        # Cache body properties
        self.body_ids = [i for i in range(1, model.nbody)]
        self.body_props = {}
        
        target_density = 800.0 # Make robot slightly lighter than water
        
        for bid in self.body_ids:
            mass = model.body_mass[bid]
            inertia = model.body_inertia[bid] # 3-vector diagonal inertia
            
            # Volume estimation
            vol = mass / target_density
            
            self.body_props[bid] = {
                'vol': vol,
                'mass': mass,
                # Use max inertia component for approximate stability check
                'max_inertia': np.max(inertia) if np.max(inertia) > 1e-6 else 1.0
            }

    def apply(self, data):
        for i in self.body_ids:
            # 1. Check if submerged
            pos = data.xpos[i]
            if pos[2] < self.water_level:
                
                props = self.body_props[i]
                
                # --- A. BUOYANCY (Archimedes) ---
                # F = rho * g * V
                # We scale by how deep it is (simple linear fade-in for smoothness)
                # Depth ratio: Full force at 5cm depth
                depth = self.water_level - pos[2]
                submersion_ratio = np.clip(depth / 0.05, 0.0, 1.0)
                
                f_buoyancy = self.density * self.gravity * props['vol'] * submersion_ratio
                data.xfrc_applied[i][2] += f_buoyancy

                # --- B. DRAG (Damping) ---
                # Get velocities (local frame is better, but world frame cvel is okay for simple drag)
                # cvel format: [rot_x, rot_y, rot_z, lin_x, lin_y, lin_z]
                vel_rot = data.cvel[i][0:3]
                vel_lin = data.cvel[i][3:6]
                
                # Dynamic Stability Clamping
                # We limit the drag force so it never exceeds F = m * v / dt (which would stop it instantly)
                # This prevents the explosion.
                
                # Linear Drag
                f_drag_lin = -self.viscosity_linear * vel_lin
                # Clamp: Don't apply more force than needed to stop it in one step
                max_f = np.abs(vel_lin) * props['mass'] / 0.005 # assuming dt=0.005
                f_drag_lin = np.clip(f_drag_lin, -max_f, max_f)
                
                # Angular Drag
                t_drag_rot = -self.viscosity_angular * vel_rot
                max_t = np.abs(vel_rot) * props['max_inertia'] / 0.005
                t_drag_rot = np.clip(t_drag_rot, -max_t, max_t)

                # Apply
                data.xfrc_applied[i][0] += f_drag_lin[0]
                data.xfrc_applied[i][1] += f_drag_lin[1]
                data.xfrc_applied[i][2] += f_drag_lin[2]
                
                data.xfrc_applied[i][3] += t_drag_rot[0]
                data.xfrc_applied[i][4] += t_drag_rot[1]
                data.xfrc_applied[i][5] += t_drag_rot[2]

