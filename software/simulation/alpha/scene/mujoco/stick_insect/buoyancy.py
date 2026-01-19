# import numpy as np

# class BuoyancyPhysics:
#     def __init__(self, model, water_level=0.0, fluid_density=1000.0):
#         self.water_level = water_level
#         self.density = fluid_density
#         self.gravity = abs(model.opt.gravity[2])
        
#         # Viscosity coefficients (Water is thick)
#         self.linear_drag = 1.0  # Resists movement
#         self.angular_drag = 0.05  # Resists rotation

#         # self.linear_drag = 0.0  # Resists movement
#         # self.angular_drag = 0.0  # Resists rotation

#         # Identify bodies to apply buoyancy to
#         # We skip the "world" body (ID 0)
#         self.body_ids = [i for i in range(1, model.nbody)]
        
#         # Pre-calculate estimated volume of each body based on mass
#         # We assume the robot is slightly lighter than water overall (density ~900)
#         # to ensure it floats. V = m / rho
#         self.body_volumes = {}
#         target_density = 800.0 # kg/m^3 (Less than water 1000, so it floats)
        
#         for bid in self.body_ids:
#             mass = model.body_mass[bid]
#             self.body_volumes[bid] = mass / target_density

#     def apply(self, data):
#         """
#         Calculates and applies buoyancy + drag forces to data.xfrc_applied
#         """
#         # Iterate over all moving bodies
#         for i in self.body_ids:
#             # 1. Get Body Position (Center of Mass)
#             pos = data.xpos[i]
            
#             # Check if submerged
#             if pos[2] < self.water_level:
#                 # --- A. BUOYANCY FORCE ---
#                 # Archimedes: F = rho * g * V_submerged
#                 # Simple approximation: Fully submerged if COM is below water
#                 vol = self.body_volumes[i]
#                 buoyancy_force = self.density * self.gravity * vol
                
#                 # Apply UPWARD force (Z-axis)
#                 # xfrc_applied index [0-2] is Force, [3-5] is Torque
#                 data.xfrc_applied[i][2] += buoyancy_force

#                 # --- B. DRAG FORCE (Damping) ---
#                 # Water resists motion. F_drag = -b * v
#                 # data.cvel gives 6D velocity (3 rot, 3 lin) in world frame?
#                 # Actually data.cvel is [rot_x, rot_y, rot_z, lin_x, lin_y, lin_z]
                
#                 velocity = data.cvel[i]
                
#                 # Linear Drag (opposes movement)
#                 data.xfrc_applied[i][0] -= self.linear_drag * velocity[3]
#                 data.xfrc_applied[i][1] -= self.linear_drag * velocity[4]
#                 data.xfrc_applied[i][2] -= self.linear_drag * velocity[5]
                
#                 # Rotational Drag (opposes spinning)
#                 data.xfrc_applied[i][3] -= self.angular_drag * velocity[0]
#                 data.xfrc_applied[i][4] -= self.angular_drag * velocity[1]
#                 data.xfrc_applied[i][5] -= self.angular_drag * velocity[2]

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