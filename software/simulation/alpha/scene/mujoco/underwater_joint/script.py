import mujoco
import mujoco.viewer
import numpy as np
import time
import matplotlib.pyplot as plt
import os

# ==========================================
# 0. XML GENERATOR (Helper for Triple Pendulum)
# ==========================================
def create_triple_pendulum_xml():
    xml_content = """
    <mujoco model="triple_pendulum">
      <option timestep="0.002" gravity="0 0 -9.81"/>
      <visual>
        <rgba haze="0.15 0.25 0.35 1"/>
        <quality shadowsize="2048"/>
      </visual>
      <asset>
        <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="512"/> 
      </asset>

      <worldbody>
        <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
        <geom type="plane" size="2 2 0.1" rgba=".9 .9 .9 1"/>

        <body pos="0 0 2.5" euler="0 0 0">
          <joint name="joint1" type="hinge" axis="0 1 0" pos="0 0 0" damping="0.2"/>
          <geom type="capsule" size="0.05" fromto="0 0 0 0.5 0 0" rgba="0.8 0.2 0.2 1" mass="1"/>
          
          <body pos="0.5 0 0">
            <joint name="joint2" type="hinge" axis="0 1 0" pos="0 0 0" damping="0.2"/>
            <geom type="capsule" size="0.04" fromto="0 0 0 0.5 0 0" rgba="0.2 0.8 0.2 1" mass="1"/>
            
            <body pos="0.5 0 0">
              <joint name="joint3" type="hinge" axis="0 1 0" pos="0 0 0" damping="0.2"/>
              <geom type="capsule" size="0.03" fromto="0 0 0 0.5 0 0" rgba="0.2 0.2 0.8 1" mass="1"/>
              <site name="tip" pos="0.5 0 0"/>
            </body>
          </body>
        </body>
      </worldbody>

      <actuator>
        <motor name="torque1" joint="joint1" gear="1" ctrllimited="true" ctrlrange="-50 50"/>
        <motor name="torque2" joint="joint2" gear="1" ctrllimited="true" ctrlrange="-30 30"/>
        <motor name="torque3" joint="joint3" gear="1" ctrllimited="true" ctrlrange="-20 20"/>
      </actuator>
    </mujoco>
    """
    with open("robot_triple.xml", "w") as f:
        f.write(xml_content)

# ==========================================
# 1. ADAPTIVE CONTROL CLASS
# ==========================================
class MuscleModel:
    def __init__(self, _a: float, _b: float, _beta: float, _init_pos: float):
        self.pos_init = _init_pos
        self.a = _a 
        self.b = _b 
        self.beta = _beta 

        self.pos_des = 0.0        
        self.pos_fb = 0.0
        self.pos_des_prev = 0.0 
        self.vel_des = 0.0
        self.vel_fb = 0.0
        
        # Internal States
        self.K = 0.0 
        self.D = 0.0 
        self.F = 0.0 
        
    def calculate(self, pos_des, pos_fb, vel_fb, dt):
        self.pos_des = pos_des
        self.pos_fb = pos_fb
        self.vel_fb = vel_fb

        # Calculate Desired Velocity
        safe_dt = dt if dt > 1e-6 else 1e-3
        self.vel_des = (self.pos_des - self.pos_des_prev) / safe_dt

        # Calculate Adaptation
        adapt_scalar = self.gen_adapt_scalar()
        if abs(adapt_scalar) < 1e-6: adapt_scalar = 1e-6 

        self.F = self.gen_track_error() / adapt_scalar
        self.K = self.F * self.gen_pos_error()
        self.D = self.F * self.gen_vel_error()

        self.pos_des_prev = self.pos_des

    def gen_pos_error(self):
        return (self.pos_fb - self.pos_init) - self.pos_des

    def gen_vel_error(self):
        return self.vel_fb - self.vel_des

    def gen_track_error(self):
        return self.gen_pos_error() + self.beta * self.gen_vel_error()

    def gen_adapt_scalar(self):
        return self.a / (1.0 + (self.b * (np.abs(self.gen_track_error()))**2))
    
    def get_stiffness(self):
        return np.float64(self.limit_value(self.K, 0.0, 500.0))
    
    def get_damping(self):
        return np.float64(self.limit_value(abs(self.D), 0.0, 5.0))
    
    def get_feedforward_force(self):
        return np.float64(self.limit_value(self.F, -50.0, 50.0)) # Increased limit for triple pendulum base
    
    def limit_value(self, value, min_val, max_val):
        return max(min_val, min(value, max_val))

# ==========================================
# 2. REAL-TIME PLOTTER CLASS (Updated for Triple Pendulum)
# ==========================================
class RealtimePlotter:
    def __init__(self, window_size=500):
        self.window_size = window_size
        self.times = []
        
        # Storing positions for all 3 joints
        self.j1_pos = []
        self.j2_pos = []
        self.j3_pos = []
        self.target_pos = []

        # Storing internal states ONLY for Joint 1 (to avoid clutter)
        self.k_vals = []
        self.d_vals = []
        self.torque_vals = []

        # Setup Plot
        plt.ion()
        self.fig, self.axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
        self.fig.suptitle("Adaptive Control: Triple Pendulum")

        # 1. Position Plot (All 3 Joints)
        self.line_target, = self.axs[0].plot([], [], 'k--', linewidth=1, label='Reference')
        self.line_j1, = self.axs[0].plot([], [], 'r-', label='Joint 1')
        self.line_j2, = self.axs[0].plot([], [], 'g-', label='Joint 2')
        self.line_j3, = self.axs[0].plot([], [], 'b-', label='Joint 3')
        self.axs[0].set_ylabel("Position (rad)")
        self.axs[0].legend(loc="upper right", fontsize='small')
        self.axs[0].grid(True)

        # 2. Impedance Plot (Joint 1 Only)
        self.line_k, = self.axs[1].plot([], [], 'r-', label='J1 Stiffness (K)')
        self.ax1_twin = self.axs[1].twinx()
        self.line_d, = self.ax1_twin.plot([], [], 'm-', label='J1 Damping (D)')
        self.axs[1].set_ylabel("Stiffness (K)", color='r')
        self.ax1_twin.set_ylabel("Damping (D)", color='m')
        self.axs[1].grid(True)

        # 3. Torque Plot (Joint 1 Only)
        self.line_tau, = self.axs[2].plot([], [], 'k-', label='J1 Torque (Nm)')
        self.axs[2].set_ylabel("Torque (Nm)")
        self.axs[2].set_xlabel("Time (s)")
        self.axs[2].grid(True)
        
        plt.tight_layout()

    def update_data(self, t, target, q_vals, k, d, tau):
        self.times.append(t)
        self.target_pos.append(target)
        self.j1_pos.append(q_vals[0])
        self.j2_pos.append(q_vals[1])
        self.j3_pos.append(q_vals[2])
        self.k_vals.append(k)
        self.d_vals.append(d)
        self.torque_vals.append(tau)

        if len(self.times) > self.window_size:
            self.times.pop(0)
            self.target_pos.pop(0)
            self.j1_pos.pop(0)
            self.j2_pos.pop(0)
            self.j3_pos.pop(0)
            self.k_vals.pop(0)
            self.d_vals.pop(0)
            self.torque_vals.pop(0)

    def draw(self):
        self.line_target.set_data(self.times, self.target_pos)
        self.line_j1.set_data(self.times, self.j1_pos)
        self.line_j2.set_data(self.times, self.j2_pos)
        self.line_j3.set_data(self.times, self.j3_pos)

        self.line_k.set_data(self.times, self.k_vals)
        self.line_d.set_data(self.times, self.d_vals)
        self.line_tau.set_data(self.times, self.torque_vals)

        for ax in self.axs:
            ax.relim()
            ax.autoscale_view()
        self.ax1_twin.relim()
        self.ax1_twin.autoscale_view()
        
        self.fig.canvas.flush_events()

# ==========================================
# 3. MAIN SIMULATION LOOP
# ==========================================
def main():
    create_triple_pendulum_xml()
    model = mujoco.MjModel.from_xml_path('robot_triple.xml')
    data = mujoco.MjData(model)

    # IDs for our 3 joints and actuators
    joint_names = ["joint1", "joint2", "joint3"]
    act_names = ["torque1", "torque2", "torque3"]
    
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in joint_names]
    act_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in act_names]

    # Initialize 3 Independent Controllers
    # We tune them slightly differently. The base joint (0) needs more power than the tip (2).
    controllers = [
        MuscleModel(_a=0.5, _b=10.0, _beta=0.8, _init_pos=0.0), # Joint 1 (Base)
        MuscleModel(_a=0.3, _b=5.0, _beta=0.6, _init_pos=0.0),  # Joint 2 (Middle)
        MuscleModel(_a=0.1, _b=2.0, _beta=0.4, _init_pos=0.0)   # Joint 3 (Tip)
    ]

    plotter = RealtimePlotter(window_size=200)

    last_plot_time = 0
    plot_interval = 0.05 

    print("Starting Triple Pendulum Simulation...")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = time.time()
        
        while viewer.is_running():
            step_start = time.time()
            t = data.time

            # --- Generate Targets ---
            # We create a wave motion: subsequent joints have a phase lag
            base_target = (np.pi/3 * np.sin(2.0 * t))
            targets = [
                base_target,                    # Joint 1 follows sine wave
                0.5 * np.sin(2.0 * t - 1.0),    # Joint 2 has phase lag and smaller amplitude
                0.5 * np.sin(2.0 * t - 2.0)     # Joint 3 has more phase lag
            ]

            # --- Control Loop for All 3 Joints ---
            q_vals = []
            
            for i in range(3):
                # Get current state
                q = data.qpos[joint_ids[i]]
                dq = data.qvel[joint_ids[i]]
                q_vals.append(q)

                # Calculate Control
                controllers[i].calculate(targets[i], q, dq, model.opt.timestep)

                K = controllers[i].get_stiffness()
                D = controllers[i].get_damping()
                F = controllers[i].get_feedforward_force()

                e_pos = controllers[i].gen_pos_error()
                e_vel = controllers[i].gen_vel_error()

                # Apply Torque
                tau = -F - (K * e_pos) - (D * e_vel)
                data.ctrl[act_ids[i]] = tau

            mujoco.mj_step(model, data)

            # --- Data Logging ---
            if (time.time() - last_plot_time) > plot_interval:
                # We log all positions, but only K/D/Tau for Joint 1 (Base)
                j1_k = controllers[0].get_stiffness()
                j1_d = controllers[0].get_damping()
                j1_tau = data.ctrl[act_ids[0]]
                
                plotter.update_data(t, targets[0], q_vals, j1_k, j1_d, j1_tau)
                plotter.draw()
                last_plot_time = time.time()

            viewer.sync()
            
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()