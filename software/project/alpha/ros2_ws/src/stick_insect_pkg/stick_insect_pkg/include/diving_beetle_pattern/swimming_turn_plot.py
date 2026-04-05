import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation

# ----------------------------
# Load data
# ----------------------------
df = pd.read_excel('diving_beetle_swimming_turning_trajectory.ods')

time_s = pd.to_numeric(df["Times/ms"], errors="coerce").to_numpy() / 1000.0

R_tau_deg   = df["R_angle τ/°"].to_numpy()
R_alpha_deg = df["R_angle α/°"].to_numpy()
R_beta_deg  = df["R_angle β/°"].to_numpy()

L_tau_deg   = df["L_angle τ/°"].to_numpy()
L_alpha_deg = df["L_angle α/°"].to_numpy()
L_beta_deg  = df["L_angle β/°"].to_numpy()

# Calculate Joint Velocities (deg/s)
v_tau   = np.abs(np.gradient(L_tau_deg, time_s))
v_alpha = np.abs(np.gradient(L_alpha_deg, time_s))
v_beta  = np.abs(np.gradient(L_beta_deg, time_s))

# ----------------------------
# Link lengths (mm)
# ----------------------------
L1, L2, L4 = 9.5, 4.6, 14.5

# ----------------------------
# Forward kinematics
# ----------------------------
def R_compute_points(tau_deg, alpha_deg, beta_deg):
    tau   = np.radians(tau_deg)
    alpha = np.radians(alpha_deg)
    beta  = np.radians(beta_deg)

    p1 = np.array([0, 0])

    p2 = np.array([
        L1 * np.sin(tau),
        L1 * np.cos(tau)
    ])

    p3 = np.array([
        L2 * np.sin(alpha - tau) + p2[0],
        -L2 * np.cos(alpha - tau) + p2[1]
    ])

    p5 = np.array([
        -L4 * np.sin(alpha + beta - tau) + p3[0],
         L4 * np.cos(alpha + beta - tau) + p3[1]
    ])

    return np.vstack([p1, p2, p3, p5])

def L_compute_points(tau_deg, alpha_deg, beta_deg):

    tau   = np.radians(tau_deg)
    alpha = np.radians(alpha_deg)
    beta  = np.radians(beta_deg)

    p1 = np.array([0, 0])

    # Hip
    p2 = np.array([
        -L1 * np.sin(tau),
         L1 * np.cos(tau)
    ])

    # Knee
    p3 = np.array([
        -L2 * np.sin(alpha - tau) + p2[0],
        -L2 * np.cos(alpha - tau) + p2[1]
    ])

    # Tarsus tip
    p5 = np.array([
         L4 * np.sin(alpha + beta - tau) + p3[0],
         L4 * np.cos(alpha + beta - tau) + p3[1]
    ])

    return np.vstack([p1, p2, p3, p5])

# ----------------------------
# Setup figure
# ----------------------------
fig, ax = plt.subplots(figsize=(8, 6))
manager = plt.get_current_fig_manager()
manager.full_screen_toggle()
ax.set_xlim(-40, 40)
ax.set_ylim(-40, 40)
ax.set_aspect('equal')

# Right and left leg lines
right_leg, = ax.plot([], [], '-o', linewidth=3)
left_leg,  = ax.plot([], [], '-o', linewidth=3)

right_trajectory_line, = ax.plot([], [], linewidth=1)
left_trajectory_line,  = ax.plot([], [], linewidth=1)


# Text object for real-time velocity display
vel_text = ax.text(-38, 35, '', fontsize=10, family='monospace', 
                   bbox=dict(facecolor='white', alpha=0.7))

right_tip_path_x = []
right_tip_path_y = []
left_tip_path_x = []
left_tip_path_y = []
# ----------------------------
# Animation update
# ----------------------------
def update(frame):

    # Right leg
    right_points = R_compute_points(
        R_tau_deg[frame],
        R_alpha_deg[frame],
        R_beta_deg[frame]
    )

    left_points = L_compute_points(
        L_tau_deg[frame],
        L_alpha_deg[frame],
        L_beta_deg[frame]
    )

    right_leg.set_data(right_points[:,0], right_points[:,1])
    left_leg.set_data(left_points[:,0], left_points[:,1])

    # Store tip trajectory
    right_tip_path_x.append(right_points[-1,0])
    right_tip_path_y.append(right_points[-1,1])
    right_trajectory_line.set_data(right_tip_path_x, right_tip_path_y)

    left_tip_path_x.append(left_points[-1,0])
    left_tip_path_y.append(left_points[-1,1])
    left_trajectory_line.set_data(left_tip_path_x, left_tip_path_y)


    v_tau_rpm = v_tau / 6
    v_alpha_rpm = v_alpha / 6
    v_beta_rpm = v_beta / 6

    current_max_v = max(v_tau_rpm[frame], v_alpha_rpm[frame], v_beta_rpm[frame])
    vel_text.set_text(f'(Left)Max Joint Vel: {current_max_v / 6:.1f} RPM\n'
                        f'τ: {v_tau_rpm[frame] / 6:.1f} RPM | α: {v_alpha_rpm[frame] / 6:.1f} RPM | β: {v_beta_rpm[frame] / 6:.1f} RPM')

    return right_leg, left_leg, right_trajectory_line, left_trajectory_line


ani = FuncAnimation(
    fig,
    update,
    frames=len(R_tau_deg),
    interval=2,
    blit=True)

plt.title("Diving Beetle Hind Legs: Right-turning-swimming Motion")
plt.xlabel("X (mm)")
plt.ylabel("Y (mm)")
plt.show()