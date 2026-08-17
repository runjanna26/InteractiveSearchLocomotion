# InteractiveSearchLocomotion — Project Overview

_Last updated: 2026-08-17. This file is a map for picking the project back up, not a spec — verify specifics against code before relying on them._

## What this is

A robotics research codebase for legged/amphibious robots that walk on land and swim/walk
underwater across varied, changing terrain ("interactive search locomotion"). Two robot
platforms recur throughout: a 4-legged amphibious "diving beetle" robot (evolved from an
earlier "stick insect" model) and a wheeled swerve-drive base (firmware-branded "HERO" /
"HERO-Alpha", even though its directory is named `dave_loadcell_setup`).

Three research threads tie the repo together:

1. **Environment/terrain classification** — classify what the robot is walking on (or
   whether it's in air vs. water) from onboard joint/leg sensor time-series, to drive
   adaptive gait selection.
2. **Gait weight learning** — CPG+RBF gait generators whose weights are optimized per
   terrain using **PI^BB** (Policy Improvement with Black-Box / Path-Integral policy
   search), a sampling-based black-box policy-gradient method.
3. **Online adaptive impedance control** — compliant "muscle"-style actuator control over
   CAN for real hardware.

## Tech stack

- **Python**: NumPy, MuJoCo Python bindings, PyTorch (CNN/RNN/TCN/HAPTR/STRN models),
  OpenCV, Jupyter notebooks for training/analysis.
- **ROS2** (Foxy/Humble): rclpy, colcon workspaces, `ros2bag` logging, micro-ROS bridging
  to microcontrollers.
- **C++/Arduino**: PlatformIO and ESP-IDF firmware for ESP32/ESP32-S3, using
  `Dynamixel2Arduino` and `micro_ros_arduino` libraries.
- **Simulation**: MuJoCo (current), legacy CoppeliaSim/V-REP `.ttt` scenes (superseded).
- **PlotJuggler** XML layouts for live telemetry visualization.
- Git LFS tracks `*.json`, `*.db3`, `*.npz`, `*.mp4` (trained weights, ROS2 bag DBs,
  demo videos) — see `.gitattributes`. No repo-wide `requirements.txt`/`setup.py`; deps
  live per-ROS2-package (`package.xml`/`setup.py`) or in a few local `requirements.txt`
  files (e.g. `util/ros2bag/adaptive_compliance_control/`).

## Directory map

### `controller/`
- **`dave_loadcell_setup/`** — Two PlatformIO ESP32 firmware projects for the swerve-drive
  wheeled base: `refine_II_clamp_wheel_module_esp32_pio` (ESP32, CAN via ESP32CAN,
  Dynamixel2Arduino, MPU6050 IMU) and `wheel_module_esp32s3_pio` (ESP32-S3, RMD-X4-36 CAN
  actuators, ICM-20948 IMU, ESP-IDF/CMake build, includes a load-cell 2-axis calibration
  source). Publishes swerve feedback, motor torque/current at 100 Hz over micro-ROS.
  Field procedure (per README): SSH into onboard Raspberry Pi → run dockerized
  `micro-ros-agent` → run joystick teleop node.
- **`interaction_control_modules/alpha/`** — ROS2 colcon workspace, two packages:
  `online_adaptive_impedance_control_pkg` (muscle model + learning + CAN drivers —
  `muscle_model*.py`, `muscle_learning.py`, `mit_can.py`, `servo_can.py`) and
  `underwater_robot_pkg`.

### `project/`
- **`alpha/`** — ROS2 workspace scaffolding for deploying onto real hardware:
  `stick_insect_pkg`, `diving_beetle_pkg`, `underwater_robot_rmd_pkg`. Currently mostly
  stub/TODO node files, not yet fleshed out.
- **`beta/`** — empty; reserved for future work.

### `simulation/alpha/`
Scene assets for MuJoCo (`stick_insect`, `stick_insect_v2`, `stick_insect_v3_1dof`,
`underwater_joint`, `underwater_robot_4_legs`) and legacy V-REP scenes (diving_beetle,
stick_insect, 4dof_quadruped_robot) — shows the project's migration from V-REP to MuJoCo
and iteration through stick-insect model versions before the underwater 4-leg variant.

### `util/enviroment_classifier/` — terrain classifier research line (v1 → v6)
Progression of approaches to classifying ground type / air-vs-water from sensor data:
- **v1–v3**: DTW + K-Means clustering, expanding from ground/water to multiple terrains
  (flat/muddy/rough/sandy/sponge).
- **v4 / v4_esn_rr**: Pivots to **ESN-RR** (Echo State Network reservoir + Ridge
  Regression readout) and runs a broad model bake-off against CNN-RNN, TCN, HAPTR, STRN
  across sensor-signal variants (joint kinematics/dynamics, torque, GRF). Compares
  accuracy, model size, noise/dropout robustness, and inference latency — ESN-RR wins on
  speed/size for real-time onboard use.
- **v5_esn_rr**: Integrates ESN-RR into a closed-loop MuJoCo pipeline for the diving-beetle
  robot (`pibb.py`, `script.py`'s `StickInsectEnv`, `hydrodynamic.py` drag/buoyancy model,
  `cpg_rbf/`, `gait_cycle_cut/`).
- **v6**: Adds Growing Neural Gas (GNG) on top of ESN reservoir states — moving toward
  unsupervised/online terrain discovery instead of fixed supervised classes.

### `util/multiterrain_integration/`
The "productionized" integration of the ESN-RR classifier with terrain-adaptive gait
control, in MuJoCo, for the diving-beetle robot:
- `pibb.py` — PI^BB implementation (cost-weighted exponentiated averaging of policy noise,
  decaying exploration temperature `h`).
- `script.py` — `StickInsectEnv`, the MuJoCo env wrapper for training rollouts.
- `replay_learned_weights.py` — replays per-terrain learned CPG-RBF weights
  (solid/soft/slippery/muddy ground, water surface), with an `ENABLE_ESN` toggle to A/B
  test classifier-driven gait switching.
- `main_scene.xml` — MuJoCo scene: pool terrain + `four_legs_diving_beetle_robot`, free
  base joint, force-arrow debug visualization.
- `metric/` — cross-evaluation of every learned gait on every terrain (cost-of-transport
  grid) — a generalization study.
- `video/` — MuJoCo rollout recordings, incl. an ESN-on vs ESN-off ablation.

### `util/optimal_weight_learning/`
Per-robot PIBB gait-weight optimization:
- **`underwater_robot_4_legs/`** — `data/README.md` is a running lab notebook (dated
  2026-07-10 to 2026-07-15) of PIBB training runs with exact hyperparameters (rollouts,
  iterations, kernel/parameter counts, reward-term weights, MuJoCo physics tuning:
  friction, armature, contact flags, drag coefficients, density/buoyancy, viscosity).
  Documents iterative debugging: first successful walk → self-collision penalty → rough
  terrain → swimming variant with left/right asymmetric params.
- **`pongbot_r2/`** — same PIBB pipeline applied to a second robot.

### `util/weight_transition/`
Smooth blending/interpolation between per-terrain learned CPG-RBF weight sets so gait
doesn't snap when the terrain classification changes mid-walk. Includes PCA visualization
of the weight space (`rbf_weights_pca_3d.html`) and earlier imitation-learned swim/walk
weight sets in `weight_cpg_rbf_old/`.

### `util/ros2bag*`, `util/plotjuggler`
- **`ros2bag/adaptive_compliance_control/`** — real hardware bag logs (named after lab
  members, e.g. `muscle_xiaofeng`, `muscle_yang`) for the impedance-control work.
- **`ros2bag_mujoco/`** — sim-side logs comparing air vs. water locomotion (GRF,
  stiffness/damping, energy) and air↔water transitions.
- **`ros2bag_underwateractuator/`** — actuator system-ID sweep: amplitude
  (0.125–1.000) × frequency (0.375–3.000) in air vs. water, with stiffness/damping/energy
  plots.
- **`plotjuggler/`** — XML layouts for live-plotting motor tests and stick-insect sim
  telemetry.

## Open questions / things to double-check next time

- The "dave" naming (`dave_loadcell_setup`) doesn't appear as such in the firmware itself
  — the on-robot branding there is "HERO"/"HERO-Alpha"/"HERO2.1". Worth clarifying which
  name is current if it comes up again.
- `project/beta` is empty — check with the user what it's meant to hold before assuming.
- No repo-wide dependency manifest; environment setup is scattered per-package. If setting
  up a fresh dev environment, expect to hunt through individual `package.xml`/
  `requirements.txt` files.
