import time
import numpy as np
import mujoco
import mujoco.viewer
import rclpy

from ros2_node import StickInsectNode
from muscle_model import MuscleModel


# ======================================================
# CONSTANTS
# ======================================================

FOOT_NAMES = ['foot_r0', 'foot_r1', 'foot_r2',
              'foot_l0', 'foot_l1', 'foot_l2']

JOINT_GROUPS = ['TR', 'CR', 'FR', 'TL', 'CL', 'FL']
NUM_JOINTS_PER_GROUP = 3


# ======================================================
# INITIALIZATION UTILITIES
# ======================================================

def get_actuator_name(model, idx):
    addr = model.name_actuatoradr[idx]
    return model.names[addr:].decode().split('\x00', 1)[0]


def init_controllers(model):
    controllers = {}
    actuator_ids = {}
    qpos_ids = {}
    qvel_ids = {}

    for i in range(model.nu):
        name = get_actuator_name(model, i)
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)

        if joint_id == -1:
            print(f"[SKIP] No joint for actuator {name}")
            continue

        actuator_ids[name] = i
        qpos_ids[name] = model.jnt_qposadr[joint_id]
        qvel_ids[name] = model.jnt_dofadr[joint_id]

        controllers[name] = MuscleModel(
            _a=0.2,
            _b=5.0,
            _beta=0.0,
            _init_pos=0.0
        )

    return controllers, actuator_ids, qpos_ids, qvel_ids


def init_foot_sensors(model):
    foot_geom_ids = {}

    for name in FOOT_NAMES:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)

        if gid == -1:
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)

        if gid == -1:
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name.upper())

        if gid != -1:
            foot_geom_ids[name] = gid
            print(f"[OK] Found foot sensor {name} (ID {gid})")
        else:
            print(f"[WARN] Missing foot sensor {name}")

    return foot_geom_ids


# ======================================================
# TARGET GENERATION
# ======================================================

def get_joint_targets(ros_node, elapsed, controllers):
    targets = {name: 0.0 for name in controllers}

    if elapsed < 5.0:
        return targets

    for group in JOINT_GROUPS:
        for i in range(NUM_JOINTS_PER_GROUP):
            joint_name = f"{group}{i}"
            targets[joint_name] = ros_node.joint_cmd[group][i]

    return targets


# ======================================================
# GROUND REACTION FORCE
# ======================================================

def get_grf(model, data, foot_geom_ids):
    grf = {name: 0.0 for name in FOOT_NAMES}

    for i in range(data.ncon):
        contact = data.contact[i]

        for name, gid in foot_geom_ids.items():
            if contact.geom1 == gid or contact.geom2 == gid:
                force = np.zeros(6)
                mujoco.mj_contactForce(model, data, i, force)
                grf[name] += force[0]

    return [grf[name] for name in FOOT_NAMES]


# ======================================================
# MAIN SIMULATION
# ======================================================

def main(args=None):
    rclpy.init(args=args)
    ros_node = StickInsectNode()

    model = mujoco.MjModel.from_xml_path("./stick_insect.xml")
    data = mujoco.MjData(model)

    controllers, actuator_ids, qpos_ids, qvel_ids = init_controllers(model)
    foot_geom_ids = init_foot_sensors(model)

    print(f"Controllers: {len(controllers)}")
    print(f"Foot sensors: {len(foot_geom_ids)}")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = time.time()

        while viewer.is_running():
            step_start = time.time()
            rclpy.spin_once(ros_node, timeout_sec=0.0)

            elapsed = step_start - start_time
            targets = get_joint_targets(ros_node, elapsed, controllers)

            # --- CONTROL UPDATE ---
            for name, ctrl in controllers.items():
                q = data.qpos[qpos_ids[name]]
                dq = data.qvel[qvel_ids[name]]
                target = targets[name]

                ctrl.calculate(target, q, dq, model.opt.timestep)
                data.ctrl[actuator_ids[name]] = ctrl.get_torque()

            # --- GRF FEEDBACK ---
            grf_data = get_grf(model, data, foot_geom_ids)
            ros_node.publish_grf(grf_data)

            # --- PHYSICS STEP ---
            mujoco.mj_step(model, data)
            viewer.sync()

            with viewer.lock():
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True

            # --- REALTIME SYNC ---
            sleep_time = model.opt.timestep - (time.time() - step_start)
            if sleep_time > 0:
                time.sleep(sleep_time)


if __name__ == "__main__":
    main()
