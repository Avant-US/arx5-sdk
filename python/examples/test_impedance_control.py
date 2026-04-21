#!/usr/bin/env python3
"""
Cartesian Impedance Controller for ARX5 using arx5-sdk torque interface.

This script implements a Cartesian impedance controller that:
1. Uses Pinocchio for kinematics/dynamics (FK, Jacobian, gravity compensation)
2. Uses arx5-sdk for hardware interface (reading state, sending torques)
3. Controls the robot to maintain a fixed desired pose with compliant behavior

Usage:
    python test_impedance_control.py X5 can0
"""

import time
import os
import sys
import signal
import numpy as np
import pinocchio as pin
import click
import matplotlib.pyplot as plt
from collections import deque
import threading

# import mujoco
# import mujoco.viewer
import yaml

_sigint_count = 0
def _sigint_handler(sig, frame):
    global _sigint_count
    _sigint_count += 1
    if _sigint_count >= 2:
        print("\nForce exit.")
        os._exit(1)
    raise KeyboardInterrupt
signal.signal(signal.SIGINT, _sigint_handler)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)
import arx5_interface as arx5

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config_arx5.yaml")


class ImpedanceControlPlotter:
    """
    Real-time plotter for impedance control tuning.
    Uses matplotlib blitting for fast black/white plotting.
    """

    def __init__(self, window_size=500, update_interval=50, enable_plotting=True, save_duration=10.0):
        self.window_size = window_size
        self.update_interval = update_interval
        self.enable_plotting = enable_plotting
        self.iteration_count = 0

        self.time_buffer = deque(maxlen=window_size)
        self.error_buffer = deque(maxlen=window_size)
        self.ref_pos_buffer = deque(maxlen=window_size)
        self.curr_pos_buffer = deque(maxlen=window_size)
        self.ref_ori_buffer = deque(maxlen=window_size)
        self.curr_ori_buffer = deque(maxlen=window_size)
        self.tau_buffer = deque(maxlen=window_size)
        self.dq_raw_buffer = deque(maxlen=window_size)
        self.dq_filt_buffer = deque(maxlen=window_size)

        self.save_duration = save_duration
        self.archive_time = []
        self.archive_error = []
        self.archive_tau = []
        self.archive_dq_raw = []
        self.archive_dq_filt = []
        self.archive_active = True

        self.lines_pos = {}
        self.lines_ori = {}
        self.lines_tau = {}
        self.lines_vel = {}
        self.background = None

        if self.enable_plotting:
            self._setup_plots()

    def _setup_plots(self):
        plt.ion()
        self.fig, self.axes = plt.subplots(2, 2, figsize=(14, 8))
        self.axes = self.axes.flatten()
        self.fig.suptitle('Impedance Control Monitor', fontsize=12)
        self._configure_axes()
        self.fig.canvas.draw()
        self.background = self.fig.canvas.copy_from_bbox(self.fig.bbox)

    def _configure_axes(self):
        ax0 = self.axes[0]
        ax0.set_title('End-Effector Position', fontsize=10)
        ax0.set_xlabel('Time [s]', fontsize=9)
        ax0.set_ylabel('Position [m]', fontsize=9)
        ax0.grid(True, alpha=0.3)
        self.lines_pos['ref_x'], = ax0.plot([], [], 'r--', linewidth=1.5, alpha=0.7, label='Ref X')
        self.lines_pos['ref_y'], = ax0.plot([], [], 'g--', linewidth=1.5, alpha=0.7, label='Ref Y')
        self.lines_pos['ref_z'], = ax0.plot([], [], 'b--', linewidth=1.5, alpha=0.7, label='Ref Z')
        self.lines_pos['cur_x'], = ax0.plot([], [], 'r-', linewidth=1.5, label='Cur X')
        self.lines_pos['cur_y'], = ax0.plot([], [], 'g-', linewidth=1.5, label='Cur Y')
        self.lines_pos['cur_z'], = ax0.plot([], [], 'b-', linewidth=1.5, label='Cur Z')
        ax0.legend(loc='upper right', fontsize=8, ncol=2)

        ax1 = self.axes[1]
        ax1.set_title('Joint Torques', fontsize=10)
        ax1.set_xlabel('Time [s]', fontsize=9)
        ax1.set_ylabel('Torque [Nm]', fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']
        for j in range(6):
            self.lines_tau[f'j{j}'], = ax1.plot([], [], color=colors[j], linewidth=1.5, label=f'J{j+1}')
        ax1.legend(loc='upper right', fontsize=8, ncol=2)

        ax2 = self.axes[2]
        ax2.set_title('Joint Velocities (J1-J6)', fontsize=10)
        ax2.set_xlabel('Time [s]', fontsize=9)
        ax2.set_ylabel('Velocity [rad/s]', fontsize=9)
        ax2.grid(True, alpha=0.3)
        colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']
        for j in range(6):
            self.lines_vel[f'raw_{j}'], = ax2.plot([], [], color=colors[j], linewidth=1, alpha=0.3)
            self.lines_vel[f'filt_{j}'], = ax2.plot([], [], color=colors[j], linewidth=2, label=f'J{j+1}')
        ax2.legend(loc='upper right', fontsize=8, ncol=2)

        ax3 = self.axes[3]
        ax3.set_title('End-Effector Orientation', fontsize=10)
        ax3.set_xlabel('Time [s]', fontsize=9)
        ax3.set_ylabel('Angle [deg]', fontsize=9)
        ax3.grid(True, alpha=0.3)
        self.lines_ori['ref_roll'], = ax3.plot([], [], 'r--', linewidth=1.5, alpha=0.7, label='Ref Roll')
        self.lines_ori['ref_pitch'], = ax3.plot([], [], 'g--', linewidth=1.5, alpha=0.7, label='Ref Pitch')
        self.lines_ori['ref_yaw'], = ax3.plot([], [], 'b--', linewidth=1.5, alpha=0.7, label='Ref Yaw')
        self.lines_ori['cur_roll'], = ax3.plot([], [], 'r-', linewidth=1.5, label='Cur Roll')
        self.lines_ori['cur_pitch'], = ax3.plot([], [], 'g-', linewidth=1.5, label='Cur Pitch')
        self.lines_ori['cur_yaw'], = ax3.plot([], [], 'b-', linewidth=1.5, label='Cur Yaw')
        ax3.legend(loc='upper right', fontsize=8, ncol=2)

    def add_data(self, time_val, error, tau, dq_raw, dq_filtered, ref_pos=None, curr_pos=None, ref_ori=None, curr_ori=None):
        self.time_buffer.append(time_val)
        self.error_buffer.append(error[:3].copy())
        if ref_pos is not None:
            self.ref_pos_buffer.append(ref_pos.copy())
        if curr_pos is not None:
            self.curr_pos_buffer.append(curr_pos.copy())
        if ref_ori is not None:
            self.ref_ori_buffer.append(self._rotation_matrix_to_euler(ref_ori))
        if curr_ori is not None:
            self.curr_ori_buffer.append(self._rotation_matrix_to_euler(curr_ori))
        self.tau_buffer.append(tau.copy())
        self.dq_raw_buffer.append(dq_raw.copy())
        self.dq_filt_buffer.append(dq_filtered.copy())

        if self.archive_active and time_val <= self.save_duration:
            self.archive_time.append(time_val)
            self.archive_error.append(error.copy())
            self.archive_tau.append(tau.copy())
            self.archive_dq_raw.append(dq_raw.copy())
            self.archive_dq_filt.append(dq_filtered.copy())
        elif time_val > self.save_duration and self.archive_active:
            self.archive_active = False
            print(f"\n✓ Archive buffer filled ({self.save_duration}s of data collected)")

        self.iteration_count += 1

    def should_update(self):
        return (self.enable_plotting and
                self.iteration_count % self.update_interval == 0 and
                len(self.time_buffer) > 10)

    def update(self):
        if not self.should_update():
            return

        t_arr = np.array(self.time_buffer)
        ref_pos_arr = np.array(self.ref_pos_buffer) if len(self.ref_pos_buffer) > 0 else None
        curr_pos_arr = np.array(self.curr_pos_buffer) if len(self.curr_pos_buffer) > 0 else None
        tau_arr = np.array(self.tau_buffer)
        dq_raw_arr = np.array(self.dq_raw_buffer)
        dq_filt_arr = np.array(self.dq_filt_buffer)

        self.fig.canvas.restore_region(self.background)

        ref_ori_arr = np.array(self.ref_ori_buffer) if len(self.ref_ori_buffer) > 0 else None
        curr_ori_arr = np.array(self.curr_ori_buffer) if len(self.curr_ori_buffer) > 0 else None

        self._update_position_lines(t_arr, ref_pos_arr, curr_pos_arr)
        self._update_torque_lines(t_arr, tau_arr)
        self._update_velocity_lines(t_arr, dq_raw_arr, dq_filt_arr)
        self._update_orientation_lines(t_arr, ref_ori_arr, curr_ori_arr)

        for ax in self.axes:
            ax.draw_artist(ax.patch)
            for line in ax.lines:
                ax.draw_artist(line)

        self.fig.canvas.blit(self.fig.bbox)
        self.fig.canvas.flush_events()

    def _update_position_lines(self, t, ref_pos, curr_pos):
        if ref_pos is not None and curr_pos is not None:
            self.lines_pos['ref_x'].set_data(t, ref_pos[:, 0])
            self.lines_pos['ref_y'].set_data(t, ref_pos[:, 1])
            self.lines_pos['ref_z'].set_data(t, ref_pos[:, 2])
            self.lines_pos['cur_x'].set_data(t, curr_pos[:, 0])
            self.lines_pos['cur_y'].set_data(t, curr_pos[:, 1])
            self.lines_pos['cur_z'].set_data(t, curr_pos[:, 2])
            self.axes[0].relim()
            self.axes[0].autoscale_view()

    def _update_torque_lines(self, t, tau):
        for j in range(6):
            self.lines_tau[f'j{j}'].set_data(t, tau[:, j])
        self.axes[1].relim()
        self.axes[1].autoscale_view()

    def _update_velocity_lines(self, t, dq_raw, dq_filt):
        for j in range(6):
            self.lines_vel[f'raw_{j}'].set_data(t, dq_raw[:, j])
            self.lines_vel[f'filt_{j}'].set_data(t, dq_filt[:, j])
        self.axes[2].relim()
        self.axes[2].autoscale_view()
        ymin, ymax = self.axes[2].get_ylim()
        if ymax - ymin < 0.2:
            y_center = (ymax + ymin) / 2
            self.axes[2].set_ylim(y_center - 0.1, y_center + 0.1)

    def _update_orientation_lines(self, t, ref_ori, curr_ori):
        if ref_ori is not None and curr_ori is not None:
            self.lines_ori['ref_roll'].set_data(t, ref_ori[:, 0])
            self.lines_ori['ref_pitch'].set_data(t, ref_ori[:, 1])
            self.lines_ori['ref_yaw'].set_data(t, ref_ori[:, 2])
            self.lines_ori['cur_roll'].set_data(t, curr_ori[:, 0])
            self.lines_ori['cur_pitch'].set_data(t, curr_ori[:, 1])
            self.lines_ori['cur_yaw'].set_data(t, curr_ori[:, 2])
            self.axes[3].relim()
            self.axes[3].autoscale_view()

    def _rotation_matrix_to_euler(self, R):
        sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
        singular = sy < 1e-6
        if not singular:
            roll = np.arctan2(R[2, 1], R[2, 2])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = np.arctan2(R[1, 0], R[0, 0])
        else:
            roll = np.arctan2(-R[1, 2], R[1, 1])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = 0
        return np.rad2deg(np.array([roll, pitch, yaw]))

    def save(self, filename=None, metadata=None):
        if filename is None:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            filename = f'impedance_tuning_{timestamp}.npz'

        save_dict = {
            'time': np.array(self.archive_time),
            'error': np.array(self.archive_error),
            'tau': np.array(self.archive_tau),
            'dq_raw': np.array(self.archive_dq_raw),
            'dq_filtered': np.array(self.archive_dq_filt)
        }
        if metadata is not None:
            save_dict.update(metadata)

        np.savez(filename, **save_dict)
        num_samples = len(self.archive_time)
        duration = self.archive_time[-1] if num_samples > 0 else 0.0
        print(f"✓ Data saved to {filename}")
        print(f"  Samples: {num_samples}, Duration: {duration:.2f}s")
        return filename

    def close(self):
        if self.enable_plotting:
            plt.close(self.fig)


class CartesianImpedanceController:
    """
    Cartesian impedance controller using Pinocchio for kinematics/dynamics.
    """
    def __init__(self, model, ee_id, Kp=None, Kd=None, joint_position_min=None, joint_position_max=None, joint_velocity_max=None, joint_torque_max=None, use_operational_space=False):
        self.model = model
        self.data = model.createData()
        self.ee_id = ee_id

        self.joint_position_max = joint_position_max
        self.joint_position_min = joint_position_min
        self.joint_torque_max = joint_torque_max

        self.use_operational_space = use_operational_space

        if Kp is None:
            self.Kp = np.diag([100, 100, 100, 50, 50, 50])
        else:
            self.Kp = Kp
        if Kd is None:
            self.Kd = np.diag([20, 20, 20, 10, 10, 10])
        else:
            self.Kd = Kd

        self.nv = model.nv
        self.dq_filtered = np.zeros(model.nv)

    def pseudo_inverse(self, matrix, damping=1e-6):
        U, s, Vt = np.linalg.svd(matrix, full_matrices=False)
        s_inv = s / (s**2 + damping**2)
        return Vt.T @ np.diag(s_inv) @ U.T

    def filter_velocity(self, dq_measured, alpha=0.1):
        self.dq_filtered = alpha * dq_measured + (1.0 - alpha) * self.dq_filtered
        return self.dq_filtered

    def filter_SE3_pose(self, target_pose, new_target_pose, alpha=0.02):
        def exponential_moving_avg(prev_value, new_value, alpha):
            return alpha * new_value + (1.0 - alpha) * prev_value

        log_target = pin.log6(target_pose)
        log_new = pin.log6(new_target_pose)
        log_filtered = exponential_moving_avg(log_target, log_new, alpha)
        return pin.exp6(log_filtered)

    def step_task(self, q, dq, desired_pose, v_des):
        model, data = self.model, self.data
        ee_id = self.ee_id

        dq_filtered = self.filter_velocity(dq, alpha=0.1)

        pin.forwardKinematics(model, data, q, dq_filtered)
        pin.updateFramePlacements(model, data)
        pin.computeJointJacobians(model, data, q)

        end_effector_pose = data.oMf[ee_id]

        p_err = desired_pose.translation - end_effector_pose.translation
        o_err = pin.log3(desired_pose.rotation @ end_effector_pose.rotation.T)
        error = np.hstack([p_err, o_err])

        # Clamp errors to prevent runaway behavior from large initial errors
        error = np.clip(error,
                        [-0.2, -0.2, -0.2, -0.1, -0.1, -0.1],
                        [0.2, 0.2, 0.2, 0.1, 0.1, 0.1])

        J = pin.getFrameJacobian(model, data, ee_id, pin.ReferenceFrame.WORLD)

        F_task = self.Kp @ error - self.Kd @ (J @ dq_filtered)

        if self.use_operational_space:
            pin.computeMinverse(model, data, q)
            Mx_inv = J @ data.Minv @ J.T
            Mx = self.pseudo_inverse(Mx_inv, damping=1e-4)
            tau = J.T @ Mx @ F_task
        else:
            tau = J.T @ F_task

        return tau, error

    def step_gravity_compensation(self, q, dq=None, damping=None):
        tau_g = pin.computeGeneralizedGravity(self.model, self.data, q)
        if dq is not None and damping is not None:
            tau_g -= damping * dq
        return tau_g

    def saturateTorqueRate(self, tau_d_calculated, tau_d_previous, max_delta_tau):
        difference = tau_d_calculated - tau_d_previous
        difference_limited = np.clip(difference, -max_delta_tau, max_delta_tau)
        return tau_d_previous + difference_limited

    def get_joint_limit_torque(self, joint_positions, safe_range, max_torque):
        dist_to_lower = joint_positions - self.joint_position_min
        dist_to_upper = self.joint_position_max - joint_positions
        near_lower = dist_to_lower < safe_range
        near_upper = dist_to_upper < safe_range
        lower_ratios = np.clip((safe_range - dist_to_lower) / safe_range, 0, 1)
        upper_ratios = np.clip((safe_range - dist_to_upper) / safe_range, 0, 1)
        return (
            np.where(near_lower, max_torque * lower_ratios, 0.0) -
            np.where(near_upper, max_torque * upper_ratios, 0.0)
        )

    def computeCoriolisMatrix(self, q, dq):
        pin.computeAllTerms(self.model, self.data, q, dq)
        C = pin.computeCoriolisMatrix(self.model, self.data, q, dq)
        return C @ dq

    def set_ee_mass(self, new_mass, ee_link_name='eef_link'):
        try:
            ee_frame_id = self.model.getFrameId(ee_link_name)
            parent_joint_id = self.model.frames[ee_frame_id].parent
            current_inertia = self.model.inertias[parent_joint_id]
            mass_ratio = new_mass / current_inertia.mass if current_inertia.mass > 0 else 1.0
            new_inertia = pin.Inertia(
                new_mass,
                current_inertia.lever,
                current_inertia.inertia * mass_ratio
            )
            self.model.inertias[parent_joint_id] = new_inertia
            print(f"✓ Updated EE mass: {new_mass:.3f} kg (link: {ee_link_name})")
            return True
        except Exception as e:
            print(f"✗ Failed to update EE mass: {e}")
            return False

    def get_ee_mass(self, ee_link_name='eef_link'):
        try:
            ee_frame_id = self.model.getFrameId(ee_link_name)
            parent_joint_id = self.model.frames[ee_frame_id].parent
            return self.model.inertias[parent_joint_id].mass
        except Exception as e:
            print(f"✗ Failed to get EE mass: {e}")
            return None


def write_config(robot_config, controller_config):
    config = {
        'robot': {
            'pos_min': robot_config.joint_pos_min.tolist(),
            'pos_max': robot_config.joint_pos_max.tolist(),
            'vel_max': robot_config.joint_vel_max.tolist(),
            'torque_max': robot_config.joint_torque_max.tolist(),
        },
        'controller': {
            'controller_dt': controller_config.controller_dt
        }
    }
    with open(CONFIG_PATH, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"✓ Config written to {CONFIG_PATH}")


def load_config(robot_config=None, controller_config=None):
    """Load config_arx5.yaml, generating it from robot/controller config if missing or empty."""
    needs_generate = not os.path.exists(CONFIG_PATH)

    if not needs_generate:
        with open(CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)
        if config is None or 'robot' not in config:
            needs_generate = True

    if needs_generate:
        if robot_config is None or controller_config is None:
            raise FileNotFoundError(
                f"Config file missing or empty: {CONFIG_PATH}\n"
                "Run in --mode real once to auto-generate it, or create it manually."
            )
        print(f"Config file missing or empty — generating from robot config...")
        write_config(robot_config, controller_config)
        with open(CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)

    return config


def fmt(x):
    return f"{x:.5g}"


def print_vec(label, arr, width=12):
    arr_str = " ".join(f"{fmt(v):>{width}}" for v in arr)
    print(f"{label:<18} {arr_str}")


@click.command()
@click.argument("model")  # ARX arm model: X5 or L5
@click.argument("interface")  # can bus name (can1 etc.)
@click.option("--urdf_path", "-u", default="../models/X5.urdf", help="URDF file path")
@click.option("--offset_x", "-x", default=0.0, help="Desired X offset from initial position (m)")
@click.option("--offset_y", "-y", default=0.0, help="Desired Y offset from initial position (m)")
@click.option("--offset_z", "-z", default=0.05, help="Desired Z offset from initial position (m)")
@click.option("--task_scale", "-s", default=0.3, help="Task torque scaling factor (0.0-1.0). Start at 0.3 for real hardware.")
@click.option("--plot/--no-plot", default=True, help="Enable/disable live plotting for tuning")
@click.option("--mode", default="sim", help="Select sim or real")
@click.option("--scene_path", default=None, help="Path to MuJoCo scene XML file (sim mode only)")
@click.option("--operational-space/--joint-space", default=False, help="Use operational space control (configuration-independent)")
def main(model: str, interface: str, urdf_path: str, offset_x: float, offset_y: float, offset_z: float, task_scale: float, plot: bool, mode: str, scene_path: str, operational_space: bool):

    joint_controller = None

    if mode == "sim":
        if scene_path is None:
            raise click.UsageError("--scene_path is required in sim mode. Provide the path to your MuJoCo scene XML.")

        if not os.path.exists(scene_path):
            print(f"Error: Scene file not found at {scene_path}")
            return

        print(f"Loading MuJoCo model from: {scene_path}")
        mujoco_model = mujoco.MjModel.from_xml_path(scene_path)
        mujoco_data = mujoco.MjData(mujoco_model)

        joint_torques = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        home_position = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        mujoco_data.qpos[:6] = home_position
        print(f"Initialized joint positions: {home_position}")
        mujoco.mj_forward(mujoco_model, mujoco_data)

        print(f"\nControl mode: Torque control")
        print(f"Joint torque commands (Nm): {joint_torques}")
        print(f"Number of actuators: {mujoco_model.nu}")

        step_count = 0
        display_interval = 100

        q_init = np.array([0.002, -0.009, -0.002, -0.067, -0.003, -0.001])
        ee_frame_name = 'eef_link'

    elif mode == "real":
        try:
            robot_config = arx5.RobotConfigFactory.get_instance().get_config(model)
            controller_config = arx5.ControllerConfigFactory.get_instance().get_config(
                "joint_controller", robot_config.joint_dof
            )

            controller_config.gravity_compensation = False
            controller_config.background_send_recv = True

            joint_controller = arx5.Arx5JointController(
                robot_config, controller_config, interface
            )

            joint_controller.set_log_level(arx5.LogLevel.INFO)
            np.set_printoptions(precision=3, suppress=True)

            print("\n[1/5] Moving to home position with position control...")
            joint_controller.reset_to_home()
            time.sleep(1.0)

            joint_state = joint_controller.get_joint_state()
            q_init = joint_state.pos().copy()

            print("\n[2/5] Loading Pinocchio model for gravity compensation...")
            ee_frame_name = robot_config.eef_link_name
        except (KeyboardInterrupt, Exception) as e:
            if joint_controller is not None:
                del joint_controller
            if isinstance(e, KeyboardInterrupt):
                print("\nInterrupted during initialization. Exiting.")
                return
            raise

    else:
        print("ERROR: --mode must be 'sim' or 'real'")
        return

    # Load config (auto-generate from robot_config if missing and in real mode)
    if mode == "real":
        config = load_config(robot_config, controller_config)
    else:
        config = load_config()

    config_pose_min = np.array(config['robot']['pos_min'])
    config_pose_max = np.array(config['robot']['pos_max'])
    config_vel_max = np.array(config['robot']['vel_max'])
    config_torque_max = np.array(config['robot']['torque_max'])
    config_controller_dt = config['controller']['controller_dt']

    if mode == "real":
        pin_model = pin.buildModelFromUrdf(urdf_path)
        pin_data = pin_model.createData()
        ee_id = pin_model.getFrameId(ee_frame_name)
        print(f"✓ Loaded URDF: {urdf_path}")
        print(f"✓ End-effector frame: {ee_frame_name} (id: {ee_id})")

        print("\n[3/5] Initializing impedance controller...")
        impedance_controller_temp = CartesianImpedanceController(
            pin_model, ee_id, None, None,
            joint_position_min=config_pose_min,
            joint_position_max=config_pose_max,
            joint_velocity_max=config_vel_max,
            joint_torque_max=config_torque_max,
            use_operational_space=operational_space
        )

        print("\n[4/5] Transitioning to torque control with gravity compensation...")
        print("Setting gains to zero - starting gravity compensation immediately")

        for kd_val in [2.0, 1.5, 1.0, 0.0]:
            gain = joint_controller.get_gain()
            gain.kp()[:] = 0.0
            gain.kd()[:] = kd_val
            gain.gripper_kp = 0.0
            gain.gripper_kd = 0.0
            joint_controller.set_gain(gain)
            time.sleep(0.1)

        cmd = arx5.JointState(robot_config.joint_dof)
        print("✓ Sending gravity compensation torques (pre-stabilization)...")
        for i in range(50):
            joint_state = joint_controller.get_joint_state()
            q = joint_state.pos().copy()
            tau_gravity = impedance_controller_temp.step_gravity_compensation(q)
            cmd.torque()[:] = tau_gravity
            joint_controller.set_joint_cmd(cmd)
            time.sleep(controller_config.controller_dt)
            if i % 10 == 0:
                print(f"  Step {i}/50: max gravity torque = {np.max(np.abs(tau_gravity)):.2f} Nm")

        print("✓ Pre-stabilization complete - robot is supported by gravity compensation")

    else:
        print("\n[2/4] Loading Pinocchio model...")
        pin_model = pin.buildModelFromUrdf(urdf_path)
        pin_data = pin_model.createData()
        ee_id = pin_model.getFrameId(ee_frame_name)
        print(f"✓ Loaded URDF: {urdf_path}")
        print(f"✓ End-effector frame: {ee_frame_name} (id: {ee_id})")

    print(f"\n[{'5/5' if mode == 'real' else '3/4'}] Computing desired end-effector pose...")
    pin.forwardKinematics(pin_model, pin_data, q_init)
    pin.updateFramePlacements(pin_model, pin_data)
    x_init = pin_data.oMf[ee_id].copy()

    new_target_translation = np.array([
        x_init.translation[0] + offset_x,
        x_init.translation[1] + offset_y,
        x_init.translation[2] + offset_z
    ])
    new_target_rotation = x_init.rotation.copy()
    new_target_pose = pin.SE3(new_target_rotation, new_target_translation)
    v_des = pin.Motion.Zero()

    print(f"✓ Initial position: {x_init.translation}")
    print(f"✓ Desired position: {new_target_pose.translation}")
    print(f"✓ Offset: [{offset_x}, {offset_y}, {offset_z}] m")

    print("\n[Safety] Configuring safety thresholds...")
    task_scale = np.clip(task_scale, 0.0, 1.0)
    if task_scale > 0.0:
        print(f"\nWARNING: Task torques are ENABLED at {task_scale*100:.0f}%")

    print(f"\n[Visualization] {'Enabling' if plot else 'Disabling'} live plotting...")
    plotter = ImpedanceControlPlotter(
        window_size=500,
        update_interval=50,
        enable_plotting=plot
    )

    print("\n" + "=" * 60)
    print("Starting impedance control loop...")
    print("Press Ctrl+C to stop and reset to home")
    print("=" * 60 + "\n")

    tau_d_previous = np.zeros(6)
    tau_friction = np.zeros(6)
    tau_wrench = np.zeros(6)
    tau_nullspace = np.zeros(6)

    loop_count = 0
    plotting_start_time = time.time()
    previous_loop_time = time.time()
    tuning_metadata = {}

    try:
        if mode == "sim":
            print("\n[3/4] Configuring impedance controller...")

            Kp = np.diag([12, 10, 10, 3, 3, 3])
            Kd = np.diag([6, 20, 6, 2, 2, 2])

            impedance_controller = CartesianImpedanceController(
                pin_model, ee_id, Kp, Kd,
                joint_position_min=config_pose_min,
                joint_position_max=config_pose_max,
                joint_velocity_max=config_vel_max,
                joint_torque_max=config_torque_max,
                use_operational_space=operational_space
            )
            print(f"✓ Control mode: {'Operational Space' if operational_space else 'Joint Space'}")
            print(f"✓ Position stiffness (xyz): {np.diag(Kp)[:3]} N/m")
            print(f"✓ Orientation stiffness (xyz): {np.diag(Kp)[3:]} Nm/rad")
            print(f"✓ Position damping (xyz): {np.diag(Kd)[:3]} Ns/m")
            print(f"✓ Orientation damping (xyz): {np.diag(Kd)[3:]} Nms/rad")

            tuning_metadata = {
                'Kp': Kp,
                'Kd': Kd,
                'task_scale': task_scale,
                'offset_x': offset_x,
                'offset_y': offset_y,
                'offset_z': offset_z,
                'controller_dt': config_controller_dt,
                'use_operational_space': operational_space
            }

            MUJOCO_SUBSTEPS = 5

            with mujoco.viewer.launch_passive(mujoco_model, mujoco_data) as viewer:
                while viewer.is_running():
                    start_time = time.perf_counter()

                    loop_start_time = time.time()
                    cycle_time = loop_start_time - previous_loop_time
                    previous_loop_time = loop_start_time

                    q = mujoco_data.qpos[:6].copy()
                    dq = mujoco_data.qvel[:6].copy()

                    tau_task, error = impedance_controller.step_task(q, dq, new_target_pose, v_des)

                    error_position = np.linalg.norm(error[:3])
                    error_orientation = np.linalg.norm(error[3:])

                    tau_task_clipped = np.clip(tau_task, -config_torque_max, config_torque_max)
                    tau_task_scaled = task_scale * tau_task_clipped
                    tau_task_magnitude = np.linalg.norm(tau_task_scaled)
                    max_task_torque_joint = np.max(np.abs(tau_task_scaled))

                    tau_joint_limits = np.zeros(6)
                    tau_coriolis = impedance_controller.computeCoriolisMatrix(q, dq)
                    tau_gravity = impedance_controller.step_gravity_compensation(q)

                    tau_d = tau_task_scaled + tau_gravity + tau_coriolis
                    print_vec("tau_task:", tau_task)
                    print_vec("tau_task_clipped:", tau_task_clipped)
                    print_vec("tau_task_scaled:", tau_task_scaled)
                    print_vec("tau_coriolis:", tau_coriolis)
                    print_vec("tau_gravity:", tau_gravity)
                    print_vec("tau_d:", tau_d)

                    tau_d = impedance_controller.saturateTorqueRate(tau_d, tau_d_previous, max_delta_tau=0.5)

                    pin.forwardKinematics(pin_model, pin_data, q)
                    pin.updateFramePlacements(pin_model, pin_data)
                    current_ee_pose = pin_data.oMf[ee_id]

                    elapsed = time.time() - plotting_start_time
                    plotter.add_data(elapsed, error, tau_d, dq, impedance_controller.dq_filtered,
                                     ref_pos=new_target_pose.translation, curr_pos=current_ee_pose.translation,
                                     ref_ori=new_target_pose.rotation, curr_ori=current_ee_pose.rotation)
                    plotter.update()

                    mujoco_data.ctrl[0:6] = tau_d
                    tau_d_previous = tau_d

                    for substep in range(MUJOCO_SUBSTEPS):
                        mujoco.mj_step(mujoco_model, mujoco_data)

                    viewer.sync()

                    loop_count += 1
                    if loop_count % display_interval == 0:
                        max_torque = np.max(np.abs(tau_d))
                        pin.forwardKinematics(pin_model, pin_data, q)
                        pin.updateFramePlacements(pin_model, pin_data)
                        current_ee_pose = pin_data.oMf[ee_id]

                        if task_scale > 0.0:
                            print(f"[{elapsed:6.1f}s] Cycle: {cycle_time*1000:5.2f}ms | "
                                  f"Pos err: {error_position*1000:.1f}mm, Ori err: {error_orientation*1000:.1f}mrad | "
                                  f"Task tau: {max_task_torque_joint:.2f}Nm | Total tau: {max_torque:.2f}Nm")
                        else:
                            print(f"[{elapsed:6.1f}s] Cycle: {cycle_time*1000:5.2f}ms | "
                                  f"Joints: {np.round(q, 3)} | "
                                  f"Velocities: {np.round(dq, 3)} | "
                                  f"Max torque: {max_torque:5.2f}Nm")

                    elapsed_loop = time.perf_counter() - start_time
                    sleep_time = config_controller_dt - elapsed_loop
                    if sleep_time > 0:
                        time.sleep(sleep_time)

        if mode == "real":
            print("\n[5/5] Configuring impedance controller...")

            if operational_space:
                Kp = np.diag([5, 5, 20, 6, 6, 6])
                Kd = np.diag([30, 30, 40, 4, 10, 4])
            else:
                Kp = np.diag([70, 70, 105, 0, 2, 0])
                Kd = np.diag([30, 30, 46, 0, 0, 0])

            impedance_controller = CartesianImpedanceController(
                pin_model, ee_id, Kp, Kd,
                joint_position_min=config_pose_min,
                joint_position_max=config_pose_max,
                joint_velocity_max=config_vel_max,
                joint_torque_max=config_torque_max,
                use_operational_space=operational_space
            )
            print(f"✓ Control mode: {'Operational Space' if operational_space else 'Joint Space'}")
            print(f"✓ Position stiffness (xyz): {np.diag(Kp)[:3]} N/m")
            print(f"✓ Orientation stiffness (xyz): {np.diag(Kp)[3:]} Nm/rad")
            print(f"✓ Position damping (xyz): {np.diag(Kd)[:3]} Ns/m")
            print(f"✓ Orientation damping (xyz): {np.diag(Kd)[3:]} Nms/rad")

            tuning_metadata = {
                'Kp': Kp,
                'Kd': Kd,
                'task_scale': task_scale,
                'offset_x': offset_x,
                'offset_y': offset_y,
                'offset_z': offset_z,
                'controller_dt': controller_config.controller_dt,
                'use_operational_space': operational_space
            }

            while True:
                loop_start_time = time.time()
                cycle_time = loop_start_time - previous_loop_time
                previous_loop_time = loop_start_time

                joint_state = joint_controller.get_joint_state()
                q = joint_state.pos().copy()
                dq = joint_state.vel().copy()

                if np.any(q < config_pose_min) or np.any(q > config_pose_max):
                    print(f"WARNING: Joint position out of bounds!   {q}")
                    cmd.torque()[:] = 0.0
                    joint_controller.set_joint_cmd(cmd)
                    break

                if np.any(np.abs(dq) > config_vel_max):
                    print(f"WARNING: Joint velocity too high!   {dq}")
                    cmd.torque()[:] = 0.0
                    joint_controller.set_joint_cmd(cmd)
                    break

                tau_task, error = impedance_controller.step_task(q, dq, new_target_pose, v_des)

                error_position = np.linalg.norm(error[:3])
                error_orientation = np.linalg.norm(error[3:])

                tau_task_clipped = np.clip(tau_task, -config_torque_max, config_torque_max)
                tau_task_scaled = task_scale * tau_task_clipped
                tau_task_magnitude = np.linalg.norm(tau_task_scaled)
                max_task_torque_joint = np.max(np.abs(tau_task_scaled))

                tau_joint_limits = np.zeros(6)
                tau_coriolis = np.zeros(6)
                tau_gravity = impedance_controller.step_gravity_compensation(q)

                tau_d = tau_task_scaled + tau_gravity + tau_coriolis
                print_vec("tau_task:", tau_task)
                print_vec("tau_task_clipped:", tau_task_clipped)
                print_vec("tau_task_scaled:", tau_task_scaled)
                print_vec("tau_coriolis:", tau_coriolis)
                print_vec("tau_gravity:", tau_gravity)
                print_vec("tau_d:", tau_d)

                tau_d = impedance_controller.saturateTorqueRate(tau_d, tau_d_previous, max_delta_tau=0.5)
                print_vec("tau_d_limited:", tau_d)
                print_vec("error:", error)

                pin.forwardKinematics(pin_model, pin_data, q)
                pin.updateFramePlacements(pin_model, pin_data)
                current_ee_pose = pin_data.oMf[ee_id]

                elapsed = time.time() - plotting_start_time
                plotter.add_data(elapsed, error, tau_d, dq, impedance_controller.dq_filtered,
                                 ref_pos=new_target_pose.translation, curr_pos=current_ee_pose.translation,
                                 ref_ori=new_target_pose.rotation, curr_ori=current_ee_pose.rotation)
                plotter.update()

                cmd.torque()[:] = tau_d
                gripper_assist_gain = 0.7
                cmd.gripper_torque = gripper_assist_gain * joint_state.gripper_vel
                joint_controller.set_joint_cmd(cmd)
                tau_d_previous = tau_d

                time.sleep(controller_config.controller_dt)

                loop_count += 1
                if loop_count % int(1.0 / controller_config.controller_dt) == 0:
                    elapsed = time.time() - plotting_start_time
                    max_torque = np.max(np.abs(tau_d))
                    pin.forwardKinematics(pin_model, pin_data, q)
                    pin.updateFramePlacements(pin_model, pin_data)
                    current_ee_pose = pin_data.oMf[ee_id]
                    if task_scale > 0.0:
                        print(f"[{elapsed:6.1f}s] Cycle: {cycle_time*1000:5.2f}ms | "
                              f"Pos err: {error_position*1000:.1f}mm, Ori err: {error_orientation*1000:.1f}mrad | "
                              f"Task tau: {max_task_torque_joint:.2f}Nm | Total tau: {max_torque:.2f}Nm")
                    else:
                        print(f"[{elapsed:6.1f}s] Cycle: {cycle_time*1000:5.2f}ms | "
                              f"Joints: {np.round(q, 3)} | "
                              f"Velocities: {np.round(dq, 3)} | "
                              f"Max torque: {max_torque:5.2f}Nm")

    except KeyboardInterrupt:
        print("\n" + "=" * 60)
        print("Keyboard interrupt detected!")
        print("Saving data and cleaning up...")
        print("=" * 60)

        if tuning_metadata:
            plotter.save(metadata=tuning_metadata)

        plotter.close()

        print("Resetting to home position...")
        if mode == "real" and joint_controller is not None:
            gain = joint_controller.get_gain()
            gain.kp()[:] = 0.0
            gain.kd()[:] = 2.0
            joint_controller.set_gain(gain)
            time.sleep(0.1)
            joint_controller.reset_to_home()
        time.sleep(1.0)
        print("✓ Reset complete. Exiting.")
    finally:
        if joint_controller is not None:
            del joint_controller


if __name__ == "__main__":
    main()
