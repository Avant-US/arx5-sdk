"""Soft spring-open gripper test.

Gripper rests at its full-open width. A human can squeeze it closed by hand;
when released, the low-stiffness PD controller gently pulls it back to open.
The arm holds its current pose (no reset_to_home).

Default gripper_kp/kd are conservative starting points — tune via CLI flags if
the spring feels too soft or too stiff on your particular arm.
"""

import time

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)
import arx5_interface as arx5
import click
import numpy as np


@click.command()
@click.argument("interface")  # can bus name (can0 etc.)
@click.option("--gripper_kp", default=0.07, help="Gripper position gain (soft spring stiffness)")
@click.option("--gripper_kd", default=0.01, help="Gripper damping gain")
@click.option(
    "--open_torque",
    default=0.2,
    help="Constant feedforward torque biasing the gripper open (overcomes stiction)",
)
@click.option(
    "--target_width",
    default=0.04,
    type=float,
    help="Gripper rest target in meters (clipped to [0, robot_config.gripper_width])",
)
def main(
    interface: str,
    gripper_kp: float,
    gripper_kd: float,
    open_torque: float,
    target_width: float,
):
    robot_config = arx5.RobotConfigFactory.get_instance().get_config("X5")
    controller_config = arx5.ControllerConfigFactory.get_instance().get_config(
        "joint_controller", robot_config.joint_dof
    )
    controller_config.background_send_recv = True

    arx5_joint_controller = arx5.Arx5JointController(
        robot_config, controller_config, interface
    )

    np.set_printoptions(precision=3, suppress=True)
    arx5_joint_controller.set_log_level(arx5.LogLevel.INFO)

    # Soft gripper PD: strong enough to return to open, weak enough for a human
    # to press closed with mild force. Arm gains are left at factory defaults.
    gain = arx5_joint_controller.get_gain()
    gain.gripper_kp = gripper_kp
    gain.gripper_kd = gripper_kd
    arx5_joint_controller.set_gain(gain)

    # Hold current arm pose — skip reset_to_home (homing with non-zero gripper
    # gains has been observed to trip the over-current fault on this arm).
    hold_pose = arx5_joint_controller.get_joint_state().pos().copy()
    full_open = robot_config.gripper_width
    target = float(np.clip(target_width, 0.0, full_open))

    print(
        f"Spring-open gripper: kp={gripper_kp}, kd={gripper_kd}, "
        f"open_torque={open_torque}, target={target:.3f} (max={full_open:.3f})"
    )
    print("Press the jaws closed by hand — they return to open on release.")
    print("Arm is holding its current pose. Press Ctrl-C to exit.")

    dt = controller_config.controller_dt
    tick = 0
    deadband = 0.003  # m — kill feedforward chatter near target
    try:
        while True:
            js = arx5_joint_controller.get_joint_state()
            # Bidirectional feedforward: push toward target in whichever
            # direction is needed, with a small deadband to avoid chatter.
            # NOTE: below-target case actively closes the gripper — keep
            # fingers clear while tuning.
            error = target - js.gripper_pos
            if abs(error) < deadband:
                ff_torque = 0.0
            elif error > 0:
                ff_torque = open_torque
            else:
                ff_torque = -open_torque

            cmd = arx5.JointState(robot_config.joint_dof)
            cmd.pos()[:] = hold_pose
            cmd.gripper_pos = target
            cmd.gripper_torque = ff_torque
            arx5_joint_controller.set_joint_cmd(cmd)

            if tick % 10 == 0:
                print(
                    f"gripper pos: {js.gripper_pos:+.3f}, "
                    f"vel: {js.gripper_vel:+.3f}, "
                    f"torque: {js.gripper_torque:+.3f}, "
                    f"ff: {ff_torque:+.3f}"
                )
            tick += 1
            time.sleep(dt)
    except KeyboardInterrupt:
        print("\nExiting (skipping reset_to_home).")
        # background_send_recv spawns a C++ thread that can keep the process
        # alive after Ctrl-C; force exit so the shell returns promptly.
        os._exit(0)


if __name__ == "__main__":
    main()
