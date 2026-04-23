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
def main(interface: str, gripper_kp: float, gripper_kd: float, open_torque: float):
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
    open_width = robot_config.gripper_width

    print(
        f"Spring-open gripper: kp={gripper_kp}, kd={gripper_kd}, "
        f"open_torque={open_torque}, open_width={open_width:.3f}"
    )
    print("Press the jaws closed by hand — they return to open on release.")
    print("Arm is holding its current pose. Press Ctrl-C to exit.")

    dt = controller_config.controller_dt
    tick = 0
    try:
        while True:
            cmd = arx5.JointState(robot_config.joint_dof)
            cmd.pos()[:] = hold_pose
            cmd.gripper_pos = open_width
            cmd.gripper_torque = open_torque
            arx5_joint_controller.set_joint_cmd(cmd)

            if tick % 10 == 0:
                js = arx5_joint_controller.get_joint_state()
                print(
                    f"gripper pos: {js.gripper_pos:+.3f}, "
                    f"vel: {js.gripper_vel:+.3f}, "
                    f"torque: {js.gripper_torque:+.3f}"
                )
            tick += 1
            time.sleep(dt)
    except KeyboardInterrupt:
        print("\nExiting (skipping reset_to_home).")


if __name__ == "__main__":
    main()
