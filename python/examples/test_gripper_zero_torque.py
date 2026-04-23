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
@click.argument("model")  # ARX arm model: X5 or L5
@click.argument("interface")  # can bus name (can0 etc.)
@click.option("--urdf_path", "-u", default="../models/arx5.urdf", help="URDF file path")
def main(model: str, interface: str, urdf_path: str):
    robot_config = arx5.RobotConfigFactory.get_instance().get_config(model)
    controller_config = arx5.ControllerConfigFactory.get_instance().get_config(
        "joint_controller", robot_config.joint_dof
    )
    controller_config.background_send_recv = True

    arx5_joint_controller = arx5.Arx5JointController(
        robot_config, controller_config, interface
    )

    np.set_printoptions(precision=3, suppress=True)
    arx5_joint_controller.set_log_level(arx5.LogLevel.INFO)

    # Zero ONLY gripper gains so the gripper is freely backdrivable.
    # Arm gains are left at factory defaults so the arm holds its current pose.
    gain = arx5_joint_controller.get_gain()
    gain.gripper_kp = 0.0
    gain.gripper_kd = 0.0
    arx5_joint_controller.set_gain(gain)

    # Snapshot the current arm pose so we can keep commanding it (hold-in-place).
    # We intentionally do NOT call reset_to_home() — homing with stiff gripper
    # gains has been observed to trip the over-current fault on this arm.
    hold_pose = arx5_joint_controller.get_joint_state().pos().copy()

    print("Gripper is in zero-torque mode. Move it by hand.")
    print("Arm is holding its current pose. Press Ctrl-C to exit.")

    dt = controller_config.controller_dt
    tick = 0
    try:
        while True:
            cmd = arx5.JointState(robot_config.joint_dof)
            cmd.pos()[:] = hold_pose
            cmd.gripper_torque = 0.0
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
