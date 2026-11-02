from dataclasses import dataclass

from lerobot.robots.config import RobotConfig

from lerobot_robot_trossen.config_widowxai_follower import WidowXAIFollowerConfig


@RobotConfig.register_subclass("widowxai_follower_robot_surge")
@dataclass
class WidowXAIFollowerSurgeConfig(WidowXAIFollowerConfig):
    """
    Config for the WidowX AI Solo follower + Alt-Bionics Surge hand.

    Inherits EVERY field of the plain WidowXAIFollowerConfig (ip_address,
    max_relative_target, joint_names, staged_positions, cameras, the include_*
    flags, ...), so existing `--robot.<field>=...` CLI options and launch.json
    entries carry over UNCHANGED. The only additions are the two hand-hardware
    fields below.

    The physical gripper is removed and its motor unplugged; the Surge hand is
    mounted in its place. The inner Trossen follower is still driven as a 7-DOF
    arm (the driver expects 7 values), but the gripper slot is pinned to
    gripper_pin_pos every step, and the gripper is omitted from the dataset. So
    the recorded state/action is 6 arm joints + 6 hand joints = 12 dims.

    Reverting to the plain gripper follower: select the plain type
    (`--robot.type=widowxai_follower_robot`) instead of this one.

    Note: the Manus mode the hand is driven in is NOT a field here. It is bound
    to the IK regression coefficients in altbionics.ik.IK_REGRESSION_MANUS_MODE
    and asserted by the robot on connect, so the mode always travels with the
    coefficients it was fit against.
    """

    # FTDI serial string of the Surge hand, used to pin THIS hand during
    # autodetect (altbionics.hand.autodetect_port) even when the Manus glove
    # dongle is also an FTDI /dev/ttyUSB device. Matches the C++ driver default.
    hand_serial: str = "FTBC5EKG"

    # Value pinned into the gripper slot (7th joint) of the inner arm's command
    # every step. The gripper motor is unplugged, so this is inert; 0.0 = closed
    # and in range. Change only if a different safe constant is ever needed.
    gripper_pin_pos: float = 0.0
