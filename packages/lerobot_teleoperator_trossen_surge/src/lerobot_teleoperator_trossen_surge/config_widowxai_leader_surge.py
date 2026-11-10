from dataclasses import dataclass

from altbionics import glove
from lerobot.teleoperators.config import TeleoperatorConfig

from lerobot_teleoperator_trossen.config_widowxai_leader import (
    WidowXAILeaderTeleopConfig,
)


@TeleoperatorConfig.register_subclass("widowxai_leader_teleop_surge")
@dataclass
class WidowXAILeaderTeleopSurgeConfig(WidowXAILeaderTeleopConfig):
    """
    Config for the WidowX AI Solo leader + Alt-Bionics Surge hand (Manus glove).

    Inherits EVERY field of the plain WidowXAILeaderTeleopConfig (ip_address,
    joint_names, staged_positions, ...), so existing `--teleop.<field>=...` CLI
    options and launch.json entries carry over UNCHANGED. The only addition is
    the Manus glove .so path below.

    Reverting to the plain gripper leader is done by selecting the plain type
    (`--teleop.type=widowxai_leader_teleop`) instead of this one — there is no
    "hand disabled" half-state here; this type always means arm + glove.

    Note: the Manus mode the hand is driven in is NOT a field here. It is bound
    to the IK regression coefficients in altbionics.ik.IK_REGRESSION_MANUS_MODE,
    so the mode always travels with the coefficients it was fit against.
    """

    # Path to the Manus Core SDK shared library (libManusSDK_Integrated.so) that
    # altbionics.glove.ManusGlove loads via ctypes. Defaults to the altbionics
    # default; override on the CLI / launch.json for a different install path.
    manus_so_path: str = glove.DEFAULT_SO


    # Enable in-situ hand-IK calibration: keyboard trims + Enter-to-capture
    # (glove xyz -> measured hand angle) during teleop. Off = normal teleop. -anr
    calibrate_hand: bool = False
