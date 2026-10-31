"""
trossen_surge_common — shared, hardware-free glue for the Trossen WidowX-AI-Solo
+ Alt-Bionics Surge hand lerobot integration.

Imported by both composite plugin packages:
  - lerobot_robot_trossen_surge        (composite follower: WidowXAIFollower + SurgeHand)
  - lerobot_teleoperator_trossen_surge (composite leader:   WidowXAILeaderTeleop + ManusGlove)

It holds the conventions those two packages must agree on — hand joint ordering,
the dataset feature keys, the deg<->rad unit convention, and the gripper-pin /
action-split logic — in exactly one place so they can never drift apart.

This package is deliberately NOT named lerobot_robot_* / lerobot_teleoperator_*,
so lerobot's register_third_party_plugins() does not try to load it as a plugin:
it is a library, not a plugin.
"""

from .surge_dataset import (
    HAND_JOINT_CODES,
    HAND_FEATURE_KEYS,
    GRIPPER_JOINT_NAME,
    GRIPPER_FEATURE_KEY,
    DEFAULT_GRIPPER_PIN,
    deg_to_rad,
    rad_to_deg,
    hand_angles_to_features,
    features_to_hand_angles,
    inject_gripper,
    strip_gripper,
    split_combined_action,
)

__all__ = [
    "HAND_JOINT_CODES",
    "HAND_FEATURE_KEYS",
    "GRIPPER_JOINT_NAME",
    "GRIPPER_FEATURE_KEY",
    "DEFAULT_GRIPPER_PIN",
    "deg_to_rad",
    "rad_to_deg",
    "hand_angles_to_features",
    "features_to_hand_angles",
    "inject_gripper",
    "strip_gripper",
    "split_combined_action",
]

__version__ = "0.1.0"
