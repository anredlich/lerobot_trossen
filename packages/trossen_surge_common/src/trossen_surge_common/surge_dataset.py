"""
surge_dataset — the small, pure, hardware-free glue between the Surge hand's
joint values and the lerobot dataset / policy representation.

Both composite wrappers (the WidowX-AI-Solo + Surge leader teleoperator and
follower robot) share this module, so the conventions live in exactly one place.

What it does
------------
1. Units.  The hand's joint values (from ik.solve() and from AT+GJA?) are treated
   as DEGREES, and the dataset/policy use RADIANS (to match the Trossen arm's
   radian joints and keep the whole state/action vector on one scale). The
   conversion is a flat deg<->rad:

       rad = deg * pi / 180        deg = rad * 180 / pi

   It is exactly reversible, and makes no per-joint assumption.

   Caveat (temporary): in Manus mode 1 the SET/action values the hand accepts are
   NOT true degrees (the thumb is off by roughly 2x); we treat them as degrees
   anyway. The GET/state values ARE true degrees in either mode. This caveat goes
   away when the IK regression is redone in Manus mode 0 (see
   altbionics.ik.IK_REGRESSION_MANUS_MODE), after which SET values are true
   degrees too — with no change to this module.

2. Key mapping.  ik.solve() returns {"X","T","I","M","R","P"} -> value. The
   dataset uses lerobot's "<name>.pos" convention with a "hand_" prefix:

       X -> hand_X.pos    T -> hand_T.pos    I -> hand_I.pos
       M -> hand_M.pos    R -> hand_R.pos    P -> hand_P.pos

   The order is HAND_JOINT_CODES = ("X","T","I","M","R","P"), matching
   altbionics.ik.JOINT_CODES.

3. Gripper pin.  The physical gripper is removed and its motor unplugged, but the
   inner Trossen driver still expects a value for the 7th joint
   ("left_carriage_joint"). We pin it to a constant on the way DOWN to the driver
   and strip it on the way UP from the observation, so the dataset/policy only
   ever see 6 arm + 6 hand = 12 dims. Default pin is 0.0 (closed; in range; inert
   with the motor unplugged — verified safe on hardware).
"""

from __future__ import annotations

import math

# Hand joint codes, in the canonical order (matches altbionics.ik.JOINT_CODES).
HAND_JOINT_CODES: tuple[str, ...] = ("X", "T", "I", "M", "R", "P")

# Dataset feature keys for the 6 hand joints, in HAND_JOINT_CODES order.
HAND_FEATURE_KEYS: tuple[str, ...] = tuple(f"hand_{c}.pos" for c in HAND_JOINT_CODES)

# The Trossen follower/leader bundle the gripper as the 7th joint. This is its
# joint name in the inner Trossen config's joint_names list.
GRIPPER_JOINT_NAME: str = "left_carriage_joint"
GRIPPER_FEATURE_KEY: str = f"{GRIPPER_JOINT_NAME}.pos"

# Default value pinned into the gripper slot of the inner arm's 7-value command.
# 0.0 = closed, in range, inert with the motor unplugged.
DEFAULT_GRIPPER_PIN: float = 0.0

_DEG2RAD = math.pi / 180.0
_RAD2DEG = 180.0 / math.pi


# --- units ------------------------------------------------------------------

def deg_to_rad(deg: float) -> float:
    """Flat degrees -> radians. Exactly reversible by rad_to_deg."""
    return deg * _DEG2RAD


def rad_to_deg(rad: float) -> float:
    """Flat radians -> degrees. Exactly reversible by deg_to_rad."""
    return rad * _RAD2DEG


# --- hand: ik-codes <-> dataset keys (with unit conversion) -----------------

def hand_angles_to_features(angles_deg: dict[str, float]) -> dict[str, float]:
    """
    Convert ik.solve()'s {code: degrees} dict into the dataset's
    {hand_<code>.pos: radians} dict.

    Parameters
    ----------
    angles_deg : dict mapping each of "X","T","I","M","R","P" to a value in
                 degrees (as returned by altbionics.ik.solve or AT+GJA?).

    Returns
    -------
    dict mapping "hand_X.pos".."hand_P.pos" to radians.
    """
    return {
        f"hand_{code}.pos": deg_to_rad(angles_deg[code])
        for code in HAND_JOINT_CODES
    }


def features_to_hand_angles(features: dict[str, float]) -> dict[str, float]:
    """
    Inverse of hand_angles_to_features: convert a dataset/policy
    {hand_<code>.pos: radians} dict back to {code: degrees} for driving the hand.

    Only the hand_*.pos keys are read; any other keys (arm joints, cameras) are
    ignored, so this is safe to call on a full combined action dict.

    Returns
    -------
    dict mapping "X".."P" to degrees, ready for altbionics.hand.set_fingers().
    """
    return {
        code: rad_to_deg(features[f"hand_{code}.pos"])
        for code in HAND_JOINT_CODES
    }


# --- gripper pin: inject on the way down, strip on the way up ---------------

def inject_gripper(arm_features: dict[str, float],
                   arm_joint_names: list[str],
                   pin: float = DEFAULT_GRIPPER_PIN) -> dict[str, float]:
    """
    Given the 6 arm-joint features (joint_0.pos..joint_5.pos) that the dataset
    carries, return the full 7-key dict the inner Trossen driver expects, with
    the gripper slot pinned to `pin`.

    `arm_joint_names` is the inner Trossen config's joint_names (7 entries, the
    last being the gripper). Only the gripper key is added; the 6 arm keys are
    passed through unchanged. Non-arm keys in arm_features (e.g. hand_*.pos) are
    NOT included — pass only the arm portion, or this still copies them through;
    see split_combined_action for the clean split.
    """
    out = dict(arm_features)
    out[f"{arm_joint_names[-1]}.pos"] = pin
    return out


def strip_gripper(obs_features: dict[str, float],
                  gripper_key: str = GRIPPER_FEATURE_KEY) -> dict[str, float]:
    """
    Remove the gripper feature from an observation dict, so the dataset/policy
    never see it. Returns a new dict; input is not mutated. Missing key is fine.
    """
    return {k: v for k, v in obs_features.items() if k != gripper_key}


# --- combined action split (the one the composite follower uses) ------------

def split_combined_action(action: dict[str, float],
                          arm_joint_names: list[str],
                          gripper_pin: float = DEFAULT_GRIPPER_PIN,
                          ) -> tuple[dict[str, float], dict[str, float]]:
    """
    Split a combined 12-key action {joint_0.pos..joint_5.pos, hand_X.pos..hand_P.pos}
    into:
      - arm_cmd  : 7-key dict for the inner Trossen follower (6 arm + pinned gripper)
      - hand_deg : {code: degrees} dict for altbionics.hand.set_fingers()

    `arm_joint_names` is the inner Trossen joint_names (7, last = gripper). The
    6 arm keys are taken as joint_names[:-1] (".pos"-suffixed); the gripper slot
    is pinned to `gripper_pin`; the hand keys are converted rad->deg.
    """
    arm_keys = [f"{name}.pos" for name in arm_joint_names[:-1]]  # 6 arm joints
    arm_cmd = {k: action[k] for k in arm_keys}
    arm_cmd[f"{arm_joint_names[-1]}.pos"] = gripper_pin          # pin gripper
    hand_deg = features_to_hand_angles(action)                  # rad->deg, hand_* only
    return arm_cmd, hand_deg
