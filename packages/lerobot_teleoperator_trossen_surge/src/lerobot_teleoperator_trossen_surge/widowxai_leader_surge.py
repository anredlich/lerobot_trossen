import logging
from typing import Optional

from altbionics import ik
from altbionics.glove import ManusGlove
from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot_teleoperator_trossen.widowxai_leader import WidowXAILeaderTeleop
from trossen_surge_common.surge_dataset import (
    GRIPPER_FEATURE_KEY,
    HAND_FEATURE_KEYS,
    hand_angles_to_features,
)

from lerobot_teleoperator_trossen_surge.config_widowxai_leader_surge import (
    WidowXAILeaderTeleopSurgeConfig,
)

logger = logging.getLogger(__name__)


class WidowXAILeaderTeleopSurge(Teleoperator):
    """
    WidowX AI Solo leader + Alt-Bionics Surge hand (driven by a Manus glove).

    Composition wrapper (the plain WidowXAILeaderTeleop is used unmodified):
    the inner arm leader provides the 6 arm joint positions; the Manus glove
    provides the 6 hand joint angles via the finger IK (altbionics.ik). The
    gripper joint of the inner leader is dropped (the physical gripper is
    replaced by the hand). get_action() therefore returns 6 arm + 6 hand = 12
    "<name>.pos" values.

    Reverting to the plain gripper leader: select --teleop.type=widowxai_leader_teleop.
    """

    config_class = WidowXAILeaderTeleopSurgeConfig
    name = "widowxai_leader_teleop_surge"

    def __init__(
        self,
        config: WidowXAILeaderTeleopSurgeConfig,
        glove: Optional[ManusGlove] = None,
    ):
        super().__init__(config)
        self.config = config

        # Composition: the inner arm leader is the plain Trossen class, used
        # exactly as-is. It is a WidowXAILeaderTeleopConfig; our config is a
        # subclass of it, so it already carries every field the inner needs.
        self._arm = WidowXAILeaderTeleop(config)

        # Glove: created from config (the lerobot CLI construction path), or
        # injected for testing.
        self._owns_glove = glove is None
        self._glove = glove if glove is not None else ManusGlove(so_path=config.manus_so_path)

        # The arm joint keys we keep (inner leader's joints minus the gripper),
        # resolved once at construction from the inner config's joint_names.
        self._arm_pos_keys = [f"{name}.pos" for name in config.joint_names[:-1]]  # drop gripper

        self._glove_started = False

    # -- feature schema (dataset / Rerun columns) ----------------------------

    @property
    def action_features(self) -> dict[str, type]:
        """6 arm joints (inner leader minus gripper) + 6 hand joints."""
        arm = {k: v for k, v in self._arm.action_features.items() if k != GRIPPER_FEATURE_KEY}
        hand = {k: float for k in HAND_FEATURE_KEYS}
        return {**arm, **hand}

    @property
    def feedback_features(self) -> dict[str, type]:
        # Force feedback is not implemented upstream; match the inner leader.
        return self._arm.feedback_features

    # -- lifecycle (explicit delegation) -------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._arm.is_connected and self._glove_started

    def connect(self, calibrate: bool = True) -> None:
        self._arm.connect(calibrate=calibrate)
        self._glove.start()
        self._glove_started = True
        logger.info(f"{self} connected (arm + Manus glove).")

    @property
    def is_calibrated(self) -> bool:
        # Trossen arms need no calibration; the hand IK is fixed coefficients.
        return self._arm.is_calibrated

    def calibrate(self) -> None:
        self._arm.calibrate()

    def configure(self) -> None:
        # Delegate any inner configuration (external-effort / gravity-comp mode).
        if hasattr(self._arm, "configure"):
            self._arm.configure()

    def disconnect(self) -> None:
        if self._owns_glove and self._glove_started:
            self._glove.stop()
            self._glove_started = False
        self._arm.disconnect()
        logger.info(f"{self} disconnected.")

    # -- the action: 6 arm (minus gripper) + 6 hand (radians) ----------------

    def get_action(self) -> dict[str, float]:
        # 1. inner arm leader -> 7 "<joint>.pos" (incl. the gripper).
        arm_action = self._arm.get_action()

        # 2. interface check: the inner leader must return exactly the keys we
        #    expect. If Trossen changes the leader's output shape, fail loudly
        #    here rather than silently building a malformed action.
        expected = set(self._arm_pos_keys) | {GRIPPER_FEATURE_KEY}
        got = set(arm_action.keys())
        if got != expected:
            raise RuntimeError(
                f"{self.name}: inner WidowXAILeaderTeleop.get_action() returned "
                f"keys {sorted(got)}, expected {sorted(expected)}. The Trossen "
                f"leader interface may have changed — update this wrapper "
                f"(config.joint_names and surge_dataset.GRIPPER_FEATURE_KEY)."
            )

        # 3. drop the gripper; keep the 6 arm joints.
        action = {k: arm_action[k] for k in self._arm_pos_keys}

        # ---- calibration seam (future [K] in-situ calibration) -------------
        # The raw glove sensor positions are read here, before IK. A future
        # calibration mode would tap `sensors` (and the hand's commanded/true
        # angles) to collect paired samples and refit altbionics.ik. Keep this
        # read/solve split clean so that hook stays easy to add.
        sensors = self._glove.read_sensors()  # list[5] of (x,y,z) m, or None
        # --------------------------------------------------------------------

        # 4. glove -> IK -> hand angles (deg) -> hand_*.pos (radians).
        #    If no fresh glove frame yet, hold the hand open (all-zero degrees)
        #    rather than emitting nothing (lerobot expects a full action dict).
        if sensors is None:
            hand_deg = {c: 0.0 for c in ik.JOINT_CODES}
        else:
            hand_deg = ik.solve(sensors)
        action.update(hand_angles_to_features(hand_deg))

        return action

    def send_feedback(self, feedback: dict[str, float]) -> None:
        # Not implemented upstream; delegate so behaviour matches the inner leader.
        self._arm.send_feedback(feedback)
