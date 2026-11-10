import logging
from typing import Optional
import math

from altbionics import ik
from altbionics.glove import ManusGlove, quat_to_euler_deg
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

# --- glove-roll -> arm joint 5 (experimental, minimal) ----------------------
# Add a small fraction of the glove's whole-hand ROLL onto leader joint 5, in
# place under the same ".pos" key so lerobot sees an ordinary leader joint. No
# zeroing/clamping yet — intentionally minimal to observe the raw effect. -anr
INJECT_GLOVE_ROLL = True
ROLL_SCALE = 1.0 #0.1   # fraction of hand-roll (rad) added to joint 5; try smaller/negative
INJECT_GLOVE_PITCH = True
PITCH_SCALE = 1.0 #0.1   # fraction of hand-pitch (rad) added to joint 4; try smaller/negative
INJECT_GLOVE_YAW = True
YAW_SCALE = 1.0 #0.25   # fraction of hand-yaw (rad) added to joint 4; try smaller/negative

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

        # Orientation zero-reference (degrees), latched by on_start() -anr
        self._yaw_ref_deg: float = 0.0
        self._j0_ref: float = 0.0

        # Hand-IK in-situ calibration (off unless config.calibrate_hand). Holds
        # all trim/capture state; this wrapper only forwards keys/obs/sensors. -anr
        self._calib = None
        if getattr(config, "calibrate_hand", False):
            from altbionics.calibrate import HandCalibrator
            self._calib = HandCalibrator()
            logger.info(f"{self}: hand calibration ON — logging to {self._calib.path}")

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

    def on_start(self) -> None:
        """Operator-pressed-start hook. Generic seam for any start-time setup;
        today it latches the current glove orientation as the zero reference so
        the hand's attitude at start maps to zero offset on all injected axes.
        Safe to call repeatedly (each call re-zeros). No-op if no glove frame
        yet (the next get_action will auto-latch instead). -anr"""
        try:
            self._j0_ref = float(self._arm.get_action().get(self._arm_pos_keys[0], 0.0))
        except Exception as e:
            self._j0_ref = 0.0
            logger.info(f"{self}: on_start() — could not latch joint-0 ref ({e}); using 0.0.")
        q = self._glove.read_hand_rotation()
        if q is None:
            logger.info(f"{self}: on_start() — no glove frame yet; will auto-zero on first action.")
            return
        roll_ref_deg, pitch_ref_deg, self._yaw_ref_deg = quat_to_euler_deg(q)
        logger.info(
            f"{self}: yaw orientation zeroed = {self._yaw_ref_deg:.1f} deg)."
        )

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

        # ---- glove roll/pitch -> arm joints 5/4 (minimal; see constants) ----
        if INJECT_GLOVE_ROLL or INJECT_GLOVE_PITCH or INJECT_GLOVE_YAW:
            q = self._glove.read_hand_rotation()   # (w,x,y,z) or None until first frame
            if q is not None:
                roll_deg, pitch_deg, yaw_deg = quat_to_euler_deg(q)
                if INJECT_GLOVE_ROLL:
                    j5 = self._arm_pos_keys[-1]
                    action[j5] = action[j5] + math.radians(roll_deg) * ROLL_SCALE
                if INJECT_GLOVE_YAW:
                    #j4 = self._arm_pos_keys[-2]
                    #d_yaw = (yaw_deg - self._yaw_ref_deg + 180.0) % 360.0 - 180.0
                    #action[j4] = action[j4] - math.radians(d_yaw) * YAW_SCALE
                    #print(f"yaw-ref: {d_yaw} yaw: {yaw_deg} yaw_ref: {self._yaw_ref_deg}")
                    # Glove yaw is absolute (world-frame, magnetometer-fused), hence need to subract j0.
                    j4 = self._arm_pos_keys[-2]
                    j0 = self._arm_pos_keys[0]
                    d_yaw = (yaw_deg - self._yaw_ref_deg + 180.0) % 360.0 - 180.0
                    base_rot = action[j0] - self._j0_ref
                    action[j4] = action[j4] - math.radians(d_yaw) * YAW_SCALE + base_rot
                if INJECT_GLOVE_PITCH:
                    j3 = self._arm_pos_keys[-3]
                    j1 = self._arm_pos_keys[1]
                    j2 = self._arm_pos_keys[2]
                    darm=action[j2] - action[j1]
                    action[j3] = action[j3] - math.radians(pitch_deg) * PITCH_SCALE - darm
        # --------------------------------------------------------------------
                
        # ---- calibration seam (future [K] in-situ calibration) -------------
        # The raw glove sensor positions are read here, before IK. A future
        # calibration mode would tap `sensors` (and the hand's commanded/true
        # angles) to collect paired samples and refit altbionics.ik. Keep this
        # read/solve split clean so that hook stays easy to add.
        sensors = self._glove.read_sensors()  # list[5] of (x,y,z) m, or None
        if self._calib is not None:
            self._calib.set_sensors(sensors)        
        # --------------------------------------------------------------------

        # 4. glove -> IK -> hand angles (deg) -> hand_*.pos (radians).
        #    If no fresh glove frame yet, hold the hand open (all-zero degrees)
        #    rather than emitting nothing (lerobot expects a full action dict).
        if sensors is None:
            hand_deg = {c: 0.0 for c in ik.JOINT_CODES}
        else:
            hand_deg = ik.solve(sensors)
            if self._calib is not None:
                # trim nudges the IK COMMAND so you can match the finger to your
                # hand; the recorded target is the measured angle (see observe). -anr
                for c in ik.JOINT_CODES:
                    hand_deg[c] = hand_deg[c] + self._calib.trim[c]
        action.update(hand_angles_to_features(hand_deg))
        
        return action

    def send_feedback(self, feedback: dict[str, float]) -> None:
        # Not implemented upstream; delegate so behaviour matches the inner leader.
        self._arm.send_feedback(feedback)

    # -- calibration hooks (no-ops unless config.calibrate_hand) -------------
    def on_key(self, key) -> None:
        """Raw keypress from the teleop loop. Forwarded to the hand calibrator
        (trim keys / r reset / Enter capture) when calibration is on. -anr"""
        if self._calib is not None:
            self._calib.on_key(key)

    def on_observation(self, obs: dict) -> None:
        """Each tick's follower observation. Carries the MEASURED hand angle
        (hand_<code>.pos) the calibrator records as the regression target. -anr"""
        if self._calib is not None:
            self._calib.observe(obs)