import logging
from typing import Any, Optional

from altbionics import ik
from altbionics.hand import SurgeHand, SurgeConfig, autodetect_port
from lerobot.robots.robot import Robot
from lerobot_robot_trossen.widowxai_follower import WidowXAIFollower
from trossen_surge_common.surge_dataset import (
    GRIPPER_FEATURE_KEY,
    GRIPPER_JOINT_NAME,
    HAND_FEATURE_KEYS,
    HAND_JOINT_CODES,
    deg_to_rad,
    features_to_hand_angles,
    split_combined_action,
)

from lerobot_robot_trossen_surge.config_widowxai_follower_surge import (
    WidowXAIFollowerSurgeConfig,
)

logger = logging.getLogger(__name__)


class WidowXAIFollowerSurge(Robot):
    """
    WidowX AI Solo follower + Alt-Bionics Surge hand.

    Composition wrapper (the plain WidowXAIFollower is used unmodified): the
    inner arm follower drives the 6 arm joints (plus a pinned gripper slot it
    still physically expects), and the Surge hand is driven from the 6 hand
    joints. The physical gripper is removed, so it is pinned to a constant on
    the command side and dropped from the dataset.

        state  (observation) = 6 arm .pos (measured) + 6 hand .pos (measured GJA) + cameras
        action              = 6 arm .pos (sent)      + 6 hand .pos (sent)

    all joint values in radians. The hand is driven via AT+SJA (the SOLE driver
    of the hand; nothing else commands it during teleop/record), and the hand's
    Manus mode is asserted on connect from ik.IK_REGRESSION_MANUS_MODE so it
    always matches the mode the IK coefficients were fit in.

    Reverting to the plain gripper follower: select --robot.type=widowxai_follower_robot.
    """

    config_class = WidowXAIFollowerSurgeConfig
    name = "widowxai_follower_robot_surge"

    def __init__(
        self,
        config: WidowXAIFollowerSurgeConfig,
        hand: Optional[SurgeHand] = None,
    ):
        super().__init__(config)
        self.config = config

        self._hand_read_mode = getattr(config, "hand_read_mode", "per_joint")

        # Composition: the inner arm follower is the plain Trossen class, used
        # exactly as-is. Our config subclasses its config, so it carries every
        # field the inner needs.
        self._arm = WidowXAIFollower(config)

        # Hand: created from config (CLI path) or injected for testing.
        self._owns_hand = hand is None
        if hand is not None:
            self._hand = hand
        else:
            port = autodetect_port(config.hand_serial)
            if port is None:
                raise RuntimeError(
                    f"{self.name}: no Surge hand serial port found "
                    f"(looked for FTDI serial {config.hand_serial!r}). Is the hand plugged in?"
                )
            # Direct control: do NOT auto-send AT+SMC=0 on connect; we assert the
            # IK's Manus mode explicitly in connect().
            hcfg = SurgeConfig(
                port=port,
                disable_manus_on_connect=False,
                send_smc_on_connect=False,
                log_commands=False,
                log_incoming=False,
            )
            self._hand = SurgeHand(config=hcfg)

        # Arm joint keys we keep (inner joints minus the gripper), resolved once.
        self._arm_pos_keys = [f"{n}.pos" for n in config.joint_names[:-1]]  # 6 arm

        self._hand_connected = False
        # Last-known hand GET angles (deg), held when a GJA read returns None so
        # the observation dict is always complete. 0.0 until first good read.
        self._last_hand_deg: dict[str, float] = {c: 0.0 for c in HAND_JOINT_CODES}

        # Last hand action actually sent (deg), for the static-accuracy diagnostic
        # (compares measured state to the command once a joint is parked).
        self._last_hand_sent_deg: dict[str, float] = {c: 0.0 for c in HAND_JOINT_CODES}

        # --- bad-read bookkeeping (production): counts kept always; per-event
        # --- warnings and the static-accuracy stats below are gated by _diag_hand.
        self._diag_hand = False   # set True to print bad-read warnings + static stats
        self._bad_count: dict[str, int] = {c: 0 for c in HAND_JOINT_CODES}
        self._read_count: int = 0
        # static-accuracy diagnostic state (only used when _diag_hand):
        self._acc_window = 100
        self._prev_sent_deg: dict[str, float] = {c: 0.0 for c in HAND_JOINT_CODES}
        self._stable_ticks: dict[str, int] = {c: 0 for c in HAND_JOINT_CODES}
        self._settled_err: dict[str, list[float]] = {c: [] for c in HAND_JOINT_CODES}
        self._settled_state: dict[str, list[float]] = {c: [] for c in HAND_JOINT_CODES}

        # --- Step-2 responsiveness diagnostics (behind _diag_hand) ---
        # Offset-agnostic (works despite the Manus-mode action/state frame offset):
        # we correlate the DERIVATIVES (Δ per tick) of command vs measured state.
        self._resp_window = 150            # ticks per responsiveness report (~12s @ 12Hz)
        self._resp_max_lag = 8             # ticks to search for best Δ-alignment
        self._cmd_hist: dict[str, list[float]] = {c: [] for c in HAND_JOINT_CODES}
        self._state_hist: dict[str, list[float]] = {c: [] for c in HAND_JOINT_CODES}

    # -- feature schema (dataset / Rerun columns) ----------------------------

    def _strip_gripper(self, feats: dict) -> dict:
        """Drop every key belonging to the gripper joint (.pos/.vel/.eff/.ext_eff)."""
        return {
            k: v for k, v in feats.items()
            if not k.startswith(f"{GRIPPER_JOINT_NAME}.")
        }

    @property
    def observation_features(self) -> dict[str, type | tuple]:
        """6 arm joints (inner minus gripper) + cameras + 6 hand joints."""
        arm = self._strip_gripper(self._arm.observation_features)
        hand = {k: float for k in HAND_FEATURE_KEYS}
        return {**arm, **hand}

    @property
    def action_features(self) -> dict[str, type]:
        """6 arm joints (inner minus gripper) + 6 hand joints."""
        arm = self._strip_gripper(self._arm.action_features)
        hand = {k: float for k in HAND_FEATURE_KEYS}
        return {**arm, **hand}

    # -- lifecycle (explicit delegation) -------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._arm.is_connected and self._hand_connected

    def connect(self, calibrate: bool = True) -> None:
        self._arm.connect(calibrate=calibrate)
        self._hand.connect()
        # Assert the Manus mode the IK was fit in — the Python mirror of the
        # C++ setManusMode(IK_REGRESSION_MANUS_MODE). The hand must already be
        # powered (it is; the robot is powered before control software starts).
        self._hand.set_manus_mode(ik.IK_REGRESSION_MANUS_MODE == 1)
        self._hand_connected = True
        logger.info(
            f"{self} connected (arm + Surge hand, Manus mode "
            f"{ik.IK_REGRESSION_MANUS_MODE})."
        )

    @property
    def is_calibrated(self) -> bool:
        return self._arm.is_calibrated

    def calibrate(self) -> None:
        self._arm.calibrate()

    def configure(self) -> None:
        if hasattr(self._arm, "configure"):
            self._arm.configure()

    def disconnect(self) -> None:
        if self._owns_hand and self._hand_connected:
            self._hand.disconnect()
            self._hand_connected = False
        self._arm.disconnect()
        logger.info(f"{self} disconnected.")

    # -- send action: 6 arm (+ pinned gripper) to arm, 6 hand to the hand ----

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        # interface check: the action must carry the 6 arm + 6 hand keys we expect.
        expected = set(self._arm_pos_keys) | set(HAND_FEATURE_KEYS)
        got = {k for k in action if k.endswith(".pos")}
        if not expected.issubset(got):
            raise RuntimeError(
                f"{self.name}: send_action got keys {sorted(got)}, expected at "
                f"least {sorted(expected)}. Action shape mismatch — check the "
                f"teleoperator/policy output and config.joint_names."
            )

        # split: 7-key arm command (6 arm + pinned gripper), 6 hand degrees.
        arm_cmd, hand_deg = split_combined_action(
            action, self.config.joint_names, self.config.gripper_pin_pos
        )

        # drive the arm; inner returns the actually-sent (clipped) arm action.
        arm_sent = self._arm.send_action(arm_cmd)

        # drive the hand; set_fingers returns {code: actually_sent_deg}.
        hand_sent_deg = self._hand.set_fingers(hand_deg)
        self._last_hand_sent_deg = dict(hand_sent_deg)   # for static-accuracy diag

        # build the combined SENT action to return: 6 arm (gripper stripped) +
        # 6 hand (deg -> rad), so the recorder can log what was actually sent.
        out = {k: v for k, v in arm_sent.items() if k != GRIPPER_FEATURE_KEY}
        for code in HAND_JOINT_CODES:
            out[f"hand_{code}.pos"] = deg_to_rad(hand_sent_deg[code])
        return out

    # -- observation: 6 arm measured + cameras + 6 hand measured (GJA) -------

    def get_observation(self) -> dict[str, Any]:
        # inner arm observation (joints + cameras); drop the gripper joint.
        obs = self._strip_gripper(self._arm.get_observation())

        # Read the 6 hand joints' TRUE measured angle (AT+GJA?). Accept ONLY a
        # physically-valid angle (-5..130 deg); for anything else — a None (dropped
        # frame), a genuine sub-(-5) read, or a >130 read (the firmware's unsigned
        # wrap of a slightly-negative angle: finger pinned at its 0 stop, reported
        # near 255, or 255+25.5≈280) — HOLD the last good value. One rule gives a
        # complete, in-range, physically-plausible 6-DOF hand state every tick,
        # which a dataset requires. Holding is safe: a finger reaches its stop
        # through small valid angles, so the held value is already near the truth
        # and self-corrects on the next valid read.
        self._read_count += 1
        # burst mode: one pipelined read of all 6; else per-joint (current path).
        if self._hand_read_mode == "burst":
            vals = self._hand.get_angles_burst(HAND_JOINT_CODES, timeout=0.08)
        else:
            vals = None
        for code in HAND_JOINT_CODES:
            #val = self._hand.get_angle_fast(code, timeout=0.05)
            val = vals[code] if vals is not None else \
                self._hand.get_angle_fast(code, timeout=0.05)
            if val is not None and -5.0 <= val <= 130.0:
                self._last_hand_deg[code] = val
            else:
                self._bad_count[code] += 1
                if self._diag_hand:
                    reason = "none" if val is None else (
                        "wrap/neg" if val > 130.0 else "sub-(-5)")
                    print(f"  HAND bad-read {code}: "
                          f"{'None' if val is None else f'{val:.1f}deg'} "
                          f"({reason}) -> held {self._last_hand_deg[code]:.1f}",
                          flush=True)
            obs[f"hand_{code}.pos"] = deg_to_rad(self._last_hand_deg[code])

        if self._diag_hand:
            for code in HAND_JOINT_CODES:
                self._cmd_hist[code].append(self._last_hand_sent_deg[code])
                self._state_hist[code].append(self._last_hand_deg[code])
            if len(self._cmd_hist[HAND_JOINT_CODES[0]]) >= self._resp_window:
                self._diag_responsiveness()
            self._diag_static_accuracy()

        return obs

    def _diag_static_accuracy(self) -> None:
        """Print per-joint read fidelity measured only while a joint is PARKED
        (command stable for SETTLE ticks), so lag/contact don't contaminate it:
        err = mean |action-state|, σ = state std, rng = state span. Plus a running
        bad-read count. Diagnostic only; no effect on recorded state."""
        import statistics
        STABLE_EPS = 0.5   # deg: command "unchanged" if it moved < this
        SETTLE = 5         # ticks command must be stable before trusting rest
        for code in HAND_JOINT_CODES:
            sent = self._last_hand_sent_deg[code]
            if abs(sent - self._prev_sent_deg[code]) < STABLE_EPS:
                self._stable_ticks[code] += 1
            else:
                self._stable_ticks[code] = 0
            self._prev_sent_deg[code] = sent
            if self._stable_ticks[code] >= SETTLE:
                self._settled_err[code].append(abs(sent - self._last_hand_deg[code]))
                self._settled_state[code].append(self._last_hand_deg[code])
        if len(self._settled_state[HAND_JOINT_CODES[0]]) >= self._acc_window:
            parts = []
            for code in HAND_JOINT_CODES:
                err, st = self._settled_err[code], self._settled_state[code]
                merr = statistics.fmean(err) if err else 0.0
                sig = statistics.pstdev(st) if len(st) > 1 else 0.0
                rng = (max(st) - min(st)) if st else 0.0
                parts.append(f"{code} err{merr:4.1f} σ{sig:4.1f} rng{rng:4.1f}")
                err.clear(); st.clear()
            bad = sum(self._bad_count.values())
            total = self._read_count * len(HAND_JOINT_CODES)
            print("  HAND static (deg):  " + "  ".join(parts)
                  + f"   | bad reads: {bad}/{total}", flush=True)

    def _diag_responsiveness(self) -> None:
        """Per-joint responsiveness over a window, OFFSET-AGNOSTIC (the Manus-mode
        action/state offset is constant, so it vanishes under differencing):
          lag  = tick shift (of Δstate vs Δcommand) that maximises their
                 correlation -> the actuator+read delay, in ticks.
          r    = correlation at that lag (how cleanly state follows command;
                 ~1 = tight tracking, low = noisy/unrelated).
          trk  = sum|Δstate| / sum|Δcommand| (how COMPLETELY state follows:
                 ~1 faithful, <1 blocked/contact, >1 overshoot).
        Diagnostic only; no effect on recorded state."""
        parts = []
        for code in HAND_JOINT_CODES:
            cmd = self._cmd_hist[code]
            st = self._state_hist[code]
            # derivatives (per-tick change) remove the constant offset.
            dc = [cmd[i] - cmd[i - 1] for i in range(1, len(cmd))]
            ds = [st[i] - st[i - 1] for i in range(1, len(st))]
            cmd_motion = sum(abs(x) for x in dc)
            st_motion = sum(abs(x) for x in ds)
            if cmd_motion < 1e-6:
                parts.append(f"{code} (still)")
                cmd.clear(); st.clear()
                continue
            trk = st_motion / cmd_motion
            # best lag: shift ds back by L ticks, correlate with dc.
            best_lag, best_r = 0, -2.0
            for L in range(0, self._resp_max_lag + 1):
                a = dc[: len(dc) - L] if L else dc
                b = ds[L:] if L else ds
                r = _corr(a, b)
                if r > best_r:
                    best_r, best_lag = r, L
            parts.append(f"{code} lag{best_lag} r{best_r:+.2f} trk{trk:.2f}")
            cmd.clear(); st.clear()
        print("  HAND resp:  " + "  ".join(parts), flush=True)

def _corr(a: list[float], b: list[float]) -> float:
    """Pearson correlation of two equal-length sequences; 0 if degenerate."""
    n = min(len(a), len(b))
    if n < 3:
        return 0.0
    a, b = a[:n], b[:n]
    ma = sum(a) / n
    mb = sum(b) / n
    num = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    da = sum((x - ma) ** 2 for x in a) ** 0.5
    db = sum((x - mb) ** 2 for x in b) ** 0.5
    return num / (da * db) if da > 0 and db > 0 else 0.0

    # def get_observation(self) -> dict[str, Any]:
    #     # snapshot of last tick's sent command, so the spike print can say whether
    #     # the spiking joint was being actively driven (the suspected cause).
    #     _cmd_prev = dict(self._prev_sent_deg)        
    #     # inner arm observation (joints + cameras); drop the gripper joint.
    #     obs = self._strip_gripper(self._arm.get_observation())

    #     # read the 6 hand joints' TRUE measured angle (AT+GJA?). On a None read
    #     # (transient FTDI hiccup), hold the last-known value so the dict stays
    #     # complete — required for a dataset (no missing fields).
    #     # Read ALL six hand joints this tick (full fresh state). Each read is the
    #     # hardened get_angle_fast: a correct angle, or None on a transient miss.
    #     # On None we hold the last-known value so the observation dict is always
    #     # complete (a dataset can have no missing fields); a held value is safe
    #     # because finger angles change slowly between ticks.
    #     # for code in HAND_JOINT_CODES:
    #     #     val = self._hand.get_angle_fast(code, timeout=0.05)
    #     #     if val is not None:
    #     #         self._last_hand_deg[code] = val
    #     #     obs[f"hand_{code}.pos"] = deg_to_rad(self._last_hand_deg[code])
    #     self._read_count += 1
    #     if self._diag_hand:
    #         import time as _t
    #         #_t.sleep(0.002)   # DIAG: settle after SJA writes — does this cut bad reads?
    #     for code in HAND_JOINT_CODES:
    #         val = self._hand.get_angle_fast(code, timeout=0.05)
    #         # Physical-range gate: the hand can only be 0..127 (protocol range).
    #         # A read outside that is a corrupt/misframed reply that slipped the
    #         # driver guard (the Rerun state spikes) — hold the last good value
    #         # instead of poisoning the dataset, and count it. This gate is a
    #         # correctness guard, worth keeping even after diagnostics come off.
    #         # if val is not None and -5.0 <= val <= 130.0:
    #         #     self._last_hand_deg[code] = val
    #         # elif val is not None:
    #         #     self._spike_count[code] += 1   # out-of-range -> rejected, hold last
    #         #     if self._diag_hand:
    #         #         print(f"  SPIKE {code}: {val:.1f} deg (rejected)", flush=True)
    #         # obs[f"hand_{code}.pos"] = deg_to_rad(self._last_hand_deg[code])
    #         if val is None:
    #             self._none_count[code] += 1    # misframed/no-reply -> held
    #         elif -5.0 <= val <= 130.0:
    #             self._last_hand_deg[code] = val
    #         else:
    #             self._spike_count[code] += 1    # out-of-range -> rejected, held
    #             if self._diag_hand:
    #                 moving = abs(self._last_hand_sent_deg[code] - _cmd_prev[code]) >= 0.5
    #                 print(f"  SPIKE {code}: {val:.1f} deg  "
    #                       f"(cmd {'MOVING' if moving else 'parked'})", flush=True)
    #         obs[f"hand_{code}.pos"] = deg_to_rad(self._last_hand_deg[code])

    #     if self._diag_hand:
    #         import statistics
    #         STABLE_EPS = 0.5   # deg: command "unchanged" if it moved < this
    #         SETTLE = 5         # ticks command must be stable before trusting rest
    #         for code in HAND_JOINT_CODES:
    #             sent = self._last_hand_sent_deg[code]
    #             if abs(sent - self._prev_sent_deg[code]) < STABLE_EPS:
    #                 self._stable_ticks[code] += 1
    #             else:
    #                 self._stable_ticks[code] = 0
    #             self._prev_sent_deg[code] = sent
    #             if self._stable_ticks[code] >= SETTLE:
    #                 state = self._last_hand_deg[code]
    #                 self._settled_err[code].append(abs(sent - state))
    #                 self._settled_state[code].append(state)
    #         if len(self._settled_state[HAND_JOINT_CODES[0]]) >= self._acc_window:
    #             parts = []
    #             for code in HAND_JOINT_CODES:
    #                 err = self._settled_err[code]
    #                 st = self._settled_state[code]
    #                 merr = statistics.fmean(err) if err else 0.0
    #                 sig = statistics.pstdev(st) if len(st) > 1 else 0.0
    #                 rng = (max(st) - min(st)) if st else 0.0
    #                 parts.append(f"{code} err{merr:4.1f} σ{sig:4.1f} rng{rng:4.1f}")
    #                 err.clear(); st.clear()
    #             spikes = sum(self._spike_count.values())
    #             nones = sum(self._none_count.values())
    #             total = self._read_count * len(HAND_JOINT_CODES)
    #             print("  HAND static (deg):  " + "  ".join(parts)
    #                   + f"   | bad reads: spike {spikes} + none {nones} / {total}",
    #                   flush=True)
    #     return obs

