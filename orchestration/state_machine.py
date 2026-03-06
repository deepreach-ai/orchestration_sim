"""
orchestration/state_machine.py
-------------------------------
Extracted and extended from scripts/pick_place_rmpflow.py.

Provides two state machines:
  - SingleArmSM   : original single-arm pick-place (Day 4 baseline)
  - DualArmSM     : dual-arm orchestration with handover zone (Day 5 target)

Both machines share the same ArmState enum and transition logging.
The LulaController dependency is imported from the existing script
so there is no code duplication.

Usage (single arm — drop-in replacement for old PickPlaceStateMachine):
    from orchestration.state_machine import SingleArmSM
    sm = SingleArmSM(ctrl, box=box, box_prim_path="/World/box")
    done = sm.tick(step)

Usage (dual arm — Day 5):
    from orchestration.state_machine import DualArmSM, ArmRole
    sm = DualArmSM(
        picker_ctrl=ctrl_a, placer_ctrl=ctrl_b,
        box=box, box_prim_path="/World/box",
        handover_xyz=HANDOVER_XYZ,
    )
    done = sm.tick(step)
"""

from __future__ import annotations

import logging
from enum import Enum, auto
from typing import Callable, Optional

import numpy as np

# NOTE: isaacsim.* imports are intentionally deferred to inside methods/
# __init__ bodies. SimulationApp must be instantiated before any isaacsim
# module is imported — importing at module load time causes:
#   ModuleNotFoundError: No module named 'isaacsim.core'
# See: Isaac Sim Carbonite framework requirement.

# ── Fallback constants (copy of confirmed-working values from pick_place_rmpflow)
# NOT imported from the script at module level — that would pull in isaacsim too.
PICK_XYZ    = np.array([0.30,  0.0,  0.50])
PLACE_XYZ   = np.array([0.20, -0.30, 0.50])
HOVER_Z     = 0.12
HOME_JOINTS = np.array([0.0, -0.009, 0.0, -0.439, 0.0, 1.92, 0.0])
N_ARM_DOFS  = 7
# Gripper: only Left_1_Joint is revolute (±0.5 rad).
# All other gripper joints are fixed (mimic removed for PhysX compat).
# Full open = +0.5 rad, fully closed = 0.0 rad.
GRIP_OPEN   = np.array([0.5])
GRIP_CLOSE  = np.array([0.0])


# ── Logger (writes to Isaac Sim-safe logger, not stdout) ─────────────────────
logger = logging.getLogger("orchestration.state_machine")
logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")

def _log(msg: str):
    logger.info(msg)
    print(msg, flush=True)


def _ee_quat_default() -> np.ndarray:
    """Lazy import of euler_angles_to_quat — only safe after SimulationApp()."""
    from isaacsim.core.utils.rotations import euler_angles_to_quat
    return euler_angles_to_quat(np.array([np.pi, 0.0, 0.0]))


# ── ArmRole — used by DualArmSM to distinguish arms ──────────────────────────
class ArmRole(Enum):
    """
    Identifies the role of an arm in a dual-arm setup.

    PICKER  : Arm A — picks from conveyor / source zone, deposits at handover.
    PLACER  : Arm B — retrieves from handover zone, places at output pile.

    In single-arm mode this distinction is unused.
    """
    PICKER = auto()
    PLACER = auto()


# ── ArmState — all possible states for a single arm ──────────────────────────
class ArmState(Enum):
    """
    Unified state enum shared by both single- and dual-arm machines.

    States are ordered by typical execution sequence.
    HANDOVER_DEPOSIT / HANDOVER_RETRIEVE are only used in DualArmSM.
    FALLBACK_RETRY / FALLBACK_SKIP are injected by fallback logic (Day 6).
    """
    IDLE              = auto()
    HOME              = auto()
    HOVER_PICK        = auto()
    DESCEND_PICK      = auto()
    GRASP             = auto()
    LIFT              = auto()
    # ── single-arm direct path ──────────────────────────────────────────────
    HOVER_PLACE       = auto()
    DESCEND_PLACE     = auto()
    RELEASE           = auto()
    # ── dual-arm handover path ──────────────────────────────────────────────
    HANDOVER_DEPOSIT  = auto()   # PICKER: move to handover zone, release
    HANDOVER_WAIT     = auto()   # PLACER: wait until PICKER signals deposit done
    HANDOVER_RETRIEVE = auto()   # PLACER: pick from handover zone
    # ── completion + fallback ───────────────────────────────────────────────
    HOME_FINAL        = auto()
    DONE              = auto()
    FALLBACK_RETRY    = auto()   # stub for Day 6 fallback_handler
    FALLBACK_SKIP     = auto()   # stub for Day 6 fallback_handler
    ERROR             = auto()


# ── Dwell table — sim steps to spend in each state ───────────────────────────
# Extracted from pick_place_rmpflow.py and extended for handover states.
# Tune these after physics benchmarking on Day 5.
_DEFAULT_DWELL: dict[ArmState, int] = {
    ArmState.HOME:              80,
    ArmState.HOVER_PICK:       120,
    ArmState.DESCEND_PICK:     120,
    ArmState.GRASP:             80,
    ArmState.LIFT:             120,
    ArmState.HOVER_PLACE:      200,
    ArmState.DESCEND_PLACE:    150,
    ArmState.RELEASE:           80,
    ArmState.HANDOVER_DEPOSIT: 150,   # PICKER deposits at handover
    ArmState.HANDOVER_WAIT:    300,   # PLACER waits — generous timeout
    ArmState.HANDOVER_RETRIEVE:120,   # PLACER picks up from handover
    ArmState.HOME_FINAL:        60,
}


# ─────────────────────────────────────────────────────────────────────────────
#  SingleArmSM
# ─────────────────────────────────────────────────────────────────────────────
class SingleArmSM:
    """
    Drop-in replacement for PickPlaceStateMachine in pick_place_rmpflow.py.

    Behaviour is identical to the original but:
      - Uses ArmState enum instead of raw strings (safer for dual-arm later)
      - Dwell table is externally configurable
      - Transition callback hook added for orchestrator / metrics logging
      - IK failure counter exposed for Day 6 fallback integration

    Day 4 task: verify this produces the same result as the original script.
    """

    def __init__(
        self,
        ctrl: "LulaController",
        box=None,
        box_prim_path: Optional[str] = None,
        pick_xyz: np.ndarray = PICK_XYZ,
        place_xyz: np.ndarray = PLACE_XYZ,
        hover_z: float = HOVER_Z,
        dwell_override: Optional[dict] = None,
        on_transition: Optional[Callable[[ArmState, ArmState, int], None]] = None,
    ):
        self.ctrl           = ctrl
        self._box           = box
        self._box_prim_path = box_prim_path
        self.pick_xyz       = pick_xyz
        self.place_xyz      = place_xyz
        self.hover_z        = hover_z
        self._ee_quat       = _ee_quat_default()

        self.dwell = dict(_DEFAULT_DWELL)
        if dwell_override:
            self.dwell.update(dwell_override)

        self.state            = ArmState.IDLE
        self.state_entry_step = 0
        self.ik_fail_count    = 0   # exposed for fallback_handler (Day 6)
        self.cycle_count      = 0   # how many full pick-place cycles completed

        # Optional external callback: on_transition(old_state, new_state, step)
        self._on_transition = on_transition

    # ── Public ───────────────────────────────────────────────────────────────

    def tick(self, step: int) -> bool:
        """
        Advance the state machine by one simulation step.
        Returns True when a full pick-place cycle completes.
        """
        n = step - self.state_entry_step  # steps elapsed in current state

        if self.state == ArmState.IDLE:
            self.ctrl.set_gripper(closed=False)
            self._go(ArmState.HOME, step)

        elif self.state == ArmState.HOME:
            self.ctrl.move_home()
            if n > self.dwell[ArmState.HOME]:
                self._go(ArmState.HOVER_PICK, step)

        elif self.state == ArmState.HOVER_PICK:
            ok = self.ctrl.move_to(
                self.pick_xyz + np.array([0, 0, self.hover_z]),
                self._ee_quat,
                warm_hint=self.ctrl.WARM_PICK,
            )
            self._track_ik(ok)
            if n > self.dwell[ArmState.HOVER_PICK]:
                self._go(ArmState.DESCEND_PICK, step)

        elif self.state == ArmState.DESCEND_PICK:
            ok = self.ctrl.move_to(self.pick_xyz, self._ee_quat, warm_hint=self.ctrl.WARM_PICK)
            self._track_ik(ok)
            if n > self.dwell[ArmState.DESCEND_PICK]:
                self._go(ArmState.GRASP, step)

        elif self.state == ArmState.GRASP:
            self.ctrl.set_gripper(closed=True)
            if n == 1 and self._box_prim_path:
                self.ctrl.attach_box(self._box_prim_path)
            if self._box:
                self.ctrl.move_box_with_ee(self._box)
            if n > self.dwell[ArmState.GRASP]:
                self._go(ArmState.LIFT, step)

        elif self.state == ArmState.LIFT:
            ok = self.ctrl.move_to(
                self.pick_xyz + np.array([0, 0, self.hover_z]),
                self._ee_quat,
                warm_hint=self.ctrl.WARM_PICK,
            )
            self._track_ik(ok)
            if self._box:
                self.ctrl.move_box_with_ee(self._box)
            if n > self.dwell[ArmState.LIFT]:
                self._go(ArmState.HOVER_PLACE, step)

        elif self.state == ArmState.HOVER_PLACE:
            ok = self.ctrl.move_to(
                self.place_xyz + np.array([0, 0, self.hover_z]),
                self._ee_quat,
                warm_hint=self.ctrl.WARM_PLACE,
            )
            self._track_ik(ok)
            if self._box:
                self.ctrl.move_box_with_ee(self._box)
            if n > self.dwell[ArmState.HOVER_PLACE]:
                self._go(ArmState.DESCEND_PLACE, step)

        elif self.state == ArmState.DESCEND_PLACE:
            ok = self.ctrl.move_to(
                self.place_xyz, self._ee_quat, warm_hint=self.ctrl.WARM_PLACE
            )
            self._track_ik(ok)
            if self._box:
                self.ctrl.move_box_with_ee(self._box)
            if n > self.dwell[ArmState.DESCEND_PLACE]:
                self._go(ArmState.RELEASE, step)

        elif self.state == ArmState.RELEASE:
            self.ctrl.set_gripper(closed=False)
            if n == 1:
                self.ctrl.detach_box()
            if n > self.dwell[ArmState.RELEASE]:
                self._go(ArmState.HOME_FINAL, step)

        elif self.state == ArmState.HOME_FINAL:
            self.ctrl.move_home()
            if n > self.dwell[ArmState.HOME_FINAL]:
                self.cycle_count += 1
                _log(f"✅ [step {step}] SingleArmSM cycle {self.cycle_count} complete "
                     f"(IK fails this cycle: {self.ik_fail_count})")
                self._go(ArmState.DONE, step)
                return True

        return False

    def reset(self, step: int = 0):
        """Reset to IDLE for a new cycle. Resets IK fail counter."""
        self.state            = ArmState.IDLE
        self.state_entry_step = step
        self.ik_fail_count    = 0

    # ── Internal ─────────────────────────────────────────────────────────────

    def _go(self, new_state: ArmState, step: int):
        old = self.state
        _log(f"   [{step:>5}] SingleArmSM: {old.name} → {new_state.name}")
        if self._on_transition:
            self._on_transition(old, new_state, step)
        self.state            = new_state
        self.state_entry_step = step

    def _track_ik(self, ok: bool):
        if not ok:
            self.ik_fail_count += 1


# ─────────────────────────────────────────────────────────────────────────────
#  DualArmSM  (Day 5 target)
# ─────────────────────────────────────────────────────────────────────────────
class DualArmSM:
    """
    Dual-arm orchestration with task handover.

    Architecture
    ------------
    Arm A (PICKER)  : conveyor → pick → lift → handover zone → deposit → home
    Arm B (PLACER)  : wait for handover signal → retrieve → place at pile → home

    Handover zone is a shared neutral XYZ position where Arm A deposits
    and Arm B retrieves. This avoids simultaneous workspace conflict.

    Synchronization
    ---------------
    A shared Event flag (_handover_ready) acts as the signal between arms:
      - Arm A sets it True  when HANDOVER_DEPOSIT completes
      - Arm B reads it in   HANDOVER_WAIT and proceeds when True
      - Arm B clears it     after HANDOVER_RETRIEVE begins

    Collision avoidance
    -------------------
    Workspace partitioning is enforced at the zone level (Day 5 scope).
    Full collision mesh checking is deferred to Day 5 / Day 6.
    The PICKER operates in +X half-space; PLACER operates in -X half-space.
    They share only the handover zone (origin ± tolerance).

    Day 5 TODO
    ----------
    [ ] Tune HANDOVER_XYZ to real warehouse scene (sim/warehouse_scene.py)
    [ ] Add workspace partition check: assert picker_pos.x > 0, placer_pos.x < 0
    [ ] Connect fallback_handler for grasp failures (Day 6)
    [ ] Add takt time pacing (orchestration doc Day 5 requirement)
    """

    def __init__(
        self,
        picker_ctrl: "LulaController",
        placer_ctrl: "LulaController",
        box=None,
        box_prim_path: Optional[str] = None,
        handover_xyz: Optional[np.ndarray] = None,
        place_xyz: np.ndarray = PLACE_XYZ,
        hover_z: float = HOVER_Z,
        dwell_override: Optional[dict] = None,
        on_transition: Optional[Callable[[ArmRole, ArmState, ArmState, int], None]] = None,
    ):
        # Default handover zone: midpoint between pick and place, raised
        self._handover_xyz = handover_xyz if handover_xyz is not None else np.array([0.0, 0.0, 0.55])

        self._ee_quat = _ee_quat_default()
        self.hover_z  = hover_z

        self.dwell = dict(_DEFAULT_DWELL)
        if dwell_override:
            self.dwell.update(dwell_override)

        self._on_transition = on_transition

        # Shared handover signal
        self._handover_ready = False

        # ── Picker SM ────────────────────────────────────────────────────────
        self._picker = _ArmFSM(
            role=ArmRole.PICKER,
            ctrl=picker_ctrl,
            box=box,
            box_prim_path=box_prim_path,
            pick_xyz=PICK_XYZ,
            target_xyz=self._handover_xyz,   # PICKER targets handover zone, not pile
            hover_z=hover_z,
            dwell=self.dwell,
            on_transition=self._handle_transition,
        )

        # ── Placer SM ────────────────────────────────────────────────────────
        self._placer = _ArmFSM(
            role=ArmRole.PLACER,
            ctrl=placer_ctrl,
            box=box,
            box_prim_path=box_prim_path,
            pick_xyz=self._handover_xyz,     # PLACER picks from handover zone
            target_xyz=place_xyz,
            hover_z=hover_z,
            dwell=self.dwell,
            on_transition=self._handle_transition,
        )

        self.cycle_count = 0

    # ── Public ───────────────────────────────────────────────────────────────

    def tick(self, step: int) -> bool:
        """
        Tick both arms. Returns True when both complete a full handover cycle.

        Picker runs first each step; Placer checks the handover flag set by Picker.
        """
        picker_done = self._picker.tick(step, handover_ready=self._handover_ready)
        placer_done = self._placer.tick(step, handover_ready=self._handover_ready)

        # Update shared handover flag from Picker's internal state
        self._handover_ready = self._picker.deposited_at_handover

        if picker_done and placer_done:
            self.cycle_count += 1
            _log(f"✅ [step {step}] DualArmSM cycle {self.cycle_count} complete")
            # Reset both arms for next cycle
            self._picker.reset(step)
            self._placer.reset(step)
            self._handover_ready = False
            return True

        return False

    @property
    def picker_state(self) -> ArmState:
        return self._picker.state

    @property
    def placer_state(self) -> ArmState:
        return self._placer.state

    # ── Internal ─────────────────────────────────────────────────────────────

    def _handle_transition(self, role: ArmRole, old: ArmState, new: ArmState, step: int):
        _log(f"   [{step:>5}] DualArmSM [{role.name:6}]: {old.name} → {new.name}")
        if self._on_transition:
            self._on_transition(role, old, new, step)


# ─────────────────────────────────────────────────────────────────────────────
#  _ArmFSM — internal single-arm FSM used by DualArmSM
# ─────────────────────────────────────────────────────────────────────────────
class _ArmFSM:
    """
    Internal FSM for one arm inside DualArmSM.

    For the PICKER:  pick_xyz=conveyor, target_xyz=handover_zone
    For the PLACER:  pick_xyz=handover_zone, target_xyz=output_pile

    The handover-specific states (HANDOVER_DEPOSIT, HANDOVER_WAIT,
    HANDOVER_RETRIEVE) are handled here so DualArmSM's tick() stays clean.

    Not intended to be instantiated directly outside of DualArmSM.
    """

    def __init__(
        self,
        role: ArmRole,
        ctrl: "LulaController",
        box,
        box_prim_path: Optional[str],
        pick_xyz: np.ndarray,
        target_xyz: np.ndarray,
        hover_z: float,
        dwell: dict,
        on_transition: Callable,
    ):
        self.role           = role
        self.ctrl           = ctrl
        self._box           = box
        self._box_prim_path = box_prim_path
        self.pick_xyz       = pick_xyz
        self.target_xyz     = target_xyz
        self.hover_z        = hover_z
        self.dwell          = dwell
        self._on_transition = on_transition
        self._ee_quat       = _ee_quat_default()

        self.state                 = ArmState.IDLE
        self.state_entry_step      = 0
        self.deposited_at_handover = False   # PICKER sets True, DualArmSM reads it
        self.ik_fail_count         = 0

    def tick(self, step: int, handover_ready: bool = False) -> bool:
        n = step - self.state_entry_step

        if self.role == ArmRole.PICKER:
            return self._tick_picker(step, n)
        else:
            return self._tick_placer(step, n, handover_ready)

    def reset(self, step: int = 0):
        self.state                 = ArmState.IDLE
        self.state_entry_step      = step
        self.deposited_at_handover = False
        self.ik_fail_count         = 0

    # ── PICKER path: pick → lift → handover deposit → home ───────────────────

    def _tick_picker(self, step: int, n: int) -> bool:
        if self.state == ArmState.IDLE:
            self.ctrl.set_gripper(closed=False)
            self._go(ArmState.HOME, step)

        elif self.state == ArmState.HOME:
            self.ctrl.move_home()
            if n > self.dwell[ArmState.HOME]:
                self._go(ArmState.HOVER_PICK, step)

        elif self.state == ArmState.HOVER_PICK:
            self.ctrl.move_to(
                self.pick_xyz + np.array([0, 0, self.hover_z]),
                self._ee_quat, warm_hint=self.ctrl.WARM_PICK,
            )
            if n > self.dwell[ArmState.HOVER_PICK]:
                self._go(ArmState.DESCEND_PICK, step)

        elif self.state == ArmState.DESCEND_PICK:
            self.ctrl.move_to(self.pick_xyz, self._ee_quat, warm_hint=self.ctrl.WARM_PICK)
            if n > self.dwell[ArmState.DESCEND_PICK]:
                self._go(ArmState.GRASP, step)

        elif self.state == ArmState.GRASP:
            self.ctrl.set_gripper(closed=True)
            if n == 1 and self._box_prim_path:
                self.ctrl.attach_box(self._box_prim_path)
            if self._box:
                self.ctrl.move_box_with_ee(self._box)
            if n > self.dwell[ArmState.GRASP]:
                self._go(ArmState.LIFT, step)

        elif self.state == ArmState.LIFT:
            self.ctrl.move_to(
                self.pick_xyz + np.array([0, 0, self.hover_z]),
                self._ee_quat, warm_hint=self.ctrl.WARM_PICK,
            )
            if self._box:
                self.ctrl.move_box_with_ee(self._box)
            if n > self.dwell[ArmState.LIFT]:
                self._go(ArmState.HANDOVER_DEPOSIT, step)

        elif self.state == ArmState.HANDOVER_DEPOSIT:
            # Move to handover zone and release — signals Placer to proceed
            self.ctrl.move_to(
                self.target_xyz + np.array([0, 0, self.hover_z]),
                self._ee_quat,
            )
            if self._box:
                self.ctrl.move_box_with_ee(self._box)
            if n > self.dwell[ArmState.HANDOVER_DEPOSIT]:
                self.ctrl.set_gripper(closed=False)
                self.ctrl.detach_box()
                self.deposited_at_handover = True  # signal to DualArmSM
                _log(f"   [{step}] PICKER: handover deposit complete ✓")
                self._go(ArmState.HOME_FINAL, step)

        elif self.state == ArmState.HOME_FINAL:
            self.ctrl.move_home()
            if n > self.dwell[ArmState.HOME_FINAL]:
                self._go(ArmState.DONE, step)
                return True

        return False

    # ── PLACER path: wait → retrieve from handover → place → home ─────────

    def _tick_placer(self, step: int, n: int, handover_ready: bool) -> bool:
        if self.state == ArmState.IDLE:
            self._go(ArmState.HANDOVER_WAIT, step)

        elif self.state == ArmState.HANDOVER_WAIT:
            # Spin until PICKER signals deposit is done
            if n % 60 == 0:
                _log(f"   [{step}] PLACER: waiting for handover... (ready={handover_ready})")
            if handover_ready:
                self._go(ArmState.HANDOVER_RETRIEVE, step)
            elif n > self.dwell[ArmState.HANDOVER_WAIT]:
                # Timeout — log warning, stay waiting (fallback_handler will handle Day 6)
                _log(f"   ⚠️ [{step}] PLACER: handover wait timeout — Picker may have failed")
                self.state_entry_step = step  # reset wait timer

        elif self.state == ArmState.HANDOVER_RETRIEVE:
            self.ctrl.move_to(
                self.pick_xyz + np.array([0, 0, self.hover_z]),
                self._ee_quat,
            )
            if n > 80:
                self.ctrl.move_to(self.pick_xyz, self._ee_quat)
            if n > self.dwell[ArmState.HANDOVER_RETRIEVE]:
                self.ctrl.set_gripper(closed=True)
                if self._box_prim_path:
                    self.ctrl.attach_box(self._box_prim_path)
                self._go(ArmState.HOVER_PLACE, step)

        elif self.state == ArmState.HOVER_PLACE:
            self.ctrl.move_to(
                self.target_xyz + np.array([0, 0, self.hover_z]),
                self._ee_quat, warm_hint=self.ctrl.WARM_PLACE,
            )
            if self._box:
                self.ctrl.move_box_with_ee(self._box)
            if n > self.dwell[ArmState.HOVER_PLACE]:
                self._go(ArmState.DESCEND_PLACE, step)

        elif self.state == ArmState.DESCEND_PLACE:
            self.ctrl.move_to(self.target_xyz, self._ee_quat, warm_hint=self.ctrl.WARM_PLACE)
            if self._box:
                self.ctrl.move_box_with_ee(self._box)
            if n > self.dwell[ArmState.DESCEND_PLACE]:
                self._go(ArmState.RELEASE, step)

        elif self.state == ArmState.RELEASE:
            self.ctrl.set_gripper(closed=False)
            if n == 1:
                self.ctrl.detach_box()
            if n > self.dwell[ArmState.RELEASE]:
                self._go(ArmState.HOME_FINAL, step)

        elif self.state == ArmState.HOME_FINAL:
            self.ctrl.move_home()
            if n > self.dwell[ArmState.HOME_FINAL]:
                self._go(ArmState.DONE, step)
                return True

        return False

    def _go(self, new_state: ArmState, step: int):
        old = self.state
        self._on_transition(self.role, old, new_state, step)
        self.state            = new_state
        self.state_entry_step = step
