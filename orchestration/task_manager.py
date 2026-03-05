"""
orchestration/task_manager.py
------------------------------
Dual-arm task manager for the DR warehouse return pipeline.

Responsibilities
----------------
1. Scene construction  : builds the dual-arm warehouse layout in Isaac Sim
                         matching the DR spec (gravity conveyor → table → pile)
2. Arm loading         : loads two RM75-B URDFs at mirrored base positions,
                         creates one LulaController per arm
3. Cycle queue         : maintains a FIFO queue of incoming boxes; dispatches
                         each box to DualArmSM for pick→handover→place
4. Takt time pacing    : enforces a minimum inter-cycle gap so the downstream
                         placer never falls behind the upstream picker
5. Metrics collection  : records cycle time, handover latency, IK fail rate
                         per cycle; ready for benchmark_run.py (Day 7)
6. Shutdown            : graceful stop — finish in-flight cycle, then halt

Layout (top-down, units = metres)
----------------------------------

  ←1.5m→ ←──────── 3m conveyor ────────→ ←1.5m table→ ←1.5m pile→
  +------+----------------------------------+------------+-----------+
  |      | [BOX FLOW →→→→→→→→→→→→→→→→→→→] |            |  OUTPUT   |
  | LOAD |  ARM_A base (PICKER)             |  ARM_B     |   PILE    |
  |      |  x=+0.45  y= 0.0                |  base      |           |
  +------+----------------------------------+ x=-0.30    +-----------+
                                              y= 0.0
  World X: conveyor runs along +X
  Handover zone: x=0.0, y=0.0, z=0.55  (between the two arm bases)

Run (standalone demo, single full cycle):
  ~/isaacsim/python.sh scripts/run_demo.py

  Or inline:
  ~/isaacsim/python.sh -c "
import sys, os; sys.path.insert(0, os.path.expanduser('~/orchestration_sim'))
from isaacsim import SimulationApp          # SimulationApp MUST come first
sim_app = SimulationApp({'headless': True})
from orchestration.task_manager import TaskManager
TaskManager.run_demo(headless=True, n_cycles=2)
  "
"""

from __future__ import annotations

import os
import sys
import time
import logging
import dataclasses
from collections import deque
from typing import List, Optional, Tuple

import numpy as np

# ── Logger ────────────────────────────────────────────────────────────────────
logger = logging.getLogger("orchestration.task_manager")
logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")

def _log(msg: str):
    logger.info(msg)
    print(msg, flush=True)

# ── Repo root (for asset paths) ───────────────────────────────────────────────
REPO = os.path.expanduser("~/orchestration_sim")

# ─────────────────────────────────────────────────────────────────────────────
#  Scene geometry — DR warehouse spec
#  All dimensions in metres, matching the PDF layout diagram
# ─────────────────────────────────────────────────────────────────────────────

# Conveyor / table heights
CONVEYOR_H  = 0.40   # top surface of gravity conveyor (same as TABLE_H in script)
TABLE_H     = 0.40   # inspection table surface
PILE_H      = 0.10   # output pile surface (lower — stacked items)

# Box dimensions (phone-sized SKU per DR spec)
BOX_SIZE    = 0.12   # 12 cm cube
BOX_MASS    = 0.25   # kg

# ── Arm base positions ────────────────────────────────────────────────────────
# ARM_A (PICKER): sits beside the conveyor end, picks boxes off the belt
# ARM_B (PLACER): sits beside the inspection table, places to output pile
#
# Both arms face +X. Placed on opposite sides of handover zone.
# 750 mm reach → safe pick at x=+0.30 from base.
ARM_A_BASE_XYZ  = np.array([ 0.45,  0.0,  0.0])   # PICKER base
ARM_B_BASE_XYZ  = np.array([-0.30,  0.0,  0.0])   # PLACER base

# ── Key task positions (world frame) ─────────────────────────────────────────
# Confirmed IK-reachable from pick_place_rmpflow.py workspace_scan
CONVEYOR_PICK_XYZ  = np.array([ 0.30,  0.0,   0.50])  # ARM_A picks here
HANDOVER_XYZ       = np.array([ 0.0,   0.0,   0.55])  # neutral deposit/retrieve zone
OUTPUT_PLACE_XYZ   = np.array([-0.20,  0.0,   0.45])  # ARM_B places here (pile)
HOVER_Z            = 0.12

# ── Takt time ─────────────────────────────────────────────────────────────────
# Takt time = minimum sim steps between cycle starts.
# Set to 0 for max throughput; increase to pace the picker rate.
# At 60Hz sim, 300 steps ≈ 5 seconds per cycle.
DEFAULT_TAKT_STEPS = 0   # 0 = pipeline as fast as possible

# ── Prim paths ────────────────────────────────────────────────────────────────
PRIM_ARM_A  = "/World/ARM_A"
PRIM_ARM_B  = "/World/ARM_B"
PRIM_BOX    = "/World/return_box_{idx}"   # formatted per box


# ─────────────────────────────────────────────────────────────────────────────
#  CycleRecord — one row of metrics per completed cycle
# ─────────────────────────────────────────────────────────────────────────────
@dataclasses.dataclass
class CycleRecord:
    """
    Metrics recorded for one full pick→handover→place cycle.
    Written by TaskManager._record_cycle(), read by benchmark_run.py (Day 7).

    Fields
    ------
    cycle_id        : sequential cycle index (0-based)
    box_sku         : SKU tag from the box (e.g. "BACKPACK_01", "PHONE_03")
    start_step      : sim step when cycle began
    end_step        : sim step when placer returned HOME_FINAL
    handover_step   : sim step when picker set handover_ready=True
    ik_fails_picker : IK convergence failures from the picker arm this cycle
    ik_fails_placer : IK convergence failures from the placer arm this cycle
    fallback_count  : number of times fallback_handler triggered this cycle
    success         : True if box reached output pile within tolerance

    Derived metrics (computed as properties)
    -----------------------------------------
    cycle_steps         : end_step - start_step
    handover_latency    : handover_step - start_step  (picker half-cycle)
    placer_latency      : end_step - handover_step    (placer half-cycle)
    """
    cycle_id:        int
    box_sku:         str
    start_step:      int
    end_step:        int   = -1
    handover_step:   int   = -1
    ik_fails_picker: int   = 0
    ik_fails_placer: int   = 0
    fallback_count:  int   = 0
    success:         bool  = False

    @property
    def cycle_steps(self) -> int:
        return max(0, self.end_step - self.start_step)

    @property
    def handover_latency(self) -> int:
        if self.handover_step < 0:
            return -1
        return self.handover_step - self.start_step

    @property
    def placer_latency(self) -> int:
        if self.handover_step < 0 or self.end_step < 0:
            return -1
        return self.end_step - self.handover_step

    def summary(self) -> str:
        return (
            f"Cycle {self.cycle_id:>3} | SKU={self.box_sku:<14} | "
            f"steps={self.cycle_steps:>5} | "
            f"handover_lat={self.handover_latency:>4} | "
            f"placer_lat={self.placer_latency:>4} | "
            f"IK_fails=({self.ik_fails_picker},{self.ik_fails_placer}) | "
            f"fallbacks={self.fallback_count} | "
            f"{'✅' if self.success else '❌'}"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  BoxItem — one unit of work in the cycle queue
# ─────────────────────────────────────────────────────────────────────────────
@dataclasses.dataclass
class BoxItem:
    """
    One box to be processed.

    sku         : stock-keeping unit identifier (matches DR's 10-15 SKU set)
    pick_xyz    : where the arm should pick (defaults to CONVEYOR_PICK_XYZ)
    place_xyz   : where to place after handover (defaults to OUTPUT_PLACE_XYZ)
    prim_path   : USD prim path once spawned in Isaac Sim
    sim_object  : DynamicCuboid handle (set by TaskManager.spawn_box)
    """
    sku:        str
    pick_xyz:   np.ndarray = dataclasses.field(
                    default_factory=lambda: CONVEYOR_PICK_XYZ.copy())
    place_xyz:  np.ndarray = dataclasses.field(
                    default_factory=lambda: OUTPUT_PLACE_XYZ.copy())
    prim_path:  str        = ""
    sim_object: object     = None   # DynamicCuboid, set after spawn


# ─────────────────────────────────────────────────────────────────────────────
#  SceneBuilder — static helpers to construct the warehouse scene
# ─────────────────────────────────────────────────────────────────────────────
class SceneBuilder:
    """
    Builds the dual-arm warehouse scene in Isaac Sim.

    Geometry matches the DR warehouse return pipeline PDF:
      - Gravity conveyor slab (3m × 0.7m, grey)
      - Inspection table (1.5m × 0.7m, brown)
      - Output pile platform (1.5m × 0.7m, dark)
      - Handover zone marker (visual sphere only, cyan)
      - Two VisualSphere markers for pick and place targets

    Call build() once after World.reset().
    """

    @staticmethod
    def build(world) -> None:
        """Add all static geometry to the world scene."""
        from isaacsim.core.api.objects import FixedCuboid, VisualSphere

        # ── Gravity conveyor (steps 1-3 in DR pipeline) ───────────────────
        world.scene.add(FixedCuboid(
            prim_path="/World/conveyor",
            name="conveyor",
            position=[ 1.95, 0.0, CONVEYOR_H / 2],   # 3m slab, centred at x=1.95
            size=1.0,
            scale=[3.0, 0.7, CONVEYOR_H],
            color=np.array([0.35, 0.35, 0.35]),
        ))

        # ── Inspection table (step 5 — SKU check zone) ────────────────────
        world.scene.add(FixedCuboid(
            prim_path="/World/inspection_table",
            name="inspection_table",
            position=[ 0.75, 0.0, TABLE_H / 2],
            size=1.0,
            scale=[1.5, 0.7, TABLE_H],
            color=np.array([0.55, 0.38, 0.18]),
        ))

        # ── Output pile platform (step 6) ─────────────────────────────────
        world.scene.add(FixedCuboid(
            prim_path="/World/output_pile",
            name="output_pile",
            position=[-0.75, 0.0, PILE_H / 2],
            size=1.0,
            scale=[1.5, 0.7, PILE_H],
            color=np.array([0.20, 0.20, 0.22]),
        ))

        # ── Handover zone — visual marker only (cyan sphere) ──────────────
        # This is NOT a physics object — just helps with visual debugging.
        # Concept: "Handover Zone" = shared neutral space, neither arm's
        # exclusive territory. ARM_A deposits here; ARM_B retrieves from here.
        world.scene.add(VisualSphere(
            prim_path="/World/handover_marker",
            name="handover_marker",
            position=HANDOVER_XYZ.tolist(),
            radius=0.04,
            color=np.array([0.0, 0.85, 0.85]),
        ))

        # ── Pick target marker (green) ────────────────────────────────────
        world.scene.add(VisualSphere(
            prim_path="/World/pick_marker",
            name="pick_marker",
            position=CONVEYOR_PICK_XYZ.tolist(),
            radius=0.02,
            color=np.array([0.1, 0.9, 0.1]),
        ))

        # ── Place target marker (red) ─────────────────────────────────────
        world.scene.add(VisualSphere(
            prim_path="/World/place_marker",
            name="place_marker",
            position=OUTPUT_PLACE_XYZ.tolist(),
            radius=0.02,
            color=np.array([0.9, 0.15, 0.15]),
        ))

        _log("✅ SceneBuilder: warehouse scene constructed")
        _log(f"   Conveyor pick  → {CONVEYOR_PICK_XYZ}")
        _log(f"   Handover zone  → {HANDOVER_XYZ}  (cyan marker)")
        _log(f"   Output pile    → {OUTPUT_PLACE_XYZ}")

    @staticmethod
    def spawn_box(world, item: BoxItem, box_idx: int):
        """
        Spawn a DynamicCuboid for `item` at its pick_xyz position.
        Assigns item.prim_path and item.sim_object in-place.
        """
        from isaacsim.core.api.objects import DynamicCuboid

        prim = PRIM_BOX.format(idx=box_idx)
        item.prim_path = prim

        obj = world.scene.add(DynamicCuboid(
            prim_path=prim,
            name=f"return_box_{box_idx}",
            position=item.pick_xyz.tolist(),
            size=BOX_SIZE,
            color=np.array([0.95, 0.75, 0.1]),
            mass=BOX_MASS,
        ))
        item.sim_object = obj
        _log(f"   📦 Spawned box [{box_idx}] SKU={item.sku} at {np.round(item.pick_xyz, 3)}")
        return obj


# ─────────────────────────────────────────────────────────────────────────────
#  ArmLoader — loads two RM75-B arms into the scene
# ─────────────────────────────────────────────────────────────────────────────
class ArmLoader:
    """
    Loads two RM75-B URDF instances into Isaac Sim and returns
    one (SingleArticulation, LulaController) pair per arm.

    ARM_A (PICKER) is placed at ARM_A_BASE_XYZ.
    ARM_B (PLACER) is placed at ARM_B_BASE_XYZ, rotated 180° around Z
    so it faces the handover zone from the other side.

    New concept: prim_path offset
    ------------------------------
    Isaac Sim identifies each robot by its USD prim path ("/World/ARM_A").
    When loading two identical URDFs we must give them different prim paths
    or the second import overwrites the first.
    """

    URDF_PATH = os.path.join(REPO, "assets/robots/rm75b/rm75b_local.urdf")
    USD_PATH  = os.path.join(REPO, "assets/robots/rm75b/rm75b.usd")

    @classmethod
    def load_both(cls, world) -> Tuple:
        """
        Returns (arm_a_prim, arm_b_prim) — USD prim path strings.
        Call after world.reset() and before arm.initialize().
        """
        arm_a_prim = cls._load_one(PRIM_ARM_A, ARM_A_BASE_XYZ, yaw_deg=0.0)
        arm_b_prim = cls._load_one(PRIM_ARM_B, ARM_B_BASE_XYZ, yaw_deg=180.0)
        return arm_a_prim, arm_b_prim

    @classmethod
    def _load_one(cls, prim_path: str, base_xyz: np.ndarray, yaw_deg: float) -> str:
        from isaacsim.asset.importer.urdf import _urdf
        from pxr import UsdGeom, Gf, Sdf
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        prim = None
        try:
            prim = stage.DefinePrim(prim_path, "Xform")
            prim.GetReferences().AddReference(cls.USD_PATH)
        except Exception:
            ui  = _urdf.acquire_urdf_interface()
            cfg = _urdf.ImportConfig()
            cfg.merge_fixed_joints             = False
            cfg.fix_base                       = True
            cfg.default_drive_type             = _urdf.UrdfJointTargetType.JOINT_DRIVE_POSITION
            cfg.default_drive_strength         = 8e3
            cfg.default_position_drive_damping = 8e2
            asset_root  = os.path.dirname(cls.URDF_PATH)
            asset_name  = os.path.basename(cls.URDF_PATH)
            robot       = ui.parse_urdf(asset_root, asset_name, cfg)
            stage_url   = stage.GetRootLayer().identifier
            created_prim_path = ui.import_robot(asset_root, asset_name, robot, cfg, stage_url)
            prim  = stage.GetPrimAtPath(created_prim_path)
            if not prim.IsValid():
                raise RuntimeError(f"URDF import returned '{created_prim_path}', but prim not found")
            if created_prim_path != prim_path:
                import omni.kit.commands
                existing = stage.GetPrimAtPath(prim_path)
                if existing and existing.IsValid():
                    omni.kit.commands.execute("DeletePrims", paths=[prim_path])
                omni.kit.commands.execute("MovePrim", path_from=created_prim_path, path_to=prim_path)
                prim = stage.GetPrimAtPath(prim_path)
                if not prim.IsValid():
                    raise RuntimeError(f"Failed to move prim to {prim_path}")

        xform = UsdGeom.Xformable(prim)
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(*base_xyz.tolist()))
        xform.AddRotateZOp().Set(float(yaw_deg))

        _log(f"✅ ArmLoader: {prim_path} loaded at {base_xyz}, yaw={yaw_deg}°")
        return prim_path

    @classmethod
    def make_controllers(cls, arm_a, arm_b) -> Tuple:
        """
        Given two initialized SingleArticulation objects,
        return (ctrl_a, ctrl_b) LulaController instances.

        WARM_PLACE for ARM_B is mirrored (negated Y component)
        because ARM_B faces the opposite direction.
        """
        # Import here to avoid circular dependency with scripts/
        sys.path.insert(0, os.path.join(REPO, "scripts"))
        from pick_place_rmpflow import LulaController

        ctrl_a = LulaController(arm=arm_a, urdf_path=cls.URDF_PATH, ee_frame="link_7")
        ctrl_b = LulaController(arm=arm_b, urdf_path=cls.URDF_PATH, ee_frame="link_7")

        # Override WARM_PLACE for ARM_B — its workspace is mirrored
        # WARM_PICK for ARM_B is the handover zone, use a new seed
        ctrl_b.WARM_PICK  = np.array([ 0.0,   0.23,  0.0,  0.664, 0.0,  1.677, 0.0])
        ctrl_b.WARM_PLACE = np.array([ 0.321, 0.414, 0.446, 1.041, 1.301, 1.271, 0.0])

        _log("✅ ArmLoader: LulaControllers initialized for ARM_A and ARM_B")
        return ctrl_a, ctrl_b


# ─────────────────────────────────────────────────────────────────────────────
#  TaskManager — the top-level orchestrator
# ─────────────────────────────────────────────────────────────────────────────
class TaskManager:
    """
    Top-level orchestrator for the dual-arm warehouse return pipeline.

    Lifecycle
    ---------
    1. __init__    : store config, create empty queue + metrics list
    2. setup()     : build scene, load arms, create DualArmSM
    3. enqueue()   : add BoxItems to the cycle queue
    4. tick(step)  : call every sim step — dispatches next box when ready,
                     ticks DualArmSM, records metrics on completion
    5. shutdown()  : graceful stop after current cycle

    Takt time pacing
    ----------------
    Concept: "Takt time" (from German "beat") is the required rate of
    production — how many seconds per unit. Here we implement it as a
    minimum number of sim steps between cycle *starts* (not completions).
    This prevents the picker from running ahead of the placer and
    stacking up at the handover zone.

    Example usage:
        tm = TaskManager(takt_steps=300)  # 5s between cycle starts at 60Hz
        tm.setup(world)
        tm.enqueue([BoxItem("PHONE_01"), BoxItem("BACKPACK_02")])
        # then call tm.tick(step) in the sim loop
    """

    def __init__(
        self,
        takt_steps: int = DEFAULT_TAKT_STEPS,
        max_cycles: int = 0,        # 0 = unlimited
        headless: bool  = False,
    ):
        self.takt_steps  = takt_steps
        self.max_cycles  = max_cycles
        self.headless    = headless

        # Internal state
        self._queue:       deque[BoxItem]   = deque()
        self._metrics:     List[CycleRecord] = []
        self._active_item: Optional[BoxItem] = None
        self._active_record: Optional[CycleRecord] = None
        self._dual_sm      = None    # DualArmSM, created in setup()
        self._last_cycle_start_step: int = -999999
        self._cycle_id:    int = 0
        self._box_idx:     int = 0
        self._running:     bool = False
        self._world        = None

        # Scene + arm handles (set by setup())
        self._ctrl_a = None
        self._ctrl_b = None

    # ── Public API ────────────────────────────────────────────────────────────

    def setup(self, world) -> None:
        """
        Build scene and load arms. Must be called after world.reset()
        and before the first tick().
        """
        from isaacsim.core.prims import SingleArticulation
        from orchestration.state_machine import DualArmSM

        self._world = world

        # 1. Build static geometry
        SceneBuilder.build(world)
        world.reset()

        # 2. Load both arms
        world.step(render=False)
        arm_a_prim, arm_b_prim = ArmLoader.load_both(world)
        world.step(render=False)

        arm_a = SingleArticulation(prim_path=arm_a_prim)
        arm_b = SingleArticulation(prim_path=arm_b_prim)
        world.step(render=False)
        arm_a.initialize()
        arm_b.initialize()
        _log(f"   ARM_A DOFs ({arm_a.num_dof}): {arm_a.dof_names}")
        _log(f"   ARM_B DOFs ({arm_b.num_dof}): {arm_b.dof_names}")

        # 3. Create controllers
        self._ctrl_a, self._ctrl_b = ArmLoader.make_controllers(arm_a, arm_b)

        # 4. Create DualArmSM (no box yet — will be set when first box dispatched)
        self._dual_sm = DualArmSM(
            picker_ctrl   = self._ctrl_a,
            placer_ctrl   = self._ctrl_b,
            box           = None,
            box_prim_path = None,
            handover_xyz  = HANDOVER_XYZ.copy(),
            place_xyz     = OUTPUT_PLACE_XYZ.copy(),
            hover_z       = HOVER_Z,
            on_transition = self._on_arm_transition,
        )

        self._running = True
        _log("✅ TaskManager.setup() complete — ready to receive boxes")

    def enqueue(self, items) -> None:
        """
        Add one BoxItem or a list of BoxItems to the cycle queue.
        Can be called at any time, even while a cycle is running.
        """
        if isinstance(items, BoxItem):
            items = [items]
        for item in items:
            self._queue.append(item)
        _log(f"   📋 Queue: +{len(items)} items → total={len(self._queue)}")

    def tick(self, step: int) -> bool:
        """
        Called every simulation step.

        Returns True if all queued items have been processed (queue empty
        and no cycle in flight). Returns False while work remains.

        Logic
        -----
        1. If no cycle in flight and queue non-empty and takt gate passed
           → dispatch next box
        2. Tick the active DualArmSM
        3. If DualArmSM returns done → record metrics, clear active item
        4. If max_cycles reached → signal shutdown
        """
        if not self._running:
            return True

        # ── Dispatch next box if ready ─────────────────────────────────────
        if self._active_item is None and self._queue:
            takt_ok = (step - self._last_cycle_start_step) >= self.takt_steps
            if takt_ok:
                self._dispatch_next(step)

        # ── Tick DualArmSM if a cycle is active ───────────────────────────
        if self._active_item is not None and self._dual_sm is not None:
            # Update metrics: handover latency
            if (self._active_record and
                self._active_record.handover_step < 0 and
                self._dual_sm._handover_ready):
                self._active_record.handover_step = step

            done = self._dual_sm.tick(step)

            # Update per-step IK fail counts from sub-FSMs
            if self._active_record:
                self._active_record.ik_fails_picker = self._dual_sm._picker.ik_fail_count
                self._active_record.ik_fails_placer = self._dual_sm._placer.ik_fail_count

            if done:
                self._complete_cycle(step)

        # ── Check if all work is done ──────────────────────────────────────
        all_done = (
            self._active_item is None and
            len(self._queue) == 0 and
            (self.max_cycles == 0 or self._cycle_id >= self.max_cycles)
        )

        if all_done:
            _log(f"\n🏁 TaskManager: all {self._cycle_id} cycles complete")
            self._print_metrics_summary()
            self._running = False
            return True

        return False

    def shutdown(self) -> None:
        """Signal graceful stop. Current cycle finishes, then halts."""
        _log("⚠️  TaskManager.shutdown() called — will stop after current cycle")
        self._queue.clear()

    @property
    def metrics(self) -> List[CycleRecord]:
        """Read-only access to completed cycle records (for benchmark_run.py)."""
        return list(self._metrics)

    @property
    def is_running(self) -> bool:
        return self._running

    # ── Internal ──────────────────────────────────────────────────────────────

    def _dispatch_next(self, step: int) -> None:
        """
        Pop the next BoxItem from the queue, spawn it in the scene,
        wire it into DualArmSM, and start the cycle.
        """
        item = self._queue.popleft()
        self._box_idx += 1

        # Spawn the box in Isaac Sim
        SceneBuilder.spawn_box(self._world, item, self._box_idx)

        # Wire into DualArmSM
        self._dual_sm._picker._box           = item.sim_object
        self._dual_sm._picker._box_prim_path = item.prim_path
        self._dual_sm._picker.pick_xyz       = item.pick_xyz
        self._dual_sm._placer._box           = item.sim_object
        self._dual_sm._placer._box_prim_path = item.prim_path
        self._dual_sm._placer.target_xyz     = item.place_xyz

        # Reset both arm FSMs for the new cycle
        self._dual_sm._picker.reset(step)
        self._dual_sm._placer.reset(step)
        self._dual_sm._handover_ready = False

        # Create metrics record
        self._active_item   = item
        self._active_record = CycleRecord(
            cycle_id   = self._cycle_id,
            box_sku    = item.sku,
            start_step = step,
        )
        self._last_cycle_start_step = step

        _log(f"\n▶  Cycle {self._cycle_id} START | SKU={item.sku} | step={step}")

    def _complete_cycle(self, step: int) -> None:
        """Called when DualArmSM signals a cycle is complete."""
        if self._active_record:
            rec = self._active_record
            rec.end_step = step

            # Check success: box should be near OUTPUT_PLACE_XYZ
            if self._active_item and self._active_item.sim_object:
                box_pos = self._active_item.sim_object.get_world_pose()[0]
                dist    = np.linalg.norm(box_pos - self._active_item.place_xyz)
                rec.success = dist < 0.15   # 15cm tolerance
            else:
                rec.success = True  # no physics check possible

            self._metrics.append(rec)
            _log(f"◼  Cycle {self._cycle_id} END   | {rec.summary()}")

        self._cycle_id   += 1
        self._active_item = None
        self._active_record = None

    def _on_arm_transition(self, role, old_state, new_state, step: int) -> None:
        """
        Callback from DualArmSM on every state transition.
        Used here to log and hook into metrics (Day 6 will add fallback here).
        """
        # Currently just logs — fallback_handler.py will patch this in Day 6
        pass   # transitions already logged by DualArmSM itself

    def _print_metrics_summary(self) -> None:
        if not self._metrics:
            return
        _log("\n── Metrics Summary ─────────────────────────────────────────")
        for rec in self._metrics:
            _log(f"  {rec.summary()}")

        cycle_steps   = [r.cycle_steps      for r in self._metrics if r.end_step > 0]
        handover_lats = [r.handover_latency  for r in self._metrics if r.handover_latency > 0]
        success_rate  = sum(r.success for r in self._metrics) / len(self._metrics)

        if cycle_steps:
            _log(f"\n  avg cycle steps    : {np.mean(cycle_steps):.1f}")
        if handover_lats:
            _log(f"  avg handover lat   : {np.mean(handover_lats):.1f} steps")
        _log(f"  success rate       : {success_rate*100:.1f}%")
        _log(f"  total IK fails     : "
             f"{sum(r.ik_fails_picker + r.ik_fails_placer for r in self._metrics)}")
        _log("────────────────────────────────────────────────────────────\n")

    # ── Static convenience: run a quick demo ──────────────────────────────────

    @staticmethod
    def run_demo(headless: bool = False, n_cycles: int = 3) -> None:
        """
        Self-contained demo: boot Isaac Sim, run n_cycles, print metrics.
        Mirrors the structure of pick_place_rmpflow.py main() for familiarity.

        Usage:
            ~/isaacsim/python.sh -c "
            import sys; sys.path.insert(0, '~/orchestration_sim')
            from orchestration.task_manager import TaskManager
            TaskManager.run_demo(headless=True, n_cycles=2)
            "
        """
        from isaacsim import SimulationApp
        sim_app = SimulationApp({"headless": headless, "width": 1280, "height": 720})

        from isaacsim.core.api import World

        world = World(stage_units_in_meters=1.0)
        world.scene.add_default_ground_plane()

        # DR SKU set: backpack (largest), phones, water bottles
        sku_names = [
            "BACKPACK_01", "PHONE_02", "PHONE_03",
            "BOTTLE_04",   "PHONE_05", "BACKPACK_06",
        ]

        tm = TaskManager(takt_steps=300, max_cycles=n_cycles, headless=headless)
        tm.setup(world)
        tm.enqueue([BoxItem(sku=sku_names[i % len(sku_names)]) for i in range(n_cycles)])

        _log(f"\n🚀 TaskManager demo — {n_cycles} cycles, takt={tm.takt_steps} steps\n")

        step = 0
        MAX_STEPS = n_cycles * 2000   # generous upper bound
        t0 = time.time()

        while sim_app.is_running() and step < MAX_STEPS:
            world.step(render=not headless)
            done = tm.tick(step)
            if done:
                break
            step += 1

        elapsed = time.time() - t0
        _log(f"\n⏱  Wall time: {elapsed:.1f}s over {step} steps")

        if not headless:
            _log("GUI open — close window to exit")
            while sim_app.is_running():
                world.step(render=True)

        sim_app.close()
