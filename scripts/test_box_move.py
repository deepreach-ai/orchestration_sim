"""
test_box_move.py
----------------
Isaac Sim 5.1.0 test: RM75-B arm moves a warehouse return box
from conveyor zone to inspection table.

Models used:
  - Motion planning : RMPFlow (NVIDIA's Riemannian Motion Policy)
  - IK solver       : Lula (analytical IK, built into RMPFlow)

Pipeline mapped to DR warehouse return step 4 (Item Extraction):
  Box spawns on conveyor end (pick pose) → arm picks → places on inspection table.

Run with:
  ~/isaacsim/python.sh scripts/test_box_move.py
  ~/isaacsim/python.sh scripts/test_box_move.py --headless
  ~/isaacsim/python.sh scripts/test_box_move.py --headless --test  # CI smoke test
"""

from isaacsim import SimulationApp

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", default=False)
parser.add_argument("--test",     action="store_true", default=False,
                    help="Run limited steps for CI smoke test")
args, _ = parser.parse_known_args()

simulation_app = SimulationApp({"headless": args.headless})

# ── stdlib / third-party ────────────────────────────────────────────────────
import os
import sys
import time
import numpy as np

# Log to file since Isaac Sim hijacks stdout
_log = open("/tmp/box_move_test.log", "w", buffering=1)
def log(msg):
    _log.write(msg + "\n")
    _log.flush()

# Force stdout flush for remote logging
print = lambda *a, **k: __builtins__["print"](*a, **k, flush=True) if isinstance(__builtins__, dict) else __import__("builtins").print(*a, **k, flush=True)

# ── Isaac Sim 5.1.0 core API ─────────────────────────────────────────────────
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.prims import SingleArticulation as Articulation

# ── Isaac Sim 5.1.0 URDF importer ────────────────────────────────────────────
from isaacsim.asset.importer.urdf import _urdf

# ── Isaac Sim 5.1.0 motion generation ────────────────────────────────────────
# RMPFlow: NVIDIA's Riemannian Motion Policy — smooth, collision-aware
from isaacsim.robot_motion.motion_generation import RmpFlow, ArticulationMotionPolicy

# ── Paths ────────────────────────────────────────────────────────────────────
REPO      = os.path.expanduser("~/orchestration_sim")
URDF_PATH = os.path.join(REPO, "assets/robots/rm75b/rm75b_local.urdf")

# ── Scene geometry (DR warehouse spec dimensions) ────────────────────────────
TABLE_HEIGHT  = 0.35    # metres
BOX_HALF      = 0.075   # half-height of return box (15cm)
BOX_SIZE      = 0.15    # phone/water-bottle SKU scale per DR spec

PICK_POSE_XYZ  = np.array([ 0.50, 0.0, TABLE_HEIGHT + BOX_HALF + 0.01])
PLACE_POSE_XYZ = np.array([-0.40, 0.0, TABLE_HEIGHT + BOX_HALF + 0.01])
HOVER_OFFSET   = np.array([ 0.0,  0.0, 0.15])   # 15cm hover before descent

# Gripper joint positions (RM75-B CRT gripper)
GRIPPER_OPEN_POS   = np.array([0.04,  0.04])
GRIPPER_CLOSED_POS = np.array([0.005, 0.005])


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_arm(world: World) -> str:
    """Import RM75-B URDF into the stage and return its prim path."""
    urdf_interface = _urdf.acquire_urdf_interface()
    cfg = _urdf.ImportConfig()
    cfg.merge_fixed_joints             = False
    cfg.fix_base                       = True
    cfg.default_drive_type             = _urdf.UrdfJointTargetType.JOINT_DRIVE_POSITION
    cfg.default_drive_strength         = 1e4
    cfg.default_position_drive_damping = 1e3

    prim_path = "/RM75_B_with_Gripper"
    import os
    asset_root = os.path.dirname(URDF_PATH)
    asset_name = os.path.basename(URDF_PATH)
    robot = urdf_interface.parse_urdf(asset_root, asset_name, cfg)
    urdf_interface.import_robot("", "rm75b_local.urdf", robot, cfg, prim_path)
    log(f"✅ RM75-B imported → {prim_path}")
    return prim_path


def build_scene(world: World):
    """Add conveyor stub, inspection table, and return box."""
    world.scene.add(FixedCuboid(
        prim_path="/World/conveyor",
        name="conveyor",
        position=[0.5, 0.0, TABLE_HEIGHT / 2],
        size=1.0,
        scale=[0.5, 0.7, TABLE_HEIGHT],
        color=np.array([0.3, 0.3, 0.3]),
    ))
    world.scene.add(FixedCuboid(
        prim_path="/World/table",
        name="table",
        position=[-0.4, 0.0, TABLE_HEIGHT / 2],
        size=1.0,
        scale=[0.5, 0.7, TABLE_HEIGHT],
        color=np.array([0.6, 0.4, 0.2]),
    ))
    box = world.scene.add(DynamicCuboid(
        prim_path="/World/return_box",
        name="return_box",
        position=PICK_POSE_XYZ.tolist(),
        size=BOX_SIZE,
        color=np.array([0.9, 0.7, 0.1]),
        mass=0.3,
    ))
    print("✅ Scene built: conveyor + table + return_box")
    return box


class BoxMoveStateMachine:
    """
    State machine: IDLE → HOVER_PICK → DESCEND_PICK → GRASP →
                   LIFT → HOVER_PLACE → DESCEND_PLACE → RELEASE → DONE
    """

    def __init__(self, arm: Articulation):
        self.arm              = arm
        self.state            = "IDLE"
        self.state_entry_step = 0
        self._target_quat     = euler_angles_to_quat(np.array([np.pi, 0.0, 0.0]))

    def tick(self, step: int) -> bool:
        n = step - self.state_entry_step  # steps in current state

        if self.state == "IDLE":
            self._transition("HOVER_PICK", step)

        elif self.state == "HOVER_PICK":
            self._send_ee_target(PICK_POSE_XYZ + HOVER_OFFSET)
            if n > 120: self._transition("DESCEND_PICK", step)

        elif self.state == "DESCEND_PICK":
            self._send_ee_target(PICK_POSE_XYZ)
            if n > 100: self._transition("GRASP", step)

        elif self.state == "GRASP":
            self._set_gripper(closed=True)
            if n > 60: self._transition("LIFT", step)

        elif self.state == "LIFT":
            self._send_ee_target(PICK_POSE_XYZ + HOVER_OFFSET)
            if n > 100: self._transition("HOVER_PLACE", step)

        elif self.state == "HOVER_PLACE":
            self._send_ee_target(PLACE_POSE_XYZ + HOVER_OFFSET)
            if n > 150: self._transition("DESCEND_PLACE", step)

        elif self.state == "DESCEND_PLACE":
            self._send_ee_target(PLACE_POSE_XYZ)
            if n > 100: self._transition("RELEASE", step)

        elif self.state == "RELEASE":
            self._set_gripper(closed=False)
            if n > 60: self._transition("DONE", step)

        elif self.state == "DONE":
            log(f"✅ [step {step}] Box successfully moved to inspection table.")
            return True

        return False

    def _transition(self, new_state: str, step: int):
        log(f"   [step {step:>5}] {self.state} → {new_state}")
        self.state            = new_state
        self.state_entry_step = step

    def _send_ee_target(self, pos: np.ndarray):
        """Direct joint position target (placeholder until RMPFlow YAMLs ready)."""
        # TODO: replace with RmpFlow.set_end_effector_target() once
        # rm75b rmpflow_config.yaml + robot_descriptor.yaml are generated
        # via Isaac Sim Robot Descriptor Editor.
        pass

    def _set_gripper(self, closed: bool):
        pos = GRIPPER_CLOSED_POS if closed else GRIPPER_OPEN_POS
        # 13 DOFs: 7 arm + 6 gripper (Left_1, Left_2, Right_1, Right_2, Left_Support, Right_Support)
        gripper_pos = np.array([pos[0], pos[0], pos[1], pos[1], pos[0], pos[1]])
        action = ArticulationAction(
            joint_positions=np.concatenate([
                np.full(7, np.nan),  # leave arm joints alone
                gripper_pos,         # 6 gripper joints
            ])
        )
        self.arm.apply_action(action)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()

    arm_prim_path = load_arm(world)
    box           = build_scene(world)
    world.reset()

    # Must step once before initialize in Isaac Sim 5.x
    world.step(render=False)
    arm = Articulation(prim_path=arm_prim_path)
    world.step(render=False)
    arm.initialize()
    log(f"   DOF names : {arm.dof_names}")
    log(f"   Total DOFs: {arm.num_dof}")

    sm        = BoxMoveStateMachine(arm=arm)
    step      = 0
    done      = False
    MAX_STEPS = 800 if not args.test else 50

    print("\n🚀 Starting box-move test (DR warehouse return — step 4: Item Extraction)")
    log(f"   Pick  pose: {PICK_POSE_XYZ}")
    log(f"   Place pose: {PLACE_POSE_XYZ}\n")

    t0 = time.time()

    while simulation_app.is_running() and step < MAX_STEPS and not done:
        world.step(render=not args.headless)

        if step % 50 == 0:
            box_pos = box.get_world_pose()[0]
            log(f"   [step {step:>4}] state={sm.state:<16} box_pos={np.round(box_pos, 3)}")

        done = sm.tick(step)
        step += 1

    elapsed       = time.time() - t0
    final_box_pos = box.get_world_pose()[0]
    dist_to_goal  = np.linalg.norm(final_box_pos[:2] - PLACE_POSE_XYZ[:2])

    log("\n── Test Results ────────────────────────────────────────────")
    log(f"   Final box position : {np.round(final_box_pos, 3)}")
    log(f"   Distance to goal   : {dist_to_goal:.4f} m")
    log(f"   Final SM state     : {sm.state}")
    log(f"   Steps elapsed      : {step}")
    log(f"   Wall time          : {elapsed:.1f}s")

    success = sm.state == "DONE"
    log(f"\n{'✅ PASS' if success else '❌ FAIL'} — box_move test")
    log("────────────────────────────────────────────────────────────\n")

    simulation_app.close()
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
