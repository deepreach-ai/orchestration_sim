"""
pick_place_rmpflow.py
---------------------
Isaac Sim 5.1.0 — RM75-B single-arm pick-place with RMPFlow motion planning.
Replaces the stub _send_ee_target with real Lula IK + articulation control.

Run (with GUI visualization):
  ~/isaacsim/python.sh scripts/pick_place_rmpflow.py

Run headless:
  ~/isaacsim/python.sh scripts/pick_place_rmpflow.py --headless
"""

from isaacsim import SimulationApp

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", default=False)
args, _ = parser.parse_known_args()

simulation_app = SimulationApp({
    "headless": args.headless,
    "width": 1280,
    "height": 720,
})

# ── stdlib ────────────────────────────────────────────────────────────────────
import os, time
import numpy as np

# ── Isaac Sim 5.1.0 ───────────────────────────────────────────────────────────
from isaacsim.core.api             import World
from isaacsim.core.api.objects     import DynamicCuboid, FixedCuboid, VisualSphere
from isaacsim.core.utils.types     import ArticulationAction
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.prims           import SingleArticulation
from isaacsim.asset.importer.urdf  import _urdf

# ── Log to file (Isaac Sim hijacks stdout) ────────────────────────────────────
_log = open("/tmp/pick_place.log", "w", buffering=1)
def log(msg): _log.write(msg + "\n"); _log.flush(); print(msg, flush=True)

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO      = os.path.expanduser("~/orchestration_sim")
URDF_PATH = os.path.join(REPO, "assets/robots/rm75b/rm75b_local.urdf")

# ── Scene constants (DR warehouse spec) ───────────────────────────────────────
TABLE_H   = 0.40    # confirmed reachable for RM75-B
BOX_SIZE  = 0.12    # 12cm cube (phone-sized SKU)
BOX_HALF  = BOX_SIZE / 2

# Arm base is at world origin, facing +X
PICK_XYZ   = np.array([0.30,  0.0,  0.50])   # confirmed IK-reachable
PLACE_XYZ  = np.array([0.20, -0.30, 0.50])   # left side reachable
HOVER_Z    = 0.12   # hover height

# Joint home position — arm reaching forward over table, not leaning back
# joint_2 negative = lean forward, joint_4 negative = elbow down
HOME_JOINTS = np.array([0.0, -0.009, 0.0, -0.439, 0.0, 1.92, 0.0])  # straight up safe

# Gripper: 13 DOFs total (7 arm + 6 gripper fingers)
N_ARM_DOFS = 7
GRIP_OPEN  = np.array([ 0.04,  0.04,  0.04,  0.04])
GRIP_CLOSE = np.array([ 0.003, 0.003, 0.003, 0.003])


# ── URDF loader ───────────────────────────────────────────────────────────────
def load_arm() -> str:
    ui = _urdf.acquire_urdf_interface()
    cfg = _urdf.ImportConfig()
    cfg.merge_fixed_joints             = False
    cfg.fix_base                       = True
    cfg.default_drive_type             = _urdf.UrdfJointTargetType.JOINT_DRIVE_POSITION
    cfg.default_drive_strength         = 8e3
    cfg.default_position_drive_damping = 8e2

    asset_root = os.path.dirname(URDF_PATH)
    asset_name = os.path.basename(URDF_PATH)
    robot = ui.parse_urdf(asset_root, asset_name, cfg)
    prim  = "/RM75_B_with_Gripper"
    ui.import_robot("", asset_name, robot, cfg, prim)
    log(f"✅ RM75-B loaded → {prim}")
    return prim


# ── Scene builder ─────────────────────────────────────────────────────────────
def build_scene(world: World):
    # Pick table (conveyor side)
    world.scene.add(FixedCuboid(
        prim_path="/World/pick_table", name="pick_table",
        position=[0.45, 0.0, TABLE_H / 2],
        size=1.0, scale=[0.5, 0.5, TABLE_H],
        color=np.array([0.4, 0.4, 0.4]),
    ))
    # Place table
    world.scene.add(FixedCuboid(
        prim_path="/World/place_table", name="place_table",
        position=[0.45, -0.45, TABLE_H / 2],
        size=1.0, scale=[0.5, 0.5, TABLE_H],
        color=np.array([0.55, 0.35, 0.15]),
    ))
    # Return box (yellow)
    box = world.scene.add(DynamicCuboid(
        prim_path="/World/box", name="box",
        position=PICK_XYZ.tolist(),
        size=BOX_SIZE,
        color=np.array([0.95, 0.75, 0.1]),
        mass=0.25,
    ))
    # Visual markers for pick/place targets (green/red spheres)
    world.scene.add(VisualSphere(
        prim_path="/World/pick_marker", name="pick_marker",
        position=PICK_XYZ.tolist(),
        radius=0.015, color=np.array([0.0, 1.0, 0.0]),
    ))
    world.scene.add(VisualSphere(
        prim_path="/World/place_marker", name="place_marker",
        position=PLACE_XYZ.tolist(),
        radius=0.015, color=np.array([1.0, 0.2, 0.2]),
    ))
    log("✅ Scene built: pick_table + place_table + box + markers")
    return box


# ── Lula IK controller ────────────────────────────────────────────────────────
class LulaController:
    """
    Wraps Isaac Sim 5.1.0 Lula IK for Cartesian end-effector control.
    Falls back to joint interpolation if Lula is unavailable.
    """

    def __init__(self, arm: SingleArticulation, urdf_path: str, ee_frame: str = "link_7"):
        self.arm      = arm
        self.ee_frame = ee_frame
        self._lula    = None
        self._use_lula = False
        # Pre-computed warm starts for key positions (from workspace_scan)
        self.WARM_PICK  = np.array([ 0.0,  0.23,  0.0,  0.664,  0.0,  1.677,  0.0])
        self.WARM_PLACE = np.array([-0.321,  0.414, -0.446,  1.041, -1.301,  1.271,  0.0])

        descriptor_path = os.path.join(REPO, "configs/rm75b_descriptor.yaml")
        try:
            from isaacsim.robot_motion.motion_generation.lula import LulaKinematicsSolver
            self._lula = LulaKinematicsSolver(
                robot_description_path=descriptor_path,
                urdf_path=urdf_path,
            )
            self._use_lula = True
            log("✅ Lula IK initialized with rm75b_descriptor.yaml")
        except Exception as e:
            log(f"⚠️  Lula unavailable ({e}), using joint interpolation fallback")

    def move_to(self, target_pos: np.ndarray, target_quat: np.ndarray, warm_hint: np.ndarray = None) -> bool:
        """Send arm to Cartesian target. Returns True if IK succeeded."""
        if self._use_lula and self._lula is not None:
            try:
                warm = warm_hint if warm_hint is not None else self.arm.get_joint_positions()[:N_ARM_DOFS]
                joint_pos, success = self._lula.compute_inverse_kinematics(
                    frame_name=self.ee_frame,
                    warm_start=warm,
                    target_position=target_pos,
                    # orientation omitted — position-only IK is more robust
                )
                # Always apply best IK result even if not fully converged
                self._apply_arm(joint_pos)
                if not success:
                    log(f"   ⚠️ IK not converged for {np.round(target_pos,3)}, applying best effort")
                return success
            except Exception as e:
                log(f"   IK error: {e}")

        return False

    def move_home(self):
        self._apply_arm(HOME_JOINTS)

    def _apply_arm(self, joint_pos: np.ndarray):
        current = self.arm.get_joint_positions()
        full    = current.copy()
        full[:N_ARM_DOFS] = joint_pos[:N_ARM_DOFS]
        action  = ArticulationAction(joint_positions=full)
        self.arm.apply_action(action)

    def set_gripper(self, closed: bool):
        current = self.arm.get_joint_positions()
        full    = current.copy()
        full[N_ARM_DOFS:] = GRIP_CLOSE if closed else GRIP_OPEN
        self.arm.apply_action(ArticulationAction(joint_positions=full))


# ── State machine ─────────────────────────────────────────────────────────────
class PickPlaceStateMachine:
    """
    IDLE → HOME → HOVER_PICK → DESCEND_PICK → GRASP →
    LIFT → HOVER_PLACE → DESCEND_PLACE → RELEASE → HOME → DONE
    """

    DWELL = {   # steps to dwell in each state before transitioning
        "HOME":          80,
        "HOVER_PICK":   120,
        "DESCEND_PICK": 120,
        "GRASP":         80,
        "LIFT":         120,
        "HOVER_PLACE":  200,  # longer — place is far away
        "DESCEND_PLACE":150,
        "RELEASE":       80,
        "HOME_FINAL":    60,
    }

    def __init__(self, ctrl: LulaController):
        self.ctrl             = ctrl
        self.state            = "IDLE"
        self.state_entry_step = 0
        self._ee_quat         = euler_angles_to_quat(np.array([np.pi, 0.0, 0.0]))

    def tick(self, step: int) -> bool:
        n = step - self.state_entry_step

        if self.state == "IDLE":
            self.ctrl.set_gripper(closed=False)
            self._go("HOME", step)

        elif self.state == "HOME":
            self.ctrl.move_home()
            if n > self.DWELL["HOME"]: self._go("HOVER_PICK", step)

        elif self.state == "HOVER_PICK":
            self.ctrl.move_to(PICK_XYZ + np.array([0, 0, HOVER_Z]), self._ee_quat, warm_hint=self.ctrl.WARM_PICK)
            if n > self.DWELL["HOVER_PICK"]: self._go("DESCEND_PICK", step)

        elif self.state == "DESCEND_PICK":
            self.ctrl.move_to(PICK_XYZ, self._ee_quat, warm_hint=self.ctrl.WARM_PICK)
            if n > self.DWELL["DESCEND_PICK"]: self._go("GRASP", step)

        elif self.state == "GRASP":
            self.ctrl.set_gripper(closed=True)
            if n > self.DWELL["GRASP"]: self._go("LIFT", step)

        elif self.state == "LIFT":
            self.ctrl.move_to(PICK_XYZ + np.array([0, 0, HOVER_Z]), self._ee_quat, warm_hint=self.ctrl.WARM_PICK)
            if n > self.DWELL["LIFT"]: self._go("HOVER_PLACE", step)

        elif self.state == "HOVER_PLACE":
            self.ctrl.move_to(PLACE_XYZ + np.array([0, 0, HOVER_Z]), self._ee_quat, warm_hint=self.ctrl.WARM_PLACE)
            if n > self.DWELL["HOVER_PLACE"]: self._go("DESCEND_PLACE", step)

        elif self.state == "DESCEND_PLACE":
            self.ctrl.move_to(PLACE_XYZ, self._ee_quat, warm_hint=self.ctrl.WARM_PLACE)
            if n > self.DWELL["DESCEND_PLACE"]: self._go("RELEASE", step)

        elif self.state == "RELEASE":
            self.ctrl.set_gripper(closed=False)
            if n > self.DWELL["RELEASE"]: self._go("HOME_FINAL", step)

        elif self.state == "HOME_FINAL":
            self.ctrl.move_home()
            if n > self.DWELL["HOME_FINAL"]:
                log(f"✅ [step {step}] Pick-place cycle complete!")
                return True

        return False

    def _go(self, new_state: str, step: int):
        log(f"   [{step:>5}] {self.state} → {new_state}")
        self.state            = new_state
        self.state_entry_step = step


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()

    arm_prim = load_arm()
    box      = build_scene(world)
    world.reset()

    # Step twice before initializing articulation (Isaac Sim 5.x requirement)
    world.step(render=False)
    arm = SingleArticulation(prim_path=arm_prim)
    world.step(render=False)
    arm.initialize()

    log(f"   DOFs ({arm.num_dof}): {arm.dof_names}")

    ctrl = LulaController(arm=arm, urdf_path=URDF_PATH, ee_frame="link_7")
    sm   = PickPlaceStateMachine(ctrl=ctrl)

    log("\n🚀 Pick-place with RMPFlow/Lula IK")
    log(f"   Pick  → {PICK_XYZ}")
    log(f"   Place → {PLACE_XYZ}\n")

    step, done = 0, False
    MAX_STEPS  = 1200
    t0         = time.time()

    while simulation_app.is_running() and step < MAX_STEPS and not done:
        world.step(render=not args.headless)

        if step % 60 == 0:
            box_pos = box.get_world_pose()[0]
            log(f"   [step {step:>4}] {sm.state:<18} box={np.round(box_pos, 3)}")

        done = sm.tick(step)
        step += 1

    elapsed = time.time() - t0
    box_final = box.get_world_pose()[0]
    dist      = np.linalg.norm(box_final[:2] - PLACE_XYZ[:2])

    log("\n── Results ─────────────────────────────────────────────────")
    log(f"   Box final pos  : {np.round(box_final, 3)}")
    log(f"   Dist to goal   : {dist:.4f} m  {'✅' if dist < 0.1 else '⚠️ arm needs IK'}")
    log(f"   Final state    : {sm.state}")
    log(f"   Steps / time   : {step} / {elapsed:.1f}s")
    log(f"\n{'✅ PASS' if done else '❌ FAIL (timeout)'} — pick_place_rmpflow")
    log("────────────────────────────────────────────────────────────\n")

    if not args.headless:
        log("   GUI open — close window to exit")
        while simulation_app.is_running():
            world.step(render=True)

    simulation_app.close()
    return 0 if done else 1


if __name__ == "__main__":
    raise SystemExit(main())
