"""
scripts/keyboard_teleop.py
---------------------------
Keyboard teleoperation for RM75-B in Isaac Sim + trajectory recording.

Controls (EE Cartesian delta mode)
------------------------------------
  W / S      : EE +X / -X  (forward / back)
  A / D      : EE +Y / -Y  (left / right)
  Q / E      : EE +Z / -Z  (up / down)
  O          : gripper open
  C          : gripper close
  R          : start/stop recording
  P          : print + save trajectory to JSON
  H          : move home
  ESC / Ctrl+C : quit

Trajectory format (saved to logs/trajectory_<timestamp>.json)
--------------------------------------------------------------
  {
    "metadata": { "date": ..., "total_steps": ..., "n_waypoints": ... },
    "waypoints": [
      { "step": 42, "ee_pos": [x,y,z], "joints": [j1..j7], "gripper": 0.5,
        "action": "move" },   # action = "move" | "grasp" | "release" | "home"
      ...
    ]
  }

New concepts
------------
- Delta Cartesian control: instead of sending absolute targets, each keypress
  adds a small delta (STEP_SIZE) to the current EE position. This is more
  intuitive for manual operation.

- Trajectory recording: waypoints are captured at configurable intervals
  (every RECORD_EVERY sim steps) while recording is active, plus always
  on gripper open/close events. This gives a compact representation of
  the motion that can later be replayed or used for imitation learning.

- get_world_pose on link_7: we read the current EE position directly from
  the prim rather than from IK, so the recorded pose is always ground-truth.

Run:
  ~/isaacsim/python.sh scripts/keyboard_teleop.py
  ~/isaacsim/python.sh scripts/keyboard_teleop.py --headless   # no GUI (testing only)
"""

import os, sys, time, json, argparse, datetime
import numpy as np

# ── parse args BEFORE SimulationApp ──────────────────────────────────────────
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--headless", action="store_true", default=False)
_args, _ = _parser.parse_known_args()

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": _args.headless, "width": 1280, "height": 720})

# ── Isaac Sim imports (must follow SimulationApp) ─────────────────────────────
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid, VisualSphere
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.utils.xforms import get_world_pose
from isaacsim.core.prims import SingleArticulation
from isaacsim.asset.importer.urdf import _urdf
import carb.input
import omni.appwindow
from pxr import Gf

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO      = os.path.expanduser("~/orchestration_sim")
URDF_PATH = os.path.join(REPO, "assets/robots/rm75b/rm75b_local.urdf")
LOG_DIR   = os.path.join(REPO, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# ── Scene constants ───────────────────────────────────────────────────────────
TABLE_H  = 0.40
BOX_SIZE = 0.05    # small enough for gripper
BOX_HALF = BOX_SIZE / 2

PICK_XYZ  = np.array([0.30,  0.0,  TABLE_H + BOX_HALF + 0.01])

# ── Teleop parameters ─────────────────────────────────────────────────────────
STEP_SIZE     = 0.01   # metres per keypress — smaller = more precise
RECORD_EVERY  = 30     # record a waypoint every N sim steps while recording

# ── Arm constants (from pick_place_rmpflow) ───────────────────────────────────
N_ARM_DOFS  = 7
HOME_JOINTS = np.array([0.0, -0.009, 0.0, -0.439, 0.0, 1.92, 0.0])
GRIP_OPEN   = np.array([0.5])
GRIP_CLOSE  = np.array([0.0])
EE_QUAT     = euler_angles_to_quat(np.array([np.pi, 0.0, 0.0]))

# ── Log ───────────────────────────────────────────────────────────────────────
_logf = open("/tmp/teleop.log", "w", buffering=1)
def log(msg): _logf.write(msg + "\n"); _logf.flush(); print(msg, flush=True)


# ─────────────────────────────────────────────────────────────────────────────
#  LulaController (minimal copy from pick_place_rmpflow)
# ─────────────────────────────────────────────────────────────────────────────
class LulaController:
    def __init__(self, arm: SingleArticulation):
        self.arm = arm
        self._lula = None
        self._attached_box = None

        descriptor_path = os.path.join(REPO, "configs/rm75b_descriptor.yaml")
        try:
            from isaacsim.robot_motion.motion_generation.lula import LulaKinematicsSolver
            self._lula = LulaKinematicsSolver(
                robot_description_path=descriptor_path,
                urdf_path=URDF_PATH,
            )
            seeds = np.array([
                [ 0.0,  0.23,  0.0,   0.664,  0.0,  1.677,  0.0],
                [-0.321, 0.414, -0.446, 1.041, -1.301, 1.271, 0.0],
                [ 0.0, -0.009,  0.0,  -0.439,  0.0,  1.92,   0.0],
            ])
            self._lula.set_default_cspace_seeds(seeds)
            log("✅ Lula IK ready")
        except Exception as e:
            log(f"⚠️  Lula unavailable: {e}")

    def move_to(self, target_pos: np.ndarray) -> bool:
        """Move EE to target_pos (position-only IK)."""
        if self._lula is None:
            return False
        warm = self.arm.get_joint_positions()[:N_ARM_DOFS]
        try:
            joints, ok = self._lula.compute_inverse_kinematics(
                frame_name="link_7",
                warm_start=warm,
                target_position=target_pos,
            )
            self._apply_arm(joints)
            return ok
        except Exception as e:
            log(f"IK error: {e}")
            return False

    def move_home(self):
        self._apply_arm(HOME_JOINTS)

    def _apply_arm(self, joints):
        cur = self.arm.get_joint_positions().copy()
        cur[:N_ARM_DOFS] = joints[:N_ARM_DOFS]
        self.arm.apply_action(ArticulationAction(joint_positions=cur))

    def set_gripper(self, pos_arr: np.ndarray):
        cur = self.arm.get_joint_positions().copy()
        n = cur.shape[0] - N_ARM_DOFS
        t = pos_arr
        if t.shape[0] < n: t = np.pad(t, (0, n - t.shape[0]), mode='edge')
        cur[N_ARM_DOFS:] = t[:n]
        self.arm.apply_action(ArticulationAction(joint_positions=cur))

    def get_ee_pos(self) -> np.ndarray:
        """Return current link_7 world position."""
        try:
            pos, _ = get_world_pose(self.arm.prim_path + "/link_7")
            return np.array(pos)
        except:
            return np.zeros(3)

    def get_joints(self) -> np.ndarray:
        return self.arm.get_joint_positions()[:N_ARM_DOFS]

    def attach_box(self, box): self._attached_box = box
    def detach_box(self):     self._attached_box = None

    def move_box_with_ee(self):
        if self._attached_box is None: return
        try:
            pos, ori = get_world_pose(self.arm.prim_path + "/link_7")
            self._attached_box.set_world_pose(position=pos, orientation=ori)
        except: pass


# ─────────────────────────────────────────────────────────────────────────────
#  KeyboardInput — thin wrapper around carb.input
# ─────────────────────────────────────────────────────────────────────────────
class KeyboardInput:
    """
    New concept: carb.input
    -----------------------
    Isaac Sim / Omniverse uses the Carbonite (carb) framework for input.
    carb.input.IInput lets us query key states directly each sim step,
    which is better than callbacks for real-time control because:
    - No threading issues
    - Key-held detection is natural (just call is_pressed every step)
    - Works in both GUI and headless modes (headless: all keys report False)
    """
    def __init__(self):
        self._input = carb.input.acquire_input_interface()
        self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        self._key = carb.input.KeyboardInput

    def pressed(self, key_name: str) -> bool:
        try:
            k = getattr(self._key, key_name)
            return self._input.get_keyboard_value(self._keyboard, k) > 0
        except:
            return False

    def any_pressed(self, *key_names) -> bool:
        return any(self.pressed(k) for k in key_names)


# ─────────────────────────────────────────────────────────────────────────────
#  TrajectoryRecorder
# ─────────────────────────────────────────────────────────────────────────────
class TrajectoryRecorder:
    """
    Records EE waypoints during teleoperation.

    New concept: Waypoint-based trajectory representation
    ------------------------------------------------------
    Instead of recording every single sim step (which would be enormous),
    we record:
      1. Periodic samples every RECORD_EVERY steps (background motion)
      2. Event waypoints on gripper open/close (critical action moments)
      3. Each waypoint stores: step, EE position, joint angles, gripper state, action tag

    This sparse representation can be:
      - Replayed by interpolating between waypoints
      - Used as demonstrations for imitation learning (e.g. ACT, Diffusion Policy)
      - Converted to a joint-space trajectory for real robot execution
    """
    def __init__(self):
        self.waypoints: list = []
        self.recording: bool = False
        self._start_step: int = 0
        self._last_record_step: int = -999

    def start(self, step: int):
        self.recording = True
        self._start_step = step
        self.waypoints = []
        log(f"\n🔴 Recording STARTED at step {step}")

    def stop(self, step: int):
        self.recording = False
        log(f"⏹  Recording STOPPED at step {step} — {len(self.waypoints)} waypoints")

    def record(self, step: int, ee_pos: np.ndarray, joints: np.ndarray,
               gripper: float, action: str = "move", force: bool = False):
        """
        Add a waypoint. Skips if not enough steps since last record
        (unless force=True, used for gripper events).
        """
        if not self.recording:
            return
        if not force and (step - self._last_record_step) < RECORD_EVERY:
            return

        self.waypoints.append({
            "step":    step,
            "ee_pos":  ee_pos.tolist(),
            "joints":  joints.tolist(),
            "gripper": float(gripper),
            "action":  action,
        })
        self._last_record_step = step

    def save(self, step: int) -> str:
        """Save trajectory to JSON. Returns filepath."""
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(LOG_DIR, f"trajectory_{ts}.json")
        data = {
            "metadata": {
                "date":        ts,
                "total_steps": step,
                "n_waypoints": len(self.waypoints),
                "step_size_m": STEP_SIZE,
                "record_every": RECORD_EVERY,
            },
            "waypoints": self.waypoints,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        log(f"💾 Trajectory saved → {path}  ({len(self.waypoints)} waypoints)")
        return path


# ─────────────────────────────────────────────────────────────────────────────
#  TeleopApp — main loop
# ─────────────────────────────────────────────────────────────────────────────
class TeleopApp:
    def __init__(self):
        self.world    = World(stage_units_in_meters=1.0)
        self.kb       = KeyboardInput()
        self.recorder = TrajectoryRecorder()
        self.ctrl     = None
        self.box      = None

        self._ee_target   = PICK_XYZ.copy()   # current Cartesian target
        self._gripper_pos = 0.5               # 0.0=closed, 0.5=open
        self._is_grasping = False
        self._step        = 0

        # Debounce flags (avoid triggering repeatedly while key held)
        self._prev = {k: False for k in ["R","P","H","O","C","ESC"]}

    def setup(self):
        self.world.scene.add_default_ground_plane()

        # Pick table
        self.world.scene.add(FixedCuboid(
            prim_path="/World/table", name="table",
            position=[0.40, 0.0, TABLE_H / 2],
            size=1.0, scale=[0.6, 0.6, TABLE_H],
            color=np.array([0.4, 0.4, 0.4]),
        ))

        # Box (small, grippable)
        self.box = self.world.scene.add(DynamicCuboid(
            prim_path="/World/box", name="box",
            position=PICK_XYZ.tolist(),
            size=BOX_SIZE,
            color=np.array([0.95, 0.75, 0.1]),
            mass=0.1,
        ))

        # EE target marker (blue sphere)
        self.world.scene.add(VisualSphere(
            prim_path="/World/target_marker", name="target_marker",
            position=self._ee_target.tolist(),
            radius=0.015, color=np.array([0.1, 0.4, 1.0]),
        ))

        # Load arm
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

        self.world.reset()
        self.world.step(render=False)

        arm = SingleArticulation(prim_path=prim)
        self.world.step(render=False)
        arm.initialize()
        log(f"✅ Arm ready — DOFs: {arm.num_dof}")

        self.ctrl = LulaController(arm)
        self.ctrl.move_home()

        # Initialise EE target from actual home EE position
        self._ee_target = self.ctrl.get_ee_pos().copy()

        self._print_help()

    def _print_help(self):
        log("""
╔══════════════════════════════════════════════════════════╗
║            RM75-B Keyboard Teleop + Recorder             ║
╠══════════════════════════════════════════════════════════╣
║  W/S   → EE +X/-X (forward/back)                        ║
║  A/D   → EE +Y/-Y (left/right)                          ║
║  Q/E   → EE +Z/-Z (up/down)                             ║
║  O     → gripper open                                   ║
║  C     → gripper close  (grasp)                         ║
║  H     → move home                                      ║
║  R     → start / stop recording                         ║
║  P     → save trajectory to JSON                        ║
║  ESC   → quit                                           ║
╚══════════════════════════════════════════════════════════╝
""")

    def _debounce(self, key: str) -> bool:
        """Returns True only on the rising edge of a key press."""
        cur = self.kb.pressed(key)
        rose = cur and not self._prev[key]
        self._prev[key] = cur
        return rose

    def _update_target_marker(self):
        """Move the blue sphere to track _ee_target."""
        try:
            from isaacsim.core.utils.xforms import set_prim_position
            set_prim_position("/World/target_marker", self._ee_target)
        except: pass

    def run(self):
        log("\n🚀 Teleop running — see controls above\n")
        MAX_STEPS = 200_000

        while simulation_app.is_running() and self._step < MAX_STEPS:
            self.world.step(render=not _args.headless)
            self._handle_input()
            self.ctrl.move_box_with_ee()
            self._step += 1

            if self._step % 120 == 0:
                ee = self.ctrl.get_ee_pos()
                rec = "🔴REC" if self.recorder.recording else "   "
                log(f"[{self._step:>6}] {rec} EE={np.round(ee,3)}  "
                    f"grip={'CLOSE' if self._gripper_pos < 0.1 else 'open '}  "
                    f"waypoints={len(self.recorder.waypoints)}")

    def _handle_input(self):
        step = self._step
        moved = False

        # ── EE delta movement ────────────────────────────────────────────────
        if self.kb.pressed("W"): self._ee_target[0] += STEP_SIZE; moved = True
        if self.kb.pressed("S"): self._ee_target[0] -= STEP_SIZE; moved = True
        if self.kb.pressed("A"): self._ee_target[1] += STEP_SIZE; moved = True
        if self.kb.pressed("D"): self._ee_target[1] -= STEP_SIZE; moved = True
        if self.kb.pressed("Q"): self._ee_target[2] += STEP_SIZE; moved = True
        if self.kb.pressed("E"): self._ee_target[2] -= STEP_SIZE; moved = True

        if moved:
            # Clamp to workspace limits (from rm75b.yaml)
            self._ee_target = np.clip(
                self._ee_target,
                [-0.70, -0.70, 0.05],
                [ 0.70,  0.70, 1.10],
            )
            self.ctrl.move_to(self._ee_target)
            self._update_target_marker()
            # Record motion waypoint
            self.recorder.record(
                step, self.ctrl.get_ee_pos(), self.ctrl.get_joints(),
                self._gripper_pos, action="move",
            )

        # ── Gripper open (O) ─────────────────────────────────────────────────
        if self._debounce("O"):
            self._gripper_pos = 0.5
            self.ctrl.set_gripper(GRIP_OPEN)
            self._is_grasping = False
            self.ctrl.detach_box()
            log(f"  [{step}] Gripper OPEN")
            self.recorder.record(
                step, self.ctrl.get_ee_pos(), self.ctrl.get_joints(),
                self._gripper_pos, action="release", force=True,
            )

        # ── Gripper close / grasp (C) ────────────────────────────────────────
        if self._debounce("C"):
            self._gripper_pos = 0.0
            self.ctrl.set_gripper(GRIP_CLOSE)
            # Check if box is close to EE → attach
            ee = self.ctrl.get_ee_pos()
            box_pos = self.box.get_world_pose()[0]
            dist = np.linalg.norm(ee - np.array(box_pos))
            if dist < 0.12:
                self.ctrl.attach_box(self.box)
                self._is_grasping = True
                log(f"  [{step}] Gripper CLOSE + GRASP (dist={dist:.3f}m)")
            else:
                log(f"  [{step}] Gripper CLOSE (box too far: {dist:.3f}m)")
            self.recorder.record(
                step, self.ctrl.get_ee_pos(), self.ctrl.get_joints(),
                self._gripper_pos, action="grasp", force=True,
            )

        # ── Home (H) ─────────────────────────────────────────────────────────
        if self._debounce("H"):
            self.ctrl.move_home()
            self._ee_target = self.ctrl.get_ee_pos().copy()
            log(f"  [{step}] HOME")
            self.recorder.record(
                step, self.ctrl.get_ee_pos(), self.ctrl.get_joints(),
                self._gripper_pos, action="home", force=True,
            )

        # ── Record toggle (R) ────────────────────────────────────────────────
        if self._debounce("R"):
            if self.recorder.recording:
                self.recorder.stop(step)
            else:
                self.recorder.start(step)

        # ── Save (P) ─────────────────────────────────────────────────────────
        if self._debounce("P"):
            if self.recorder.waypoints:
                self.recorder.save(step)
            else:
                log("  ⚠️  No waypoints recorded yet — press R to start recording first")

        # ── Quit (ESC) ───────────────────────────────────────────────────────
        if self._debounce("ESC"):
            log("\n👋 ESC pressed — saving and quitting...")
            if self.recorder.waypoints:
                self.recorder.save(step)
            simulation_app.close()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = TeleopApp()
    app.setup()
    app.run()
    simulation_app.close()
