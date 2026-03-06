"""
scripts/keyboard_teleop.py
---------------------------
Keyboard teleoperation for RM75-B in Isaac Sim + trajectory recording.

Keyboard input strategy
------------------------
carb.input only works when the Isaac Sim viewport has focus — it cannot
read terminal keypresses. Instead we run a background thread that reads
single raw keypresses from stdin using tty/termios (UNIX raw mode).

New concept: raw terminal mode
-------------------------------
Normally the terminal buffers input and only sends it to the program when
you press Enter. In "raw mode" (termios.setraw), every keypress is
immediately available as a byte, with no Enter required and no echo.
We restore the original terminal settings on exit with a try/finally.

Controls
--------
  W / S      : EE +X / -X
  A / D      : EE +Y / -Y
  Q / E      : EE +Z / -Z
  O          : gripper open
  C          : gripper close / grasp
  H          : move home
  R          : start / stop recording
  P          : save trajectory JSON
  ESC or X   : quit

Run:
  ~/isaacsim/python.sh scripts/keyboard_teleop.py
"""

import os, sys, time, json, argparse, datetime, threading, queue
import numpy as np
import tty, termios

# ── parse args BEFORE SimulationApp ──────────────────────────────────────────
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--headless", action="store_true", default=False)
_args, _ = _parser.parse_known_args()

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": _args.headless, "width": 1280, "height": 720})

from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid, VisualSphere
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.prims import SingleArticulation
from isaacsim.asset.importer.urdf import _urdf

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO      = os.path.expanduser("~/orchestration_sim")
URDF_PATH = os.path.join(REPO, "assets/robots/rm75b/rm75b_local.urdf")
LOG_DIR   = os.path.join(REPO, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

TABLE_H  = 0.40
BOX_SIZE = 0.05
BOX_HALF = BOX_SIZE / 2

N_ARM_DOFS  = 7
HOME_JOINTS = np.array([0.0, -0.009, 0.0, -0.439, 0.0, 1.92, 0.0])
GRIP_OPEN   = np.array([0.5])
GRIP_CLOSE  = np.array([0.0])
STEP_SIZE   = 0.01   # metres per keypress
RECORD_EVERY = 30

_logf = open("/tmp/teleop.log", "w", buffering=1)
def log(msg): _logf.write(msg+"\n"); _logf.flush(); print(msg, flush=True)


# ─────────────────────────────────────────────────────────────────────────────
#  RawKeyReader — background thread, reads single chars from stdin
# ─────────────────────────────────────────────────────────────────────────────
class RawKeyReader:
    """
    Reads single keypresses from stdin in a daemon thread.
    Puts each char (lowercase) into a thread-safe queue.
    The main sim loop calls .get() each step.
    """
    def __init__(self):
        self._q = queue.Queue()
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._fd = sys.stdin.fileno()
        self._old = termios.tcgetattr(self._fd)

    def start(self):
        self._t.start()

    def stop(self):
        self._stop.set()
        # Restore terminal
        try:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)
        except Exception:
            pass

    def _run(self):
        try:
            tty.setraw(self._fd)
            while not self._stop.is_set():
                ch = sys.stdin.read(1)
                if ch:
                    self._q.put(ch.lower())
        finally:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)

    def get_all(self) -> list:
        """Drain all pending keypresses this sim step."""
        keys = []
        while True:
            try:
                keys.append(self._q.get_nowait())
            except queue.Empty:
                break
        return keys


# ─────────────────────────────────────────────────────────────────────────────
#  LulaController
# ─────────────────────────────────────────────────────────────────────────────
class LulaController:

    @staticmethod
    def _find_ee_prim(root_path: str, link_name: str) -> str:
        """
        BFS the USD stage from root_path to find the prim named link_name.
        URDF import nests links under auto-generated joint prims, so the
        exact depth is unpredictable — we search instead of hardcoding.
        """
        import omni.usd
        from collections import deque
        stage = omni.usd.get_context().get_stage()
        root  = stage.GetPrimAtPath(root_path)
        if not root.IsValid():
            log(f"⚠️  root prim not found: {root_path}")
            return root_path + "/" + link_name
        q = deque([root])
        while q:
            p = q.popleft()
            if p.GetName() == link_name:
                found = str(p.GetPath())
                log(f"✅ EE prim auto-found: {found}")
                return found
            for child in p.GetChildren():
                q.append(child)
        log(f"⚠️  '{link_name}' not found under {root_path}")
        return root_path + "/" + link_name

    def __init__(self, arm: SingleArticulation, arm_prim_path: str):
        self.arm          = arm
        self._ee_prim     = self._find_ee_prim(arm_prim_path, "link_7")
        self._lula        = None
        self._attached_box = None

        descriptor = os.path.join(REPO, "configs/rm75b_descriptor.yaml")
        try:
            from isaacsim.robot_motion.motion_generation.lula import LulaKinematicsSolver
            self._lula = LulaKinematicsSolver(
                robot_description_path=descriptor,
                urdf_path=URDF_PATH,
            )
            seeds = np.array([
                [ 0.0, -0.5,  0.0, -1.0,  0.0,  1.5,  0.0],
                [-0.5, -0.4,  0.0, -1.0,  0.0,  1.5,  0.0],
                [ 0.0, -0.009, 0.0, -0.439, 0.0, 1.92, 0.0],
            ])
            self._lula.set_default_cspace_seeds(seeds)
            log(f"✅ Lula IK ready  EE prim: {self._ee_prim}")
        except Exception as e:
            log(f"⚠️  Lula: {e}")

    def move_to(self, pos: np.ndarray) -> bool:
        if self._lula is None: return False
        warm = self.arm.get_joint_positions()[:N_ARM_DOFS]
        try:
            joints, ok = self._lula.compute_inverse_kinematics(
                frame_name="link_7", warm_start=warm, target_position=pos)
            self._apply(joints)
            return ok
        except Exception as e:
            log(f"IK err: {e}"); return False

    def move_home(self):
        self._apply(HOME_JOINTS)

    def _apply(self, joints):
        cur = self.arm.get_joint_positions().copy()
        cur[:N_ARM_DOFS] = joints[:N_ARM_DOFS]
        self.arm.apply_action(ArticulationAction(joint_positions=cur))

    def set_gripper(self, arr: np.ndarray):
        cur = self.arm.get_joint_positions().copy()
        n = cur.shape[0] - N_ARM_DOFS
        t = arr
        if t.shape[0] < n: t = np.pad(t, (0, n-t.shape[0]), mode='edge')
        cur[N_ARM_DOFS:] = t[:n]
        self.arm.apply_action(ArticulationAction(joint_positions=cur))

    def get_ee_pos(self) -> np.ndarray:
        """Read link_7 world position directly from USD stage."""
        try:
            import omni.usd
            from pxr import UsdGeom
            stage = omni.usd.get_context().get_stage()
            prim  = stage.GetPrimAtPath(self._ee_prim)
            if not prim.IsValid():
                log(f"⚠️  EE prim not found: {self._ee_prim}")
                return np.zeros(3)
            xform = UsdGeom.Xformable(prim)
            mat   = xform.ComputeLocalToWorldTransform(0)
            t     = mat.ExtractTranslation()
            return np.array([t[0], t[1], t[2]])
        except Exception as e:
            log(f"get_ee_pos err: {e}")
            return np.zeros(3)

    def get_joints(self) -> np.ndarray:
        return self.arm.get_joint_positions()[:N_ARM_DOFS].copy()

    def attach_box(self, box):
        self._attached_box = box
        log("   📦 attach")

    def detach_box(self):
        self._attached_box = None
        log("   📦 detach")

    def move_box_with_ee(self):
        if self._attached_box is None: return
        try:
            pos = self.get_ee_pos()
            self._attached_box.set_world_pose(position=pos)
        except: pass


# ─────────────────────────────────────────────────────────────────────────────
#  TrajectoryRecorder
# ─────────────────────────────────────────────────────────────────────────────
class TrajectoryRecorder:
    def __init__(self):
        self.waypoints = []
        self.recording = False
        self._last = -999

    def start(self, step):
        self.recording = True; self.waypoints = []; self._last = -999
        log(f"\n🔴 REC START  step={step}")

    def stop(self, step):
        self.recording = False
        log(f"⏹  REC STOP   step={step}  waypoints={len(self.waypoints)}")

    def record(self, step, ee, joints, grip, action="move", force=False):
        if not self.recording: return
        if not force and (step - self._last) < RECORD_EVERY: return
        self.waypoints.append({
            "step": step, "ee_pos": ee.tolist(),
            "joints": joints.tolist(), "gripper": float(grip), "action": action,
        })
        self._last = step

    def save(self, step) -> str:
        ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(LOG_DIR, f"trajectory_{ts}.json")
        with open(path, "w") as f:
            json.dump({"metadata": {"date": ts, "total_steps": step,
                                    "n_waypoints": len(self.waypoints)},
                       "waypoints": self.waypoints}, f, indent=2)
        log(f"💾 Saved → {path}  ({len(self.waypoints)} waypoints)")
        return path


# ─────────────────────────────────────────────────────────────────────────────
#  Main app
# ─────────────────────────────────────────────────────────────────────────────
def main():
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()

    # Table
    world.scene.add(FixedCuboid(
        prim_path="/World/table", name="table",
        position=[0.35, 0.0, TABLE_H/2], size=1.0,
        scale=[0.6, 0.6, TABLE_H], color=np.array([0.4,0.4,0.4])))

    # Box on table surface
    box_pos = [0.30, 0.0, TABLE_H + BOX_HALF]
    box = world.scene.add(DynamicCuboid(
        prim_path="/World/box", name="box",
        position=box_pos, size=BOX_SIZE,
        color=np.array([0.95,0.75,0.1]), mass=0.05))

    # Load URDF
    ui  = _urdf.acquire_urdf_interface()
    cfg = _urdf.ImportConfig()
    cfg.merge_fixed_joints             = False
    cfg.fix_base                       = True
    cfg.default_drive_type             = _urdf.UrdfJointTargetType.JOINT_DRIVE_POSITION
    cfg.default_drive_strength         = 8e3
    cfg.default_position_drive_damping = 8e2
    asset_root = os.path.dirname(URDF_PATH)
    asset_name = os.path.basename(URDF_PATH)
    robot = ui.parse_urdf(asset_root, asset_name, cfg)
    ui.import_robot("", asset_name, robot, cfg, "/RM75_B_with_Gripper")

    world.reset()
    world.step(render=False)

    arm = SingleArticulation(prim_path="/RM75_B_with_Gripper")
    world.step(render=False)
    arm.initialize()
    log(f"✅ Arm DOFs={arm.num_dof}  prim={arm.prim_path}")

    ctrl = LulaController(arm, arm.prim_path)
    ctrl.move_home()
    world.step(render=False)

    # Initialise EE target from actual home pose
    ee_target = ctrl.get_ee_pos().copy()
    log(f"   Initial EE pos: {np.round(ee_target, 3)}")

    rec   = TrajectoryRecorder()
    kb    = RawKeyReader()
    grip  = 0.5   # current gripper state
    step  = 0
    quit_ = False

    WORKSPACE_LO = np.array([-0.70, -0.70, 0.05])
    WORKSPACE_HI = np.array([ 0.70,  0.70, 1.10])

    log("""
╔══════════════════════════════════════════════╗
║       RM75-B Keyboard Teleop (stdin mode)    ║
╠══════════════════════════════════════════════╣
║  W/S  → EE +X/-X    A/D  → EE +Y/-Y         ║
║  Q/E  → EE +Z/-Z                            ║
║  O    → gripper open                        ║
║  C    → gripper close / grasp               ║
║  H    → home                                ║
║  R    → start/stop recording                ║
║  P    → save trajectory JSON                ║
║  X or ESC → quit                            ║
╠══════════════════════════════════════════════╣
║  Click THIS TERMINAL window to type!        ║
╚══════════════════════════════════════════════╝
""")

    kb.start()

    try:
        while simulation_app.is_running() and not quit_:
            world.step(render=not _args.headless)
            ctrl.move_box_with_ee()

            # ── Process all keypresses queued this step ───────────────────
            moved = False
            for ch in kb.get_all():
                if   ch == 'w': ee_target[0] += STEP_SIZE; moved = True
                elif ch == 's': ee_target[0] -= STEP_SIZE; moved = True
                elif ch == 'a': ee_target[1] += STEP_SIZE; moved = True
                elif ch == 'd': ee_target[1] -= STEP_SIZE; moved = True
                elif ch == 'q': ee_target[2] += STEP_SIZE; moved = True
                elif ch == 'e': ee_target[2] -= STEP_SIZE; moved = True

                elif ch == 'o':
                    grip = 0.5; ctrl.set_gripper(GRIP_OPEN)
                    ctrl.detach_box()
                    log(f"  [{step}] gripper OPEN")
                    rec.record(step, ctrl.get_ee_pos(), ctrl.get_joints(),
                               grip, "release", force=True)

                elif ch == 'c':
                    grip = 0.0; ctrl.set_gripper(GRIP_CLOSE)
                    ee  = ctrl.get_ee_pos()
                    bp  = np.array(box.get_world_pose()[0])
                    dist = np.linalg.norm(ee - bp)
                    if dist < 0.15:
                        ctrl.attach_box(box)
                        log(f"  [{step}] GRASP ✅  dist={dist:.3f}m")
                    else:
                        log(f"  [{step}] gripper CLOSE (box far: {dist:.3f}m)")
                    rec.record(step, ctrl.get_ee_pos(), ctrl.get_joints(),
                               grip, "grasp", force=True)

                elif ch == 'h':
                    ctrl.move_home()
                    ee_target = ctrl.get_ee_pos().copy()
                    log(f"  [{step}] HOME  EE={np.round(ee_target,3)}")

                elif ch == 'r':
                    if rec.recording: rec.stop(step)
                    else:             rec.start(step)

                elif ch == 'p':
                    if rec.waypoints: rec.save(step)
                    else: log("  ⚠️  No waypoints — press R first")

                elif ch in ('x', '\x1b'):   # ESC = \x1b
                    log("\n👋 Quit")
                    if rec.waypoints: rec.save(step)
                    quit_ = True; break

            if moved:
                ee_target = np.clip(ee_target, WORKSPACE_LO, WORKSPACE_HI)
                ctrl.move_to(ee_target)
                rec.record(step, ctrl.get_ee_pos(), ctrl.get_joints(),
                           grip, "move")

            if step % 120 == 0:
                ee = ctrl.get_ee_pos()
                r  = "🔴" if rec.recording else "  "
                log(f"[{step:>6}] {r} EE={np.round(ee,3)}  "
                    f"grip={'CLOSE' if grip<0.1 else 'open '}  "
                    f"wp={len(rec.waypoints)}")
            step += 1

    finally:
        kb.stop()
        simulation_app.close()


if __name__ == "__main__":
    main()
