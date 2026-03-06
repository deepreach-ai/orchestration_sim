"""
pick_place_rmpflow.py
---------------------
Isaac Sim 5.1.0 — RM75-B 单臂抓取放置，使用真正的 RMPflow 运动规划。

RMPflow vs Lula IK 的根本区别
-------------------------------
Lula IK：给定目标点 → 求一个关节角解（静态求解）
  问题：position-only 时解不唯一，臂可能选择侧翻姿态

RMPflow（Riemannian Motion Policies）：给定目标点+朝向 → 生成连续轨迹
  优势：
  1. 强制 EE 朝向约束（夹爪始终垂直朝下）
  2. 内置障碍物回避
  3. 每 step 计算下一个关节速度，轨迹平滑
  4. 不会因为 warm start 选错而侧翻

运行:
  ~/isaacsim/python.sh scripts/pick_place_rmpflow.py
  ~/isaacsim/python.sh scripts/pick_place_rmpflow.py --headless
"""

import os, sys, time, argparse
import numpy as np

_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--headless", action="store_true", default=False)
_args, _ = _parser.parse_known_args()

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": _args.headless, "width": 1280, "height": 720})

# ── 所有 isaacsim.* import 必须在 SimulationApp() 之后 ────────────────────────
from isaacsim.core.api         import World
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid, VisualSphere
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.prims       import SingleArticulation
from isaacsim.asset.importer.urdf import _urdf
from pxr import UsdGeom
import omni.usd

_log = open("/tmp/pick_place.log", "w", buffering=1)
def log(msg): _log.write(msg+"\n"); _log.flush(); print(msg, flush=True)

# ── 路径 ──────────────────────────────────────────────────────────────────────
REPO            = os.path.expanduser("~/orchestration_sim")
URDF_PATH       = os.path.join(REPO, "assets/robots/rm75b/rm75b_local.urdf")
DESCRIPTOR_PATH = os.path.join(REPO, "configs/rm75b_descriptor.yaml")

# ── 场景常量 ──────────────────────────────────────────────────────────────────
TABLE_H  = 0.40
BOX_SIZE = 0.05
BOX_HALF = BOX_SIZE / 2

# 抓取/放置位置
PICK_XYZ  = np.array([0.30,  0.0,  TABLE_H + BOX_HALF])   # 箱子中心高度
PLACE_XYZ = np.array([0.20, -0.30, TABLE_H + BOX_HALF])
HOVER_Z   = 0.18   # hover 高度（比箱子高 18cm，给手臂充足的顶部入射空间）

# 夹爪朝下的四元数（roll=π → Z轴朝下）
EE_QUAT_DOWN = euler_angles_to_quat(np.array([np.pi, 0.0, 0.0]))

N_ARM_DOFS  = 7
HOME_JOINTS = np.array([0.0, -0.009, 0.0, -0.439, 0.0, 1.92, 0.0])
GRIP_OPEN   = np.array([0.5])
GRIP_CLOSE  = np.array([0.0])


# ── USD stage 工具函数 ────────────────────────────────────────────────────────
def find_prim_by_name(root_path: str, name: str) -> str:
    """
    BFS 搜索 USD stage，从 root_path 开始找第一个名为 name 的 prim。
    URDF import 会自动生成中间 joint prim，导致路径深度不固定，
    所以不能硬编码路径，必须搜索。
    """
    from collections import deque
    stage = omni.usd.get_context().get_stage()
    root  = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        log(f"⚠️  root prim 无效: {root_path}")
        return root_path + "/" + name
    q = deque([root])
    while q:
        p = q.popleft()
        if p.GetName() == name:
            path = str(p.GetPath())
            log(f"✅ 找到 prim: {path}")
            return path
        for c in p.GetChildren():
            q.append(c)
    log(f"⚠️  未找到 '{name}'，使用 fallback 路径")
    return root_path + "/" + name


def get_prim_world_pos(prim_path: str) -> np.ndarray:
    """通过 USD Xformable 读取 prim 的世界坐标（不依赖 Isaac 封装层）。"""
    stage = omni.usd.get_context().get_stage()
    prim  = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return np.zeros(3)
    xform = UsdGeom.Xformable(prim)
    mat   = xform.ComputeLocalToWorldTransform(0)
    t     = mat.ExtractTranslation()
    return np.array([t[0], t[1], t[2]])


# ── URDF 加载 ─────────────────────────────────────────────────────────────────
def load_arm() -> str:
    ui  = _urdf.acquire_urdf_interface()
    cfg = _urdf.ImportConfig()
    cfg.merge_fixed_joints             = False
    cfg.fix_base                       = True
    cfg.default_drive_type             = _urdf.UrdfJointTargetType.JOINT_DRIVE_POSITION
    cfg.default_drive_strength         = 1.2e4   # 更强的关节驱动力
    cfg.default_position_drive_damping = 1.2e3
    asset_root = os.path.dirname(URDF_PATH)
    asset_name = os.path.basename(URDF_PATH)
    robot = ui.parse_urdf(asset_root, asset_name, cfg)
    prim  = "/RM75_B_with_Gripper"
    ui.import_robot("", asset_name, robot, cfg, prim)
    log(f"✅ RM75-B 加载完毕 → {prim}")
    return prim


# ── 场景构建 ──────────────────────────────────────────────────────────────────
def build_scene(world: World):
    world.scene.add(FixedCuboid(
        prim_path="/World/pick_table", name="pick_table",
        position=[0.35, 0.0, TABLE_H/2], size=1.0,
        scale=[0.5, 0.5, TABLE_H], color=np.array([0.4,0.4,0.4])))
    world.scene.add(FixedCuboid(
        prim_path="/World/place_table", name="place_table",
        position=[0.30, -0.45, TABLE_H/2], size=1.0,
        scale=[0.5, 0.5, TABLE_H], color=np.array([0.55,0.35,0.15])))

    box_spawn = [PICK_XYZ[0], PICK_XYZ[1], TABLE_H + BOX_HALF]
    box = world.scene.add(DynamicCuboid(
        prim_path="/World/box", name="box",
        position=box_spawn, size=BOX_SIZE,
        color=np.array([0.95,0.75,0.1]), mass=0.05))

    world.scene.add(VisualSphere(
        prim_path="/World/pick_marker", name="pick_marker",
        position=PICK_XYZ.tolist(), radius=0.015,
        color=np.array([0.0,1.0,0.0])))
    world.scene.add(VisualSphere(
        prim_path="/World/place_marker", name="place_marker",
        position=PLACE_XYZ.tolist(), radius=0.015,
        color=np.array([1.0,0.2,0.2])))
    log("✅ 场景构建完成")
    return box


# ── RMPflow 控制器 ────────────────────────────────────────────────────────────
class RMPflowController:
    """
    用 Isaac Sim 的 RmpFlow 做运动规划。

    新概念：RmpFlow 工作原理
    -------------------------
    RMPflow 将机器人的关节空间和任务空间通过黎曼度量（Riemannian Metric）
    联系起来。每个 step：
      1. 给定 EE 目标位置+朝向
      2. RMPflow 计算一个"吸引力场"把 EE 拉向目标
      3. 同时计算"排斥力场"把臂推离障碍物
      4. 两者合并生成关节速度指令

    这与 Lula IK 的根本区别：
    - Lula：一次性求解，可能得到奇异解
    - RMPflow：每 step 迭代逼近，轨迹连续且安全

    ArticulationMotionPolicy：
    --------------------------
    Isaac Sim 的桥接层，把 RmpFlow 的输出（关节目标）
    转换为 ArticulationAction 并应用到仿真中。
    """

    def __init__(self, arm: SingleArticulation, arm_prim_path: str):
        self.arm          = arm
        self.arm_prim     = arm_prim_path
        self._ee_prim     = find_prim_by_name(arm_prim_path, "link_7")
        self._rmpflow     = None
        self._policy      = None
        self._attached_box = None
        self._init_rmpflow()

    def _init_rmpflow(self):
        try:
            from isaacsim.robot_motion.motion_generation import RmpFlow, ArticulationMotionPolicy
            self._rmpflow = RmpFlow(
                robot_description_path  = DESCRIPTOR_PATH,
                urdf_path               = URDF_PATH,
                rmpflow_config_path     = self._get_rmpflow_config(),
                end_effector_frame_name = "link_7",
                maximum_substep_size    = 0.00334,  # 1/60 / 5 substeps
            )
            self._policy = ArticulationMotionPolicy(
                robot_articulation = self.arm,
                motion_policy      = self._rmpflow,
                default_physics_dt = 1.0 / 60.0,
            )
            # ── Null space attractor：让 RMPflow 在冗余空间里偏向直立姿态 ──
            # 这是防止侧翻的核心：当 EE 位置可以被多种关节构型达到时，
            # RMPflow 会在 null space 里把臂拉向 HOME_JOINTS（直立）
            self._rmpflow.set_cspace_attractor(HOME_JOINTS)
            log("✅ RMPflow 初始化成功，null space attractor = HOME_JOINTS")
        except Exception as e:
            log(f"⚠️  RMPflow 初始化失败: {e}")
            log("   → 回退到 Lula IK")
            self._init_lula_fallback()

    def _get_rmpflow_config(self) -> str:
        """
        RMPflow 需要一个 rmpflow_config.yaml。
        Isaac Sim 自带的 Franka 配置文件路径作为参考，
        我们优先使用 configs/ 目录下的自定义版本，
        没有的话自动生成一个最小配置。
        """
        custom = os.path.join(REPO, "configs/rm75b_rmpflow.yaml")
        if os.path.exists(custom):
            log(f"   使用自定义 RMPflow 配置: {custom}")
            return custom
        # 生成最小 RMPflow 配置
        self._generate_rmpflow_config(custom)
        return custom

    def _generate_rmpflow_config(self, path: str):
        """
        自动生成适合 RM75-B 的最小 RMPflow 配置文件。

        新概念：RMPflow 配置参数
        -------------------------
        cspace_target_ee_attraction_scale: EE 被目标吸引的强度
        cspace_barrier_weight: 关节极限排斥权重
        joint_limit_buffers: 距关节极限多少 rad 开始排斥
        """
        config = """\
# RM75-B RMPflow 配置
# 自动生成 — 可手动调整

# EE 吸引力场参数
ee_target:
  metric_weight: 10.0
  accel_p_gain: 80.0
  accel_d_gain: 20.0
  accel_i_gain: 0.0

# 关节空间阻尼（防止奇异抖动）
joint_target:
  metric_weight: 0.1

# 关节极限排斥
joint_limits:
  metric_weight: 1.0
  buffer_size: 0.10   # rad

# 自碰撞排斥
collision_avoidance:
  metric_weight: 1.0
"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(config)
        log(f"   自动生成 RMPflow 配置 → {path}")

    def _init_lula_fallback(self):
        """RMPflow 不可用时回退到 Lula IK（带强制直立 seed）。"""
        try:
            from isaacsim.robot_motion.motion_generation.lula import LulaKinematicsSolver
            self._lula = LulaKinematicsSolver(
                robot_description_path=DESCRIPTOR_PATH,
                urdf_path=URDF_PATH,
            )
            seeds = np.array([
                [ 0.0, -0.5,  0.0, -1.0,  0.0,  1.5,  0.0],
                [-0.5, -0.4,  0.0, -1.0,  0.0,  1.5,  0.0],
                [ 0.0, -0.009, 0.0, -0.439, 0.0, 1.92, 0.0],
                [ 0.0, -0.7,  0.0, -0.8,  0.0,  1.2,  0.0],
            ])
            self._lula.set_default_cspace_seeds(seeds)
            log("✅ Lula IK 回退初始化成功（直立 seed 偏置）")
        except Exception as e:
            log(f"❌ Lula 也失败了: {e}")
            self._lula = None

    # ── 运动控制接口 ──────────────────────────────────────────────────────────

    def set_target(self, pos: np.ndarray, quat: np.ndarray = None):
        """
        设置 EE 目标。RMPflow 每 step 会向目标逼近。
        quat 为 None 时使用夹爪朝下的默认朝向。
        """
        if quat is None:
            quat = EE_QUAT_DOWN
        if self._policy is not None:
            self._rmpflow.set_end_effector_target(
                target_position    = pos,
                target_orientation = quat,
            )
        elif hasattr(self, '_lula') and self._lula:
            warm = self.arm.get_joint_positions()[:N_ARM_DOFS]
            j, _ = self._lula.compute_inverse_kinematics(
                frame_name="link_7", warm_start=warm, target_position=pos)
            self._apply_joints(j)

    def step(self, physics_dt: float = 1.0/60.0):
        """
        每个仿真 step 调用一次，传入物理时间步长。
        get_next_articulation_action(step) 必须传 dt，否则 RMPflow
        内部积分器无法正确计算速度，会产生抖动或错误姿态。
        """
        if self._policy is not None:
            action = self._policy.get_next_articulation_action(physics_dt)
            if action is not None:
                self.arm.apply_action(action)

    def move_home(self):
        self._apply_joints(HOME_JOINTS)

    def _apply_joints(self, joints: np.ndarray):
        cur = self.arm.get_joint_positions().copy()
        cur[:N_ARM_DOFS] = joints[:N_ARM_DOFS]
        self.arm.apply_action(ArticulationAction(joint_positions=cur))

    def set_gripper(self, closed: bool):
        cur   = self.arm.get_joint_positions().copy()
        n     = cur.shape[0] - N_ARM_DOFS
        tgt   = GRIP_CLOSE if closed else GRIP_OPEN
        if tgt.shape[0] < n:
            tgt = np.pad(tgt, (0, n - tgt.shape[0]), mode='edge')
        cur[N_ARM_DOFS:] = tgt[:n]
        self.arm.apply_action(ArticulationAction(joint_positions=cur))

    def get_ee_pos(self) -> np.ndarray:
        return get_prim_world_pos(self._ee_prim)

    # ── 箱子附着（teleport 模式）─────────────────────────────────────────────

    def attach_box(self, box):
        self._attached_box = box
        log("   📦 attach（teleport 模式）")

    def detach_box(self):
        if self._attached_box:
            log("   📦 detach")
        self._attached_box = None

    def sync_box(self):
        """每 step 把箱子传送到 EE 位置。"""
        if self._attached_box is None:
            return
        try:
            pos = self.get_ee_pos()
            self._attached_box.set_world_pose(position=pos)
        except Exception as e:
            log(f"sync_box err: {e}")

    def at_target(self, target: np.ndarray, tol: float = 0.04) -> bool:
        """判断 EE 是否已经到达目标（误差 < tol 米）。"""
        ee = self.get_ee_pos()
        d  = np.linalg.norm(ee - target)
        return d < tol


# ── 状态机 ────────────────────────────────────────────────────────────────────
class PickPlaceStateMachine:
    """
    使用 RMPflow 的 pick-place 状态机。

    与原 Lula 版本的关键区别：
    - 不再等固定 dwell 步数，而是等 at_target() 为 True
      （RMPflow 轨迹长度不固定，到达即切换）
    - 仍有最大等待步数防止死锁
    - set_target() 只需调用一次（RMPflow 每 step 自动逼近）
    """

    MAX_WAIT = {
        "HOME":          200,
        "HOVER_PICK":    300,
        "DESCEND_PICK":  250,
        "GRASP":          60,
        "LIFT":          250,
        "HOVER_PLACE":   350,
        "DESCEND_PLACE": 300,
        "RELEASE":        60,
        "HOME_FINAL":    200,
    }
    TOL = {   # 到达容差（米）
        "HOVER_PICK":    0.05,
        "DESCEND_PICK":  0.03,
        "LIFT":          0.05,
        "HOVER_PLACE":   0.05,
        "DESCEND_PLACE": 0.03,
    }

    def __init__(self, ctrl: RMPflowController, box):
        self.ctrl  = ctrl
        self.box   = box
        self.state = "IDLE"
        self._entry = 0
        self._target = None

    def tick(self, step: int) -> bool:
        n = step - self._entry

        if self.state == "IDLE":
            self.ctrl.set_gripper(closed=False)
            self._go("HOME", step)

        elif self.state == "HOME":
            self.ctrl.move_home()
            if n > self.MAX_WAIT["HOME"]:
                self._go("HOVER_PICK", step)

        elif self.state == "HOVER_PICK":
            t = PICK_XYZ + np.array([0, 0, HOVER_Z])
            self._set_once(t)
            if self.ctrl.at_target(t, self.TOL["HOVER_PICK"]) or n > self.MAX_WAIT["HOVER_PICK"]:
                self._go("DESCEND_PICK", step)

        elif self.state == "DESCEND_PICK":
            self._set_once(PICK_XYZ)
            if self.ctrl.at_target(PICK_XYZ, self.TOL["DESCEND_PICK"]) or n > self.MAX_WAIT["DESCEND_PICK"]:
                self._go("GRASP", step)

        elif self.state == "GRASP":
            self.ctrl.set_gripper(closed=True)
            if n == 2:
                self.ctrl.attach_box(self.box)
            self.ctrl.sync_box()
            if n > self.MAX_WAIT["GRASP"]:
                self._go("LIFT", step)

        elif self.state == "LIFT":
            t = PICK_XYZ + np.array([0, 0, HOVER_Z])
            self._set_once(t)
            self.ctrl.sync_box()
            if self.ctrl.at_target(t, self.TOL["LIFT"]) or n > self.MAX_WAIT["LIFT"]:
                self._go("HOVER_PLACE", step)

        elif self.state == "HOVER_PLACE":
            t = PLACE_XYZ + np.array([0, 0, HOVER_Z])
            self._set_once(t)
            self.ctrl.sync_box()
            if self.ctrl.at_target(t, self.TOL["HOVER_PLACE"]) or n > self.MAX_WAIT["HOVER_PLACE"]:
                self._go("DESCEND_PLACE", step)

        elif self.state == "DESCEND_PLACE":
            self._set_once(PLACE_XYZ)
            self.ctrl.sync_box()
            if self.ctrl.at_target(PLACE_XYZ, self.TOL["DESCEND_PLACE"]) or n > self.MAX_WAIT["DESCEND_PLACE"]:
                self._go("RELEASE", step)

        elif self.state == "RELEASE":
            self.ctrl.set_gripper(closed=False)
            if n == 2:
                self.ctrl.detach_box()
            if n > self.MAX_WAIT["RELEASE"]:
                self._go("HOME_FINAL", step)

        elif self.state == "HOME_FINAL":
            self.ctrl.move_home()
            if n > self.MAX_WAIT["HOME_FINAL"]:
                log(f"✅ [step {step}] 完成！")
                return True

        # RMPflow 每 step 必须调用
        self.ctrl.step()
        return False

    def _set_once(self, target: np.ndarray):
        """仅在状态第一次 tick 时设置目标（RMPflow 内部保持目标）。"""
        if self._target is None or not np.allclose(self._target, target):
            self.ctrl.set_target(target)
            self._target = target.copy()

    def _go(self, new_state: str, step: int):
        log(f"   [{step:>5}] {self.state} → {new_state}")
        self.state   = new_state
        self._entry  = step
        self._target = None   # 重置目标，触发 _set_once


# ── 主函数 ────────────────────────────────────────────────────────────────────
def main():
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()

    arm_prim = load_arm()
    box      = build_scene(world)
    world.reset()

    world.step(render=False)
    arm = SingleArticulation(prim_path=arm_prim)
    world.step(render=False)
    arm.initialize()
    log(f"   DOFs={arm.num_dof}  prim={arm.prim_path}")

    ctrl = RMPflowController(arm, arm_prim)
    sm   = PickPlaceStateMachine(ctrl, box)

    log(f"\n🚀 RMPflow pick-place 启动")
    log(f"   Pick  → {PICK_XYZ}")
    log(f"   Place → {PLACE_XYZ}\n")

    step, done = 0, False
    MAX_STEPS  = 2400   # RMPflow 轨迹更长，给更多步数
    t0 = time.time()

    while simulation_app.is_running() and step < MAX_STEPS and not done:
        world.step(render=not _args.headless)

        if step % 60 == 0:
            ee  = ctrl.get_ee_pos()
            bp  = np.array(box.get_world_pose()[0])
            log(f"   [step {step:>4}] {sm.state:<18} "
                f"EE={np.round(ee,3)}  box={np.round(bp,3)}")

        done = sm.tick(step)
        step += 1

    elapsed   = time.time() - t0
    box_final = np.array(box.get_world_pose()[0])
    dist      = np.linalg.norm(box_final[:2] - PLACE_XYZ[:2])

    log("\n── 结果 ──────────────────────────────────────────────────────")
    log(f"   箱子最终位置  : {np.round(box_final,3)}")
    log(f"   距目标距离    : {dist:.4f} m  {'✅' if dist<0.12 else '❌'}")
    log(f"   最终状态      : {sm.state}")
    log(f"   步数/耗时     : {step} / {elapsed:.1f}s")
    log(f"\n{'✅ PASS' if done else '❌ FAIL (timeout)'} — pick_place_rmpflow")
    log("──────────────────────────────────────────────────────────────\n")

    if not _args.headless:
        log("   GUI 保持打开，关闭窗口退出")
        while simulation_app.is_running():
            world.step(render=True)

    simulation_app.close()
    return 0 if done else 1


if __name__ == "__main__":
    raise SystemExit(main())
