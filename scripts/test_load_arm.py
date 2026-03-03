"""
test_load_arm.py
----------------
Isaac Sim 5.1.0 smoke test: load RM75-B arm + gripper and spawn a pick target cube.

Run with:
  ~/isaacsim/python.sh scripts/test_load_arm.py
"""

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import os
import numpy as np

# ── Isaac Sim 5.1.0 API ───────────────────────────────────────────────────────
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
from isaacsim.asset.importer.urdf import _urdf

URDF_PATH = os.path.expanduser("~/orchestration_sim/assets/robots/rm75b/rm75b_local.urdf")

# ── Import robot via URDF ─────────────────────────────────────────────────────
urdf_interface = _urdf.acquire_urdf_interface()
cfg = _urdf.ImportConfig()
cfg.merge_fixed_joints             = False
cfg.fix_base                       = True
cfg.default_drive_type             = _urdf.UrdfJointTargetType.JOINT_DRIVE_POSITION
cfg.default_drive_strength         = 1e4
cfg.default_position_drive_damping = 1e3

urdf_interface.import_robot(URDF_PATH, "/World/rm75b", cfg)
print(f"✅ Robot imported from {URDF_PATH}")

# ── Setup world ───────────────────────────────────────────────────────────────
world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()

world.scene.add(FixedCuboid(
    prim_path="/World/table",
    name="table",
    position=[0.4, 0.0, 0.3],
    size=0.6,
    scale=[1.0, 1.5, 0.05],
    color=[0.6, 0.4, 0.2],
))

world.scene.add(DynamicCuboid(
    prim_path="/World/target_cube",
    name="target_cube",
    position=[0.4, 0.0, 0.35],
    size=0.04,
    color=[1.0, 0.0, 0.0],
    mass=0.1,
))

world.reset()
print("✅ Scene ready: RM75-B + table + red cube")
print("   Drag joints in the GUI to verify motion")

while simulation_app.is_running():
    world.step(render=True)

simulation_app.close()
