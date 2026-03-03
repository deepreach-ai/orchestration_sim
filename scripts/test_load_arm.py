"""
Load RM75-B arm + gripper in Isaac Sim and spawn a pick target cube.
Run with: ~/.local/share/ov/pkg/isaac-sim-*/python.sh scripts/test_load_arm.py
"""
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import os
import numpy as np
from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid, FixedCuboid
from omni.isaac.urdf import _urdf

URDF_PATH = os.path.expanduser("~/orchestration_sim/assets/robots/rm75b/rm75b_local.urdf")

# --- Import robot via URDF ---
urdf_interface = _urdf.acquire_urdf_interface()
import_config = _urdf.ImportConfig()
import_config.merge_fixed_joints = False
import_config.fix_base = True
import_config.default_drive_type = _urdf.UrdfJointTargetType.JOINT_DRIVE_POSITION
import_config.default_drive_strength = 1e4
import_config.default_position_drive_damping = 1e3

result = urdf_interface.parse_urdf(URDF_PATH, import_config)
dest_path = "/World/rm75b"
urdf_interface.import_robot(URDF_PATH, dest_path, import_config)
print(f"✅ Robot imported from {URDF_PATH}")

# --- Setup world ---
world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()

# Table
table = world.scene.add(
    FixedCuboid(
        prim_path="/World/table",
        name="table",
        position=[0.4, 0.0, 0.3],
        size=0.6,
        scale=[1.0, 1.5, 0.05],
        color=[0.6, 0.4, 0.2],
    )
)

# Pick target cube
cube = world.scene.add(
    DynamicCuboid(
        prim_path="/World/target_cube",
        name="target_cube",
        position=[0.4, 0.0, 0.35],
        size=0.04,
        color=[1.0, 0.0, 0.0],
        mass=0.1,
    )
)

world.reset()
print("✅ Scene ready: RM75-B + table + red cube")
print("   Drag joints in the GUI to verify motion")

while simulation_app.is_running():
    world.step(render=True)

simulation_app.close()
