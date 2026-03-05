"""
orchestration/
--------------
Robot orchestration logic for the DR warehouse return pipeline sim.

Modules
-------
state_machine   : SingleArmSM, DualArmSM, ArmState, ArmRole
task_manager    : TaskManager, BoxItem, CycleRecord, SceneBuilder, ArmLoader

IMPORTANT — Isaac Sim import rule
----------------------------------
All isaacsim.* symbols are imported lazily (inside __init__ / method bodies).
Do NOT add top-level `from isaacsim import ...` anywhere in this package.
SimulationApp must be instantiated before any isaacsim module loads.

Safe usage pattern:
    from isaacsim import SimulationApp
    sim_app = SimulationApp({"headless": True})   # <-- must come first

    # only NOW is it safe to import orchestration classes
    from orchestration.state_machine import SingleArmSM, DualArmSM, ArmState, ArmRole
    from orchestration.task_manager  import TaskManager, BoxItem
"""

# Intentionally no top-level imports from orchestration sub-modules here.
# Import directly from the sub-modules in your scripts, e.g.:
#
#   from orchestration.state_machine import SingleArmSM, DualArmSM, ArmState, ArmRole
#   from orchestration.task_manager  import TaskManager, BoxItem, CycleRecord
#
# This avoids triggering isaacsim sub-module imports before SimulationApp exists.
