"""
orchestration/
--------------
Robot orchestration logic for the DR warehouse return pipeline sim.

Modules
-------
state_machine   : SingleArmSM, DualArmSM, ArmState, ArmRole
                  (Day 4 single-arm baseline + Day 5 dual-arm handover)
task_manager    : TaskManager, BoxItem, CycleRecord, SceneBuilder, ArmLoader
                  (Day 5 — scene setup, cycle queue, takt pacing, metrics)

Coming (Day 6)
--------------
fallback_handler : grasp failure retry, skip-and-alert, reassign-to-other-arm
amr_controller   : AMR motion + timing sync with arms
"""

from orchestration.state_machine import (
    SingleArmSM,
    DualArmSM,
    ArmState,
    ArmRole,
)

from orchestration.task_manager import (
    TaskManager,
    BoxItem,
    CycleRecord,
    SceneBuilder,
    ArmLoader,
    CONVEYOR_PICK_XYZ,
    HANDOVER_XYZ,
    OUTPUT_PLACE_XYZ,
)

__all__ = [
    # state_machine
    "SingleArmSM", "DualArmSM", "ArmState", "ArmRole",
    # task_manager
    "TaskManager", "BoxItem", "CycleRecord", "SceneBuilder", "ArmLoader",
    "CONVEYOR_PICK_XYZ", "HANDOVER_XYZ", "OUTPUT_PLACE_XYZ",
]
