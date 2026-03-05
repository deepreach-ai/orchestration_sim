"""
scripts/run_demo.py
-------------------
Entrypoint for the dual-arm warehouse orchestration demo.

Correct Isaac Sim import order:
  1. SimulationApp instantiated first (Carbonite framework requirement)
  2. All isaacsim.* and orchestration imports come AFTER

Run:
  ~/isaacsim/python.sh scripts/run_demo.py
  ~/isaacsim/python.sh scripts/run_demo.py --headless
  ~/isaacsim/python.sh scripts/run_demo.py --headless --cycles 5
"""

import argparse
import os
import sys

# ── Step 1: parse args before SimulationApp (no isaacsim needed yet) ─────────
parser = argparse.ArgumentParser(description="Dual-arm warehouse orchestration demo")
parser.add_argument("--headless", action="store_true", default=False)
parser.add_argument("--cycles",   type=int, default=3, help="Number of pick-place cycles")
parser.add_argument("--takt",     type=int, default=300, help="Min steps between cycle starts")
args, _ = parser.parse_known_args()

# ── Step 2: SimulationApp MUST be instantiated before any isaacsim.* import ──
from isaacsim import SimulationApp
sim_app = SimulationApp({"headless": args.headless, "width": 1280, "height": 720})

# ── Step 3: safe to import orchestration + isaacsim modules now ───────────────
sys.path.insert(0, os.path.expanduser("~/orchestration_sim"))

from isaacsim.core.api import World
from orchestration.task_manager import TaskManager, BoxItem

# ── Step 4: build world and run ───────────────────────────────────────────────
import time

world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()

# DR SKU set: backpack (largest), phone-sized, water-bottle-sized
SKU_POOL = ["BACKPACK_01", "PHONE_02", "PHONE_03", "BOTTLE_04", "PHONE_05", "BACKPACK_06"]

tm = TaskManager(takt_steps=args.takt, max_cycles=args.cycles, headless=args.headless)
tm.setup(world)
tm.enqueue([BoxItem(sku=SKU_POOL[i % len(SKU_POOL)]) for i in range(args.cycles)])

print(f"\n🚀 run_demo.py | cycles={args.cycles} | takt={args.takt} | headless={args.headless}\n")

step      = 0
MAX_STEPS = args.cycles * 2000
t0        = time.time()

while sim_app.is_running() and step < MAX_STEPS:
    world.step(render=not args.headless)
    if tm.tick(step):
        break
    step += 1

elapsed = time.time() - t0
print(f"\n⏱  Done: {step} steps in {elapsed:.1f}s")

if not args.headless:
    print("GUI open — close window to exit")
    while sim_app.is_running():
        world.step(render=True)

sim_app.close()
