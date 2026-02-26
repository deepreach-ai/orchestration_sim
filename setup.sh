#!/bin/bash
set -e

REPO=~/orchestration_sim
SRC=~/teleop_system

echo "=== Step 1: Copy assets ==="
cp -r "$SRC/robots/rm75b" "$REPO/assets/robots/"
cp -r "$SRC/crt_ctag2f90d_gripper_visualization" "$REPO/assets/robots/gripper"
cp "$SRC/configs/robots/realman_rm75b.yaml" "$REPO/configs/rm75b.yaml"
echo "✅ Assets copied"

echo "=== Step 2: Fix URDF mesh paths ==="
cd "$REPO/assets/robots/rm75b"
sed 's|package://RM75-B/meshes/|meshes/|g' rm75b_w_gripper.urdf > rm75b_local.urdf
sed -i '' 's|package://RM75-B/meshes/|meshes/|g' RM75-B.urdf || sed -i 's|package://RM75-B/meshes/|meshes/|g' RM75-B.urdf
echo "✅ URDF paths fixed"
echo ""
echo "Verifying mesh paths:"
grep "filename" rm75b_local.urdf | head -5
echo ""

echo "=== Step 1+2 complete ==="

echo "=== Init git ==="
cd "$REPO"
echo "__pycache__/
*.pyc
.venv/
.DS_Store" > .gitignore
git init
git add -A
git commit -m "init: scaffold repo with rm75b + gripper assets"
echo "✅ Git repo initialized and committed"
