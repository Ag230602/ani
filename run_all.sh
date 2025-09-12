#!/bin/bash
set -e

cd ULIP

# Ensure output dirs exist
mkdir -p ../runs/baseline ../runs/variation ../logs

# Generate timestamp
TS=$(date +"%Y%m%d-%H%M%S")

echo "=== Running ULIP Baseline ==="
python main_zero_shot.py \
    --dataset modelnet40 \
    --pc_root ../data/ModelNet40_PC \
    --text_root ../data/texts \
    --ckpt ../pretrained/ulip_ckpt.pth \
    > ../logs/baseline_${TS}.log 2>&1

echo "Baseline complete → logs/baseline_${TS}.log"

echo "=== Running ULIP Variation (batch_size=128) ==="
python main_zero_shot.py \
    --dataset modelnet40 \
    --pc_root ../data/ModelNet40_PC \
    --text_root ../data/texts \
    --ckpt ../pretrained/ulip_ckpt.pth \
    --batch_size 128 \
    > ../logs/variation_${TS}.log 2>&1

echo "Variation complete → logs/variation_${TS}.log"

echo "All runs finished. Logs saved with timestamp ${TS}."
