#!/usr/bin/env bash
# Reproduce every result in the paper. Total runtime is about 3 h on a 4-core CPU.
set -euo pipefail
cd "$(dirname "$0")/src"
python3 prepare_data.py
python3 baselines.py --level aggregate
python3 baselines.py --level household
for seed in 0 1 2; do
  for model in pv-tcqn mlp-q lstm-gauss; do
    python3 train_deep.py --level aggregate --model "$model" --seed "$seed" --epochs 25 --steps 150
  done
done
for ab in csky pvhist noskip; do       # ablations of PV-TCQN
  for seed in 0 1 2; do
    python3 train_deep.py --level aggregate --model pv-tcqn --seed "$seed" --epochs 25 --steps 150 --ablate "$ab"
  done
done
for model in pv-tcqn mlp-q lstm-gauss; do
  python3 train_deep.py --level household --model "$model" --seed 0 --epochs 20 --steps 300
done
python3 evaluate.py
python3 make_figures.py
