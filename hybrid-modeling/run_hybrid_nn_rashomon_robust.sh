#!/bin/bash
# Run Rashomon-robust hybrid model for a single problem.
# Data must be in ../sr-data-factory/data/ (generate via sr-data-factory first).
# Default: test run with jin_3, noise=0.0001, n=2048 (data file jin_3_correlation0.5u_noise0.0001_train2048_data.csv).

noise=0.0001
sample_size=2048
equation_name="jin"
equation_number="3"
equation_version="3"
config="base_model.yaml"
file_name="problems_eq.json"
index="test"
l2_net_decay="0.0"
data_file_name="jin_3_correlation0.5u_noise0.0001_train2048_data.csv"

python scripts/hybrid_nn_rashomon_robust.py \
  --noise "$noise" \
  --sample_size "$sample_size" \
  --l2_net_decay "$l2_net_decay" \
  --file_name "$file_name" \
  --equation_name "$equation_name" \
  --equation_number "$equation_number" \
  --equation_version "$equation_version" \
  --config "$config" \
  --index "$index" \
  --data_file_name "$data_file_name" \
  --default_epsilon 0.01 \
  --max_diameter 0.1
