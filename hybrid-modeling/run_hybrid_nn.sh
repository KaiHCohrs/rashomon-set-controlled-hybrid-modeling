#!/bin/bash
# Default: test run with jin_3, noise=0.0001, n=2048 (data: jin_3_correlation0.5u_noise0.0001_train2048_data.csv).

noise=0.0001
sample_size=2048
equation_name="jin"
equation_number="3"
equation_version="3"
config="base_model.yaml"
file_name="problems_eq.json"
index="test"
type="hybrid_nn" # partial_physics, full_nn, hybrid_nn
l2_net_decay="0.01"
data_file_name="jin_3_correlation0.5u_noise0.0001_train2048_data.csv"

python scripts/hybrid_nn.py \
  --noise "$noise" \
  --l2_net_decay "$l2_net_decay" \
  --sample_size "$sample_size" \
  --file_name "$file_name" \
  --equation_name "$equation_name" \
  --equation_number "$equation_number" \
  --equation_version "$equation_version" \
  --config "$config" \
  --model_type "$type" \
  --index "$index" \
  --data_file_name "$data_file_name"
