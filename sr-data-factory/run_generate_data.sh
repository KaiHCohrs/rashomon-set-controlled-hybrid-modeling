#!/bin/bash

noise_levels=(0.0001 0.001 0.01 0.1)
training_sizes=(32 128 512 2048)

file_path="data_factory/functions/equations/problems_add.json"
equation_names=(jin_1 jin_3 jin_4 korns_12 nguyen_13 pagie_1 new_1 new_2 new_3 new_4)

seed=42
sample_size_test=1000

for equation_name in "${equation_names[@]}"
do
  for noise_level in "${noise_levels[@]}"
  do
    for training_size in "${training_sizes[@]}"
    do
      python scripts/generate_data.py "$file_path" "$equation_name" "$seed" "$noise_level" "$training_size" "$sample_size_test" --correlation 0.5
    done
  done
done


file_path="data_factory/functions/equations/problems_multipl.json"
equation_names=(jin_5 vladislavleva_1 vladislavleva_8 new_5 new_6)

seed=42
sample_size_test=1000

for equation_name in "${equation_names[@]}"
do
  for noise_level in "${noise_levels[@]}"
  do
    for training_size in "${training_sizes[@]}"
    do
      python scripts/generate_data.py "$file_path" "$equation_name" "$seed" "$noise_level" "$training_size" "$sample_size_test" --correlation 0.5
    done
  done
done