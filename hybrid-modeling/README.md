# Hybrid modeling (supplementary code)

This package implements hybrid modeling with neural networks: standard formulations (full NN, partial physics, hybrid with fixed or varying regularization) and a Rashomon-robust formulation that fits a Rashomon set with controlled diameter.

## Setup

1. **Create the environment:**

   ```bash
   mamba env create -f environment.yml
   conda activate hybrid_modeling
   pip install .
   ```

2. **Data:** Place data CSVs in `../sr-data-factory/data/` relative to this package (e.g. generate with sr-data-factory in the sibling folder; see its README). A test data file is provided: `../sr-data-factory/data/jin_3_correlation0.5u_noise0.0001_train2048_data.csv`.

## Test run (default)

Both run scripts are set up for a **default test run** using:

- **Data file:** `jin_3_correlation0.5u_noise0.0001_train2048_data.csv` (in `../sr-data-factory/data/`)
- **Problem:** jin_3_3 (equation_name=jin, equation_number=3, equation_version=3)
- **Noise:** 0.0001, **sample size:** 2048

From the `hybrid-modeling` directory:

```bash
# Standard hybrid (full_nn, partial_physics, or hybrid_nn)
bash run_hybrid_nn.sh

# Rashomon-robust hybrid
bash run_hybrid_nn_rashomon_robust.sh
```

Results are written to `results/`. For the Rashomon script, epsilon/delta use the aggregate CSVs in `results/` by default (`full_nn_aggregate_for_epsilon.csv`, `partial_physics_aggregate_for_delta.csv`) if present.

## Running other setups

Edit the variables at the top of `run_hybrid_nn.sh` or `run_hybrid_nn_rashomon_robust.sh`: `noise`, `sample_size`, `l2_net_decay` (variance regularization on network output), `equation_name`, `equation_number`, `equation_version`, `file_name` (e.g. `problems_eq.json`), and for the standard script `type` (`hybrid_nn`, `full_nn`, `partial_physics`). Set `data_file_name` to the CSV filename in `../sr-data-factory/data/`, or leave it empty in the Rashomon script to use default naming from equation/noise/sample_size.
