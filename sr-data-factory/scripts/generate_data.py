import argparse
import numpy as np
import pandas as pd
import json
import subprocess
from pathlib import Path
from datetime import datetime
from numpy import exp, cos, sin


def generate_data(
    file_path,
    name,
    seed,
    noise_level,
    correlation,
    sample_type,
    sample_size_train,
    sample_size_test,
):
    # Load JSON configuration
    with open(file_path, "r") as file:
        equations = json.load(file)

    # Find the equation
    equation = next((eq for eq in equations if eq["name"] == name), None)
    if equation is None:
        raise ValueError(f"Equation with name {name} not found.")

    # Extract equation details
    equation_str = equation["equation"]
    input_vars = equation["input_variables"]
    params = equation["parameters"]
    param_values = equation["parameter_values"]
    var_ranges = equation["input_variable_ranges"]
    if "seed" in equation.keys():
        seed = equation["seed"] + sample_size_train
        print("Seed found in JSON file. Overriding seed argument.")

    # Extract equation details
    np.random.seed(seed)

    # Evaluate the equation
    def evaluate_equation(inputs):
        locals_dict = {var: inputs[i] for i, var in enumerate(input_vars)}
        locals_dict.update(param_values)
        return eval(equation_str, locals_dict, {"exp": exp, "cos": cos, "sin": sin})

    # Generate samples
    def generate_samples(sample_size, ranges):
        samples = []
        for _ in range(sample_size):
            inputs = []
            for var in input_vars:
                inputs.append(np.random.uniform(*ranges[var]))
            y = evaluate_equation(inputs)
            samples.append(inputs + [y])
        return np.array(samples)

    def generate_correlated_samples(
        sample_size, ranges, correlation=0.5, sample_type="uniform"
    ):
        samples = []

        while len(samples) < sample_size:
            inputs = []
            if sample_type == "uniform":
                for var in input_vars:
                    inputs.append(np.random.uniform(*ranges[var]))
                for i in range(1, len(input_vars)):
                    inputs[i] = inputs[0] * correlation + inputs[i]
            if all(
                ranges[var][0] <= inputs[i] <= ranges[var][1]
                for i, var in enumerate(input_vars)
            ):
                y = evaluate_equation(inputs)
                samples.append(inputs + [y])

        return np.array(samples)

    # Generate training data
    train_data = generate_correlated_samples(
        sample_size_train,
        {var: var_ranges[var][0] for var in input_vars},
        correlation=correlation,
        sample_type=sample_type,
    )
    train_df = pd.DataFrame(train_data, columns=input_vars + ["y"])
    train_df["scenario"] = "train"
    y_rms = np.sqrt(np.mean(train_df["y"] ** 2))
    noise_std = noise_level * y_rms
    train_df["y_obs"] = train_df["y"] + np.random.normal(
        0, noise_std, train_df.shape[0]
    )

    # Generate in-distribution test data
    test_data_in_dist = generate_correlated_samples(
        sample_size_test,
        {var: var_ranges[var][0] for var in input_vars},
        correlation=correlation,
        sample_type=sample_type,
    )
    test_df_in_dist = pd.DataFrame(test_data_in_dist, columns=input_vars + ["y"])
    test_df_in_dist["scenario"] = "test"
    test_df_in_dist["y_obs"] = test_df_in_dist["y"] + np.random.normal(
        0, noise_std, test_df_in_dist.shape[0]
    )

    # Generate out-of-distribution test data
    test_dfs_out_dist = []
    for var in input_vars:
        ranges = {
            v: var_ranges[var][1] if v == var else var_ranges[v][0] for v in input_vars
        }
        test_data_out_dist = generate_samples(sample_size_test, ranges)
        test_df_out_dist = pd.DataFrame(test_data_out_dist, columns=input_vars + ["y"])
        test_df_out_dist["scenario"] = f"test_{var}"
        test_df_out_dist["y_obs"] = test_df_out_dist["y"] + np.random.normal(
            0, noise_std, test_df_out_dist.shape[0]
        )
        test_dfs_out_dist.append(test_df_out_dist)

    # Generate out-of-distribution test data for all input variables
    test_data_all_out_dist = generate_samples(
        sample_size_test, {var: var_ranges[var][1] for var in input_vars}
    )
    test_df_all_out_dist = pd.DataFrame(
        test_data_all_out_dist, columns=input_vars + ["y"]
    )
    test_df_all_out_dist["scenario"] = f"test_" + "".join(input_vars)
    test_df_all_out_dist["y_obs"] = test_df_all_out_dist["y"] + np.random.normal(
        0, noise_std, test_df_all_out_dist.shape[0]
    )

    # Combine all dataframes
    combined_df = pd.concat(
        [train_df, test_df_in_dist] + test_dfs_out_dist + [test_df_all_out_dist],
        ignore_index=True,
    )

    # Create dataset name
    current_date = datetime.now().strftime("%Y-%m-%d")
    if correlation == 0:
        dataset_name = f"{name}_noise{noise_level}_train{sample_size_train}"
    elif sample_type == "uniform":
        dataset_name = f"{name}_correlation{correlation}u_noise{noise_level}_train{sample_size_train}"

    # Create the data folder if it does not exist
    data_folder = Path(args.data_folder)
    data_folder.mkdir(parents=True, exist_ok=True)

    # Save combined data to CSV
    combined_csv = f"{dataset_name}_data.csv"
    combined_df.to_csv(data_folder / combined_csv, index=False)

    # Get the current git commit hash
    commit_hash = (
        subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    )

    # Save combined data and metadata
    meta_data = {
        "name": name,
        "date": current_date,
        "commit_hash": commit_hash,
        "equation": equation_str,
        "input_variables": input_vars,
        "parameters": params,
        "parameter_values": param_values,
        "input_variable_ranges": var_ranges,
        "sample_size_train": sample_size_train,
        "sample_size_test": sample_size_test,
        "noise_level": noise_level,
        "correlation": correlation,
        "sample_type": sample_type,
        "seed": seed,
        "current_date": current_date,
    }

    meta_df = pd.DataFrame([meta_data])
    meta_csv = f"{dataset_name}_meta.csv"
    meta_df.to_csv(data_folder / meta_csv, index=False)

    print(f"Data saved to {combined_csv}")
    print(f"Metadata saved to {meta_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate data based on equations in a JSON file."
    )
    parser.add_argument(
        "file_path", type=str, help="Path to the JSON file containing equations."
    )
    parser.add_argument("name", type=str, help="Name of the equation to use.")
    parser.add_argument("seed", type=int, help="Random seed for reproducibility.")
    parser.add_argument(
        "noise_level", type=float, help="Noise level to add to the data."
    )
    parser.add_argument(
        "sample_size_train", type=int, help="Number of training samples to generate."
    )
    parser.add_argument(
        "sample_size_test", type=int, help="Number of test samples to generate."
    )
    parser.add_argument(
        "--sample_type", type=str, help="Type of sample to generate.", default="uniform"
    )
    parser.add_argument(
        "--correlation",
        type=float,
        help="Correlation in the input variables.",
        default=0,
    )
    parser.add_argument(
        "--data_folder",
        type=str,
        help="Path to the folder where data should be stored.",
        default="data",
    )

    args = parser.parse_args()

    generate_data(
        file_path=Path(args.file_path),
        name=args.name,
        seed=args.seed,
        noise_level=args.noise_level,
        correlation=args.correlation,
        sample_type=args.sample_type,
        sample_size_train=args.sample_size_train,
        sample_size_test=args.sample_size_test,
    )
