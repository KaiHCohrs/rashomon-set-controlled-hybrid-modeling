import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from hybridmodeling.models.models import (
    EnsembleHybridJNN,
    EnsembleJNN,
    EnsemblePartialPhysics,
)
from hybridmodeling.utility.helpers import all_scores, convert_wd
from scipy.stats import norm
import json
import yaml
import jax.numpy as jnp
import time


def main(
    noise,
    sample_size,
    l2_net_decay,
    file_name,
    equation_name,
    equation_number,
    equation_version,
    config,
    model_type,
    index,
    optimizer,
    data_file_name,
    seed,
):
    # Define paths and load data
    if data_file_name is None:
        file = f"{equation_name}_{equation_number}_correlation0.5u_noise{noise}_train{sample_size}_data.csv"
    else:
        file = data_file_name
    data_path = Path(__file__).parent.parent.parent / "sr-data-factory" / "data" / file
    base_path = Path(__file__).parent.parent

    if index is None:
        index = pd.Timestamp.now().strftime("%Y%m%d%H%M%S")

    data = pd.read_csv(data_path)

    # Define results path
    if model_type == "hybrid_nn":
        results_path = (
            base_path
            / "results"
            / f"{equation_name}_{equation_number}_{equation_version}_NN_results_{index}.csv"
        )
    else:
        results_path = (
            base_path
            / "results"
            / f"{equation_name}_{equation_number}_{equation_version}_ablation_results_{index}.csv"
        )

    # Check if results already exist
    if results_path.exists():
        df = pd.read_csv(results_path)
        if args.overwrite:
            print(
                f"Overwriting results for noise level {noise}, sample size {sample_size} and l2_net_decay {l2_net_decay}"
            )
        else:
            if model_type != "hybrid_nn":
                if (
                    df[
                        (df["noise_level"] == noise)
                        & (df["sample_size"] == sample_size)
                        & (df["model_type"] == model_type)
                    ].shape[0]
                    > 0
                ):
                    print(
                        f"Results already exist for noise level {noise}, sample size {sample_size} and model_type {model_type}"
                    )
                    return
            else:
                if (
                    df[
                        (df["noise_level"] == noise)
                        & (df["sample_size"] == sample_size)
                        & (df["l2_net_decay"] == l2_net_decay)
                    ].shape[0]
                    > 0
                ):
                    print(
                        f"Results already exist for noise level {noise}, sample size {sample_size} and l2_net_decay {l2_net_decay}"
                    )
                    return

    # Load configurations
    equation_configs = json.load(open(base_path / f"data/{file_name}"))
    configs = yaml.load(
        open(base_path / f"configurations/{config}"), Loader=yaml.FullLoader
    )
    model_config = configs["model_config"]
    trainer_config = configs["trainer_config"]

    if seed is not None:
        model_config["seed"] = seed
    if l2_net_decay is not None:
        trainer_config["l2_net_decay"] = l2_net_decay

    # Load and adjust equation configuration
    equation_config = next(
        (
            config
            for config in equation_configs
            if config["name"] == f"{equation_name}_{equation_number}_{equation_version}"
        ),
        None,
    )

    # The parameters that are not in the equation_config['parameters'] are considered as fixed parameters
    for p, v in equation_config["parameter_values"].items():
        if p not in equation_config["parameters"]:
            equation_config["fp"] = equation_config["fp"].replace(p, str(v))

    # Set up model type-specific configurations
    if model_type == "full_nn":
        equation_config["type"] = "fa"
        inputs = sum(
            [1 for i in range(10) if f"x{i}" in equation_config["equation_gt"]]
        )
        equation_config["fa_input"] = list(range(inputs))
        equation_config["parameters"] = []
    elif model_type == "partial_physics":
        equation_config["type"] = equation_config["partial_physics"]
        equation_config["parameters"].append("b0")

    for p, v in equation_config["parameter_values"].items():
        if p not in equation_config["parameters"]:
            equation_config["type"] = equation_config["type"].replace(p, str(v))

    # Initialize the model
    if model_type in ["hybrid_nn"]:
        hybrid_nn = EnsembleHybridJNN(model_config, trainer_config, equation_config)
    elif model_type in ["full_nn"]:
        hybrid_nn = EnsembleJNN(model_config, trainer_config, equation_config)
    else:
        hybrid_nn = EnsemblePartialPhysics(model_config, equation_config)

    # Get input variables and training/testing data
    input_variables = equation_config["input_variables"]
    X_train = data[data["scenario"] == "train"][input_variables].values
    X_test = data[data["scenario"] == "test"][input_variables].values
    X_test_x0OOD = data[data["scenario"] == "test_x0"][input_variables].values
    X_test_x1OOD = data[data["scenario"] == "test_x1"][input_variables].values
    X_test_x0x1OOD = data[data["scenario"] == "test_x0x1"][input_variables].values

    Y_test = data[data["scenario"] == "test"][["y_obs"]].values
    Y_train = data[data["scenario"] == "train"][["y_obs"]].values
    Y_test_x0OOD = data[data["scenario"] == "test_x0"][["y_obs"]].values
    Y_test_x1OOD = data[data["scenario"] == "test_x1"][["y_obs"]].values
    Y_test_x0x1OOD = data[data["scenario"] == "test_x0x1"][["y_obs"]].values

    # Fit the model on the training data
    hybrid_nn.fit(X_train, Y_train)

    # Get l2_loss
    l2_loss = hybrid_nn.v_l2_loss()

    if model_type in ["partial_physics"]:
        Y_train = Y_train.flatten()
        Y_test = Y_test.flatten()
        Y_test_x0OOD = Y_test_x0OOD.flatten()
        Y_test_x1OOD = Y_test_x1OOD.flatten()
        Y_test_x0x1OOD = Y_test_x0x1OOD.flatten()

    # Score the model on the observed data
    score_obs_train, NRMSEr_obs_train, NRMSEm_obs_train = all_scores(
        hybrid_nn, X_train, Y_train
    )
    score_obs_test, NRMSEr_obs_test, NRMSEm_obs_test = all_scores(
        hybrid_nn, X_test, Y_test
    )
    score_obs_test_x0OOD, NRMSEr_obs_test_x0OOD, NRMSEm_obs_test_x0OOD = all_scores(
        hybrid_nn, X_test_x0OOD, Y_test_x0OOD
    )
    score_obs_test_x1OOD, NRMSEr_obs_test_x1OOD, NRMSEm_obs_test_x1OOD = all_scores(
        hybrid_nn, X_test_x1OOD, Y_test_x1OOD
    )
    (
        score_obs_test_x0x1OOD,
        NRMSEr_obs_test_x0x1OOD,
        NRMSEm_obs_test_x0x1OOD,
    ) = all_scores(hybrid_nn, X_test_x0x1OOD, Y_test_x0x1OOD)

    print(
        f"Train RMSE (obs): {score_obs_train[0].mean()} +/- {score_obs_train[0].std()} (min: {score_obs_train[0].min()}, max: {score_obs_train[0].max()})"
    )
    print(
        f"Train R2 (obs): {score_obs_train[1].mean()} +/- {score_obs_train[1].std()} (min: {score_obs_train[1].min()}, max: {score_obs_train[1].max()})"
    )

    Y_test = data[data["scenario"] == "test"][["y"]].values
    Y_train = data[data["scenario"] == "train"][["y"]].values
    Y_test_x0OOD = data[data["scenario"] == "test_x0"][["y"]].values
    Y_test_x1OOD = data[data["scenario"] == "test_x1"][["y"]].values
    Y_test_x0x1OOD = data[data["scenario"] == "test_x0x1"][["y"]].values

    if model_type in ["partial_physics"]:
        Y_train = Y_train.flatten()
        Y_test = Y_test.flatten()
        Y_test_x0OOD = Y_test_x0OOD.flatten()
        Y_test_x1OOD = Y_test_x1OOD.flatten()
        Y_test_x0x1OOD = Y_test_x0x1OOD.flatten()

    # Score the model on the clean data
    score_clean_train, NRMSEr_clean_train, NRMSEm_clean_train = all_scores(
        hybrid_nn, X_train, Y_train
    )
    score_clean_test, NRMSEr_clean_test, NRMSEm_clean_test = all_scores(
        hybrid_nn, X_test, Y_test
    )
    (
        score_clean_test_x0OOD,
        NRMSEr_clean_test_x0OOD,
        NRMSEm_clean_test_x0OOD,
    ) = all_scores(hybrid_nn, X_test_x0OOD, Y_test_x0OOD)
    (
        score_clean_test_x1OOD,
        NRMSEr_clean_test_x1OOD,
        NRMSEm_clean_test_x1OOD,
    ) = all_scores(hybrid_nn, X_test_x1OOD, Y_test_x1OOD)
    (
        score_clean_test_x0x1OOD,
        NRMSEr_clean_test_x0x1OOD,
        NRMSEm_clean_test_x0x1OOD,
    ) = all_scores(hybrid_nn, X_test_x0x1OOD, Y_test_x0x1OOD)

    # Print the score
    print(
        f"Train RMSE (clean): {score_clean_train[0].mean()} +/- {score_clean_train[0].std()} (min: {score_clean_train[0].min()}, max: {score_clean_train[0].max()})"
    )
    print(
        f"Train NRMSE (clean) range: {NRMSEr_clean_train.mean()} +/- {NRMSEr_clean_train.std()} (min: {NRMSEr_clean_train.min()}, max: {NRMSEr_clean_train.max()})"
    )
    print(
        f"Train NRMSE (clean) mean: {NRMSEm_clean_train.mean()} +/- {NRMSEm_clean_train.std()} (min: {NRMSEm_clean_train.min()}, max: {NRMSEm_clean_train.max()})"
    )
    print(
        f"Train R2 (clean): {score_clean_train[1].mean()} +/- {score_clean_train[1].std()} (min: {score_clean_train[1].min()}, max: {score_clean_train[1].max()})"
    )
    print(
        f"Test RMSE (ID): {score_clean_test[0].mean()} +/- {score_clean_test[0].std()} (min: {score_clean_test[0].min()}, max: {score_clean_test[0].max()})"
    )
    print(
        f"Test NRMSE (ID) range: {NRMSEr_clean_test.mean()} +/- {NRMSEr_clean_test.std()} (min: {NRMSEr_clean_test.min()}, max: {NRMSEr_clean_test.max()})"
    )
    print(
        f"Test NRMSE (ID) mean: {NRMSEm_clean_test.mean()} +/- {NRMSEm_clean_test.std()} (min: {NRMSEm_clean_test.min()}, max: {NRMSEm_clean_test.max()})"
    )
    print(
        f"Test R2 (ID): {score_clean_test[1].mean()} +/- {score_clean_test[1].std()} (min: {score_clean_test[1].min()}, max: {score_clean_test[1].max()})"
    )
    print(
        f"Test RMSE (OOD x0): {score_clean_test_x0OOD[0].mean()} +/- {score_clean_test_x0OOD[0].std()} (min: {score_clean_test_x0OOD[0].min()}, max: {score_clean_test_x0OOD[0].max()})"
    )
    print(
        f"Test NRMSE (OOD x0) range: {NRMSEr_clean_test_x0OOD.mean()} +/- {NRMSEr_clean_test_x0OOD.std()} (min: {NRMSEr_clean_test_x0OOD.min()}, max: {NRMSEr_clean_test_x0OOD.max()})"
    )
    print(
        f"Test NRMSE (OOD x0) mean: {NRMSEm_clean_test_x0OOD.mean()} +/- {NRMSEm_clean_test_x0OOD.std()} (min: {NRMSEm_clean_test_x0OOD.min()}, max: {NRMSEm_clean_test_x0OOD.max()})"
    )
    print(
        f"Test R2 (OOD x0): {score_clean_test_x0OOD[1].mean()} +/- {score_clean_test_x0OOD[1].std()} (min: {score_clean_test_x0OOD[1].min()}, max: {score_clean_test_x0OOD[1].max()})"
    )
    print(
        f"Test RMSE (OOD x1): {score_clean_test_x1OOD[0].mean()} +/- {score_clean_test_x1OOD[0].std()} (min: {score_clean_test_x1OOD[0].min()}, max: {score_clean_test_x1OOD[0].max()})"
    )
    print(
        f"Test NRMSE (OOD x1) range: {NRMSEr_clean_test_x1OOD.mean()} +/- {NRMSEr_clean_test_x1OOD.std()} (min: {NRMSEr_clean_test_x1OOD.min()}, max: {NRMSEr_clean_test_x1OOD.max()})"
    )
    print(
        f"Test NRMSE (OOD x1) mean: {NRMSEm_clean_test_x1OOD.mean()} +/- {NRMSEm_clean_test_x1OOD.std()} (min: {NRMSEm_clean_test_x1OOD.min()}, max: {NRMSEm_clean_test_x1OOD.max()})"
    )
    print(
        f"Test R2 (OOD x1): {score_clean_test_x1OOD[1].mean()} +/- {score_clean_test_x1OOD[1].std()} (min: {score_clean_test_x1OOD[1].min()}, max: {score_clean_test_x1OOD[1].max()})"
    )
    print(
        f"Test RMSE (OOD x0x1): {score_clean_test_x0x1OOD[0].mean()} +/- {score_clean_test_x0x1OOD[0].std()} (min: {score_clean_test_x0x1OOD[0].min()}, max: {score_clean_test_x0x1OOD[0].max()})"
    )
    print(
        f"Test NRMSE (OOD x0x1) range: {NRMSEr_clean_test_x0x1OOD.mean()} +/- {NRMSEr_clean_test_x0x1OOD.std()} (min: {NRMSEr_clean_test_x0x1OOD.min()}, max: {NRMSEr_clean_test_x0x1OOD.max()})"
    )
    print(
        f"Test NRMSE (OOD x0x1) mean: {NRMSEm_clean_test_x0x1OOD.mean()} +/- {NRMSEm_clean_test_x0x1OOD.std()} (min: {NRMSEm_clean_test_x0x1OOD.min()}, max: {NRMSEm_clean_test_x0x1OOD.max()})"
    )
    print(
        f"Test R2 (OOD x0x1): {score_clean_test_x0x1OOD[1].mean()} +/- {score_clean_test_x0x1OOD[1].std()} (min: {score_clean_test_x0x1OOD[1].min()}, max: {score_clean_test_x0x1OOD[1].max()})"
    )

    if model_type in ["partial_physics", "hybrid_nn"]:
        inferred_parameter = hybrid_nn.params_best
    else:
        inferred_parameter = np.zeros(
            (model_config["ensemble_size"], len(equation_config["parameter_values"]))
        )
    # Store all performances with the inferred parameter values per index in a dataframe
    df = pd.DataFrame(
        inferred_parameter, columns=equation_config["parameter_values"].keys()
    )

    for p, v in equation_config["parameter_values"].items():
        if p not in equation_config["parameters"]:
            df[p] = None

    # save clean scores
    (
        df["RMSE_clean_train"],
        df["R2_clean_train"],
        df["NRMSEr_clean_train"],
        df["NRMSEm_clean_train"],
    ) = (
        score_clean_train[0],
        score_clean_train[1],
        NRMSEr_clean_train,
        NRMSEm_clean_train,
    )
    (
        df["RMSE_clean_test"],
        df["R2_clean_test"],
        df["NRMSEr_clean_test"],
        df["NRMSEm_clean_test"],
    ) = (score_clean_test[0], score_clean_test[1], NRMSEr_clean_test, NRMSEm_clean_test)
    (
        df["RMSE_clean_test_x0OOD"],
        df["R2_clean_test_x0OOD"],
        df["NRMSEr_clean_test_x0OOD"],
        df["NRMSEm_clean_test_x0OOD"],
    ) = (
        score_clean_test_x0OOD[0],
        score_clean_test_x0OOD[1],
        NRMSEr_clean_test_x0OOD,
        NRMSEm_clean_test_x0OOD,
    )
    (
        df["RMSE_clean_test_x1OOD"],
        df["R2_clean_test_x1OOD"],
        df["NRMSEr_clean_test_x1OOD"],
        df["NRMSEm_clean_test_x1OOD"],
    ) = (
        score_clean_test_x1OOD[0],
        score_clean_test_x1OOD[1],
        NRMSEr_clean_test_x1OOD,
        NRMSEm_clean_test_x1OOD,
    )
    (
        df["RMSE_clean_test_x0x1OOD"],
        df["R2_clean_test_x0x1OOD"],
        df["NRMSEr_clean_test_x0x1OOD"],
        df["NRMSEm_clean_test_x0x1OOD"],
    ) = (
        score_clean_test_x0x1OOD[0],
        score_clean_test_x0x1OOD[1],
        NRMSEr_clean_test_x0x1OOD,
        NRMSEm_clean_test_x0x1OOD,
    )

    (
        df["RMSE_obs_train"],
        df["R2_obs_train"],
        df["NRMSEr_obs_train"],
        df["NRMSEm_obs_train"],
    ) = (score_obs_train[0], score_obs_train[1], NRMSEr_obs_train, NRMSEm_obs_train)
    (
        df["RMSE_obs_test"],
        df["R2_obs_test"],
        df["NRMSEr_obs_test"],
        df["NRMSEm_obs_test"],
    ) = (score_obs_test[0], score_obs_test[1], NRMSEr_obs_test, NRMSEm_obs_test)
    (
        df["RMSE_obs_test_x0OOD"],
        df["R2_obs_test_x0OOD"],
        df["NRMSEr_obs_test_x0OOD"],
        df["NRMSEm_obs_test_x0OOD"],
    ) = (
        score_obs_test_x0OOD[0],
        score_obs_test_x0OOD[1],
        NRMSEr_obs_test_x0OOD,
        NRMSEm_obs_test_x0OOD,
    )
    (
        df["RMSE_obs_test_x1OOD"],
        df["R2_obs_test_x1OOD"],
        df["NRMSEr_obs_test_x1OOD"],
        df["NRMSEm_obs_test_x1OOD"],
    ) = (
        score_obs_test_x1OOD[0],
        score_obs_test_x1OOD[1],
        NRMSEr_obs_test_x1OOD,
        NRMSEm_obs_test_x1OOD,
    )
    (
        df["RMSE_obs_test_x0x1OOD"],
        df["R2_obs_test_x0x1OOD"],
        df["NRMSEr_obs_test_x0x1OOD"],
        df["NRMSEm_obs_test_x0x1OOD"],
    ) = (
        score_obs_test_x0x1OOD[0],
        score_obs_test_x0x1OOD[1],
        NRMSEr_obs_test_x0x1OOD,
        NRMSEm_obs_test_x0x1OOD,
    )

    df["l2_loss"] = l2_loss
    df["noise_level"] = noise
    df["sample_size"] = sample_size
    if model_type == "hybrid_nn":
        df["l2_net_decay"] = l2_net_decay
    else:
        df["model_type"] = model_type
    if optimizer is not None:
        df["optimizer"] = trainer_config.get("optimizer", "adam")
        df["batch_size"] = trainer_config["batch_size"]
        df["init_lr_params"] = trainer_config["init_lr_params"]
        df["init_lr_net"] = trainer_config["init_lr_net"]
    df["i"] = np.arange(model_config["ensemble_size"])
    df["iter_best"] = hybrid_nn.it_best

    # Write results to CSV
    df.to_csv(results_path, mode="a", header=not results_path.exists(), index=False)

    parameter_values = np.array(list(equation_config["parameter_values"].values()))
    bias = np.mean(inferred_parameter, axis=0) - parameter_values
    spread = np.std(inferred_parameter, axis=0)
    mae = np.mean(np.abs(inferred_parameter - parameter_values), axis=0)
    mse = np.mean((inferred_parameter - parameter_values) ** 2, axis=0)
    mad = np.median(
        np.abs(inferred_parameter - np.median(inferred_parameter, axis=0)), axis=0
    )
    cv = spread / np.mean(inferred_parameter, axis=0)
    n = inferred_parameter.shape[0]
    mean_inferred = np.mean(inferred_parameter, axis=0)
    ci_half_width = norm.ppf(0.975) * (spread / np.sqrt(n))
    ci_lower = mean_inferred - ci_half_width
    ci_upper = mean_inferred + ci_half_width

    if equation_config["parameters"] == []:
        print("No parameters to infer")
    else:
        indices = np.array(
            [
                list(equation_config["parameter_values"].keys()).index(p)
                for p in equation_config["parameters"]
            ]
        )
        print("Parameters:", equation_config["parameters"])
        print("True values:", parameter_values[indices])
        print("Mean:", mean_inferred[indices])
        print("Bias:", bias[indices])
        print("Spread (Standard Deviation):", spread[indices])
        print("Mean Absolute Error (MAE):", mae[indices])
        print("Mean Squared Error (MSE):", mse[indices])
        print("Median Absolute Deviation (MAD):", mad[indices])
        print("Coefficient of Variation (CV):", cv[indices])
        print("95% Confidence Interval (CI):")
        print(" Lower bound:", ci_lower[indices])
        print(" Upper bound:", ci_upper[indices])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run hybrid modeling experiment")
    parser.add_argument(
        "--noise", type=float, required=True, help="Noise level in the data"
    )
    parser.add_argument(
        "--sample_size", type=int, required=True, help="Sample size for training"
    )
    parser.add_argument(
        "--l2_net_decay", type=str, default=0.0, help="L2 variance regularization on network output (l2_net_decay)"
    )
    parser.add_argument(
        "--file_name", type=str, required=True, help="Name of the file with equations"
    )
    parser.add_argument(
        "--equation_name", type=str, required=True, help="Name of the equation"
    )
    parser.add_argument(
        "--equation_number", type=str, required=True, help="Number of the equation"
    )
    parser.add_argument(
        "--equation_version", type=str, required=True, help="Version of the equation"
    )
    parser.add_argument("--config", type=str, required=True, help="Configuration file")
    parser.add_argument(
        "--model_type",
        type=str,
        default="hybrid_nn",
        help="Special types of training: full_nn, full_physics_wp, full_physics_wop, partial_physics, hybrid_nn",
    )
    parser.add_argument(
        "--index", type=str, default=None, help="Index to mark experiment."
    )
    parser.add_argument(
        "--optimizer", type=str, default=None, help="Optimizer for the model"
    )
    parser.add_argument(
        "--data_file_name",
        type=str,
        required=False,
        default=None,
        help="Name of the data file",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="overwrite existing results"
    )
    parser.add_argument("--seed", type=int, help="Seed for the experiment")

    args = parser.parse_args()
    args.l2_net_decay = convert_wd(args.l2_net_decay)

    main(
        noise=args.noise,
        sample_size=args.sample_size,
        l2_net_decay=args.l2_net_decay,
        file_name=args.file_name,
        equation_name=args.equation_name,
        equation_number=args.equation_number,
        equation_version=args.equation_version,
        config=args.config,
        model_type=args.model_type,
        index=args.index,
        optimizer=args.optimizer,
        data_file_name=args.data_file_name,
        seed=args.seed,
    )
