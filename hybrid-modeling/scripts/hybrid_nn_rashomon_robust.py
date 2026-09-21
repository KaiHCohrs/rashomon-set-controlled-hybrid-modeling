"""
Run Rashomon-robust hybrid model: hybrid JNN with Rashomon-set fitting routine.
Epsilon and Rashomon set are defined in MSE. Results are stored like hybrid_nn;
lambda runs are stored with min MSE, Rashomon threshold, and diameter.
No plotting; suitable for supplementary material.
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import time

from hybridmodeling.models.models import RashomonRobustEnsembleHybridJNN
from hybridmodeling.utility.helpers import all_scores, convert_wd
import json
import yaml


def get_epsilon_from_aggregate(
    aggregate_path,
    problem,
    noise_level,
    sample_size,
    default_epsilon=0.01,
    setup=None,
    seed=None,
):
    """Look up std_MSE_obs_train_inliers from aggregate CSV; return 2 * that as epsilon (MSE)."""
    if aggregate_path is None or not Path(aggregate_path).exists():
        return default_epsilon
    try:
        agg = pd.read_csv(aggregate_path)
    except Exception:
        return default_epsilon
    mask = (
        (agg["problem"] == problem)
        & (agg["noise_level"] == noise_level)
        & (agg["sample_size"] == sample_size)
    )
    if setup is not None and "setup" in agg.columns:
        mask = mask & (agg["setup"] == setup)
    if seed is not None and "seed" in agg.columns:
        mask = mask & (agg["seed"] == seed)
    row = agg[mask]
    if row.empty or "std_MSE_obs_train_inliers" not in agg.columns:
        return default_epsilon
    eps = float(row["std_MSE_obs_train_inliers"].iloc[0])
    if pd.isna(eps) or eps <= 0:
        return default_epsilon
    return eps


def get_delta_from_partial_physics_aggregate(
    aggregate_path, problem, noise_level, sample_size, epsilon, setup=None, seed=None
):
    """Look up hessian_train_loss_autodiff from partial_physics aggregate; return (delta, hessian)."""
    if aggregate_path is None or not Path(aggregate_path).exists():
        return np.nan, np.nan
    try:
        agg = pd.read_csv(aggregate_path)
    except Exception:
        return np.nan, np.nan
    mask = (
        (agg["problem"] == problem)
        & (agg["noise_level"] == noise_level)
        & (agg["sample_size"] == sample_size)
    )
    if setup is not None and "setup" in agg.columns:
        mask = mask & (agg["setup"] == setup)
    if seed is not None and "seed" in agg.columns:
        mask = mask & (agg["seed"] == seed)
    row = agg[mask]
    if row.empty or "hessian_train_loss_autodiff" not in row.columns:
        return np.nan, np.nan
    h = float(row["hessian_train_loss_autodiff"].iloc[0])
    if pd.isna(h) or h <= 0 or not np.isfinite(epsilon) or epsilon <= 0:
        return np.nan, (h if not pd.isna(h) else np.nan)
    delta = np.sqrt(3.0) * np.sqrt(2.0 * epsilon / h)
    return delta, h


def main(
    noise,
    sample_size,
    l2_net_decay,
    file_name,
    equation_name,
    equation_number,
    equation_version,
    config,
    index,
    optimizer,
    data_file_name,
    seed,
    max_diameter,
    h_lambda,
    aggregate_csv,
    default_epsilon,
    regularizer="l2_net",
    partial_physics_aggregate=None,
    delta_constant=None,
    setup=None,
    lambda_min=1e-10,
    overwrite=False,
):
    if data_file_name is None:
        file = f"{equation_name}_{equation_number}_correlation0.5u_noise{noise}_train{sample_size}_data.csv"
    else:
        file = data_file_name
    base_path = Path(__file__).parent.parent
    data_path = base_path.parent / "sr-data-factory" / "data" / file

    if index is None:
        index = pd.Timestamp.now().strftime("%Y%m%d%H%M%S")

    results_path = (
        base_path
        / "results"
        / f"{equation_name}_{equation_number}_{equation_version}_rashomon_robust_results_eq_{index}.csv"
    )
    lambda_runs_path = (
        base_path
        / "results"
        / f"{equation_name}_{equation_number}_{equation_version}_rashomon_robust_lambda_runs_eq_{index}.csv"
    )

    if results_path.exists():
        df = pd.read_csv(results_path)
        if (
            not overwrite
            and df[
                (df["noise_level"] == noise) & (df["sample_size"] == sample_size)
            ].shape[0]
            > 0
        ):
            print(
                f"Results already exist for noise level {noise}, sample size {sample_size}"
            )
            return

    problem = f"{equation_name}_{equation_number}_{equation_version}"
    epsilon = get_epsilon_from_aggregate(
        aggregate_csv,
        problem,
        noise,
        sample_size,
        default_epsilon=default_epsilon,
        setup=setup,
        seed=seed,
    )
    print(
        f"Using Rashomon epsilon (MSE) = {epsilon} for problem={problem}, noise={noise}, sample_size={sample_size}"
    )

    def _compute_delta(chosen_epsilon):
        if delta_constant is not None and np.isfinite(delta_constant):
            return float(delta_constant), np.nan
        return get_delta_from_partial_physics_aggregate(
            partial_physics_aggregate,
            problem,
            noise,
            sample_size,
            chosen_epsilon,
            setup=setup,
            seed=seed,
        )

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

    equation_config = next(
        (
            c
            for c in equation_configs
            if c["name"] == f"{equation_name}_{equation_number}_{equation_version}"
        ),
        None,
    )
    if equation_config is None:
        raise ValueError(
            f"No equation config for {equation_name}_{equation_number}_{equation_version}"
        )

    for p, v in equation_config["parameter_values"].items():
        if p not in equation_config["parameters"]:
            equation_config["fp"] = equation_config["fp"].replace(p, str(v))
    for p, v in equation_config["parameter_values"].items():
        if p not in equation_config["parameters"]:
            equation_config["type"] = equation_config["type"].replace(p, str(v))

    hybrid_nn = RashomonRobustEnsembleHybridJNN(
        model_config,
        trainer_config,
        equation_config,
        max_diameter=max_diameter,
        h_lambda=h_lambda,
        rashomon_epsilon=epsilon,
        regularizer=regularizer,
        seed=model_config.get("seed", 1),
        lambda_min=lambda_min,
    )

    data = pd.read_csv(data_path)
    input_variables = equation_config["input_variables"]
    X_train = data[data["scenario"] == "train"][input_variables].values
    X_test = data[data["scenario"] == "test"][input_variables].values
    X_test_x0OOD = data[data["scenario"] == "test_x0"][input_variables].values
    X_test_x1OOD = data[data["scenario"] == "test_x1"][input_variables].values
    X_test_x0x1OOD = data[data["scenario"] == "test_x0x1"][input_variables].values
    Y_train = data[data["scenario"] == "train"][["y_obs"]].values
    Y_test = data[data["scenario"] == "test"][["y_obs"]].values
    Y_test_x0OOD = data[data["scenario"] == "test_x0"][["y_obs"]].values
    Y_test_x1OOD = data[data["scenario"] == "test_x1"][["y_obs"]].values
    Y_test_x0x1OOD = data[data["scenario"] == "test_x0x1"][["y_obs"]].values

    hybrid_nn.fit(X_train, Y_train, noise_level=noise, compute_delta=_compute_delta)
    epsilon = hybrid_nn.rashomon_epsilon
    delta, hessian_autodiff = _compute_delta(epsilon)
    if not np.isfinite(delta) or delta <= 0:
        delta = hybrid_nn.max_diameter
        hessian_autodiff = np.nan

    print(
        f"\n[Rashomon] Summary: max_diameter={hybrid_nn.max_diameter}  epsilon={epsilon}\n"
    )

    l2_loss = hybrid_nn.v_l2_loss()
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

    Y_train_clean = data[data["scenario"] == "train"][["y"]].values
    Y_test_clean = data[data["scenario"] == "test"][["y"]].values
    Y_test_x0OOD_clean = data[data["scenario"] == "test_x0"][["y"]].values
    Y_test_x1OOD_clean = data[data["scenario"] == "test_x1"][["y"]].values
    Y_test_x0x1OOD_clean = data[data["scenario"] == "test_x0x1"][["y"]].values

    score_clean_train, NRMSEr_clean_train, NRMSEm_clean_train = all_scores(
        hybrid_nn, X_train, Y_train_clean
    )
    score_clean_test, NRMSEr_clean_test, NRMSEm_clean_test = all_scores(
        hybrid_nn, X_test, Y_test_clean
    )
    (
        score_clean_test_x0OOD,
        NRMSEr_clean_test_x0OOD,
        NRMSEm_clean_test_x0OOD,
    ) = all_scores(hybrid_nn, X_test_x0OOD, Y_test_x0OOD_clean)
    (
        score_clean_test_x1OOD,
        NRMSEr_clean_test_x1OOD,
        NRMSEm_clean_test_x1OOD,
    ) = all_scores(hybrid_nn, X_test_x1OOD, Y_test_x1OOD_clean)
    (
        score_clean_test_x0x1OOD,
        NRMSEr_clean_test_x0x1OOD,
        NRMSEm_clean_test_x0x1OOD,
    ) = all_scores(hybrid_nn, X_test_x0x1OOD, Y_test_x0x1OOD_clean)

    inferred_parameter = hybrid_nn.params_best
    df = pd.DataFrame(
        np.array(inferred_parameter),
        columns=list(equation_config["parameter_values"].keys()),
    )
    for p, v in equation_config["parameter_values"].items():
        if p not in equation_config["parameters"]:
            df[p] = None

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
    ) = (
        score_clean_test[0],
        score_clean_test[1],
        NRMSEr_clean_test,
        NRMSEm_clean_test,
    )
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
    ) = (
        score_obs_train[0],
        score_obs_train[1],
        NRMSEr_obs_train,
        NRMSEm_obs_train,
    )
    (
        df["RMSE_obs_test"],
        df["R2_obs_test"],
        df["NRMSEr_obs_test"],
        df["NRMSEm_obs_test"],
    ) = (
        score_obs_test[0],
        score_obs_test[1],
        NRMSEr_obs_test,
        NRMSEm_obs_test,
    )
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

    params_arr = np.array(hybrid_nn.params_best)
    param_names = list(equation_config["parameter_values"].keys())
    best_idx = int(hybrid_nn.rashomon_best_idx)
    in_idx = np.asarray(hybrid_nn.rashomon_indices)
    best_params = params_arr[best_idx]
    if len(in_idx) > 0:
        params_in_set = params_arr[in_idx]
        rashomon_min = np.min(params_in_set, axis=0)
        rashomon_max = np.max(params_in_set, axis=0)
    else:
        rashomon_min = best_params.copy()
        rashomon_max = best_params.copy()
    for j, p in enumerate(param_names):
        df[f"{p}_best"] = best_params[j]
        df[f"{p}_rashomon_min"] = rashomon_min[j]
        df[f"{p}_rashomon_max"] = rashomon_max[j]

    df["l2_loss"] = l2_loss
    df["noise_level"] = noise
    df["sample_size"] = sample_size
    df["epsilon_estimated"] = epsilon
    df["rashomon_epsilon"] = epsilon
    df["delta_estimated"] = delta
    df["delta"] = delta
    df["hessian_train_loss_autodiff"] = (
        hessian_autodiff if np.isfinite(hessian_autodiff) else np.nan
    )
    df["max_diameter"] = hybrid_nn.max_diameter
    df["h_lambda"] = h_lambda
    df["regularizer"] = regularizer
    df["final_lambda"] = getattr(hybrid_nn, "_final_lambda", None)
    in_rashomon_mask = np.zeros(model_config["ensemble_size"], dtype=bool)
    in_rashomon_mask[in_idx] = True
    df["in_rashomon"] = in_rashomon_mask
    df["termination_case"] = getattr(hybrid_nn, "_termination_case", None)
    if optimizer is not None:
        df["optimizer"] = trainer_config.get("optimizer", "adam")
        df["batch_size"] = trainer_config["batch_size"]
        df["init_lr_params"] = trainer_config["init_lr_params"]
        df["init_lr_net"] = trainer_config["init_lr_net"]
    df["i"] = np.arange(model_config["ensemble_size"])
    df["iter_best"] = np.array(hybrid_nn.it_best)

    time.sleep(np.random.randint(1, 60))
    df.to_csv(results_path, mode="a", header=not results_path.exists(), index=False)

    runs_rows = []
    for run in hybrid_nn.lambda_runs:
        lam = run["lambda"]
        min_mse = run["min_performance"]
        threshold = min_mse + epsilon
        diameter = run["diameter"]
        n_in_rashomon = int(np.sum(run["in_rashomon"]))
        runs_rows.append(
            {
                "equation_name": equation_name,
                "equation_number": equation_number,
                "equation_version": equation_version,
                "noise_level": noise,
                "sample_size": sample_size,
                "rashomon_epsilon": epsilon,
                "regularizer": regularizer,
                "lambda": lam,
                "k": run.get("k"),
                "min_MSE_obs_train": min_mse,
                "rashomon_threshold": threshold,
                "diameter": diameter,
                "n_in_rashomon": n_in_rashomon,
            }
        )
    runs_df = pd.DataFrame(runs_rows)
    runs_df.to_csv(
        lambda_runs_path, mode="a", header=not lambda_runs_path.exists(), index=False
    )

    check_dir = results_path.parent / "check_lists"
    check_dir.mkdir(parents=True, exist_ok=True)
    with open(check_dir / f"{index}.txt", "a") as f:
        f.write(
            f"{equation_name}, {equation_number}, {equation_version}, {noise}, {sample_size}, rashomon_robust, success\n"
        )

    print(f"Saved main results to {results_path}")
    print(f"Saved lambda runs ({len(runs_rows)} runs) to {lambda_runs_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Rashomon-robust hybrid NN experiment"
    )
    parser.add_argument(
        "--noise", type=float, required=True, help="Noise level in the data"
    )
    parser.add_argument(
        "--sample_size", type=int, required=True, help="Sample size for training"
    )
    parser.add_argument(
        "--l2_net_decay", type=str, default="0.0", help="L2 variance regularization on network output (l2_net_decay)"
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
        "--index", type=str, default=None, help="Index to mark experiment"
    )
    parser.add_argument(
        "--optimizer", type=str, default=None, help="Optimizer for the model"
    )
    parser.add_argument(
        "--data_file_name", type=str, default=None, help="Name of the data file"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing results"
    )
    parser.add_argument("--seed", type=int, default=33, help="Seed for the experiment")
    parser.add_argument(
        "--setup", type=str, default=None, help="Dataset setup for aggregate lookup"
    )
    parser.add_argument(
        "--max_diameter",
        type=float,
        default=0.1,
        help="Fallback max Rashomon set diameter",
    )
    parser.add_argument(
        "--h_lambda", type=float, default=0.5, help="Regularization reduction factor"
    )
    parser.add_argument(
        "--aggregate_csv",
        type=str,
        default=None,
        help="Path to aggregate CSV for epsilon (std_MSE_obs_train_inliers)",
    )
    parser.add_argument(
        "--default_epsilon",
        type=float,
        default=0.01,
        help="Default epsilon if aggregate lookup fails",
    )
    parser.add_argument(
        "--regularizer",
        type=str,
        default="l2_net",
        choices=("l1", "l2", "l1_net", "l2_net"),
    )
    parser.add_argument(
        "--partial_physics_aggregate",
        type=str,
        default=None,
        help="Path to partial_physics aggregate for delta",
    )
    parser.add_argument(
        "--delta_constant",
        type=float,
        default=None,
        help="If set, use this constant for delta",
    )
    parser.add_argument("--lambda_min", type=float, default=1e-10)
    args = parser.parse_args()
    args.l2_net_decay = convert_wd(args.l2_net_decay)

    base_path = Path(__file__).parent.parent
    if args.aggregate_csv is None:
        args.aggregate_csv = base_path / "results" / "full_nn_aggregate_for_epsilon.csv"
    else:
        args.aggregate_csv = Path(args.aggregate_csv)
    if args.partial_physics_aggregate is None:
        args.partial_physics_aggregate = (
            base_path / "results" / "partial_physics_aggregate_for_delta.csv"
        )
    else:
        args.partial_physics_aggregate = Path(args.partial_physics_aggregate)

    main(
        noise=args.noise,
        sample_size=args.sample_size,
        l2_net_decay=args.l2_net_decay,
        file_name=args.file_name,
        equation_name=args.equation_name,
        equation_number=args.equation_number,
        equation_version=args.equation_version,
        config=args.config,
        index=args.index,
        optimizer=args.optimizer,
        data_file_name=args.data_file_name,
        seed=args.seed,
        max_diameter=args.max_diameter,
        h_lambda=args.h_lambda,
        aggregate_csv=args.aggregate_csv,
        default_epsilon=args.default_epsilon,
        regularizer=args.regularizer,
        partial_physics_aggregate=args.partial_physics_aggregate,
        delta_constant=args.delta_constant,
        setup=args.setup,
        lambda_min=args.lambda_min,
        overwrite=args.overwrite,
    )
