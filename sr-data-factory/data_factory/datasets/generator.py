import importlib
import numpy as np
import pandas as pd


def load_function(module_name, function_name):
    try:
        module = importlib.import_module(module_name)
        function = getattr(module, function_name)
        return function
    except ModuleNotFoundError:
        print(f"Module '{module_name}' not found.")
    except AttributeError:
        print(f"Function '{function_name}' not found in module '{module_name}'.")


def generate(config):
    np.random.seed(config["seed"])
    equation = load_function(config["module_name"], config["function_name"])

    dataset = pd.DataFrame()
    for i in range(config["samples"]):
        # Generate training data
        training_X = sample(
            config["training"]["input_distribution"],
            config["input_dimension"],
            config["training"]["sample_size"],
        )
        training_gt = equation(training_X)
        training_Y, scale = noise(training_gt, config["noise"])

        training_data = pd.DataFrame(
            np.concatenate(
                (training_X, training_gt[np.newaxis, :], training_Y[np.newaxis, :]),
                axis=0,
            ).T,
            columns=[f"X{i}" for i in range(training_X.shape[0])] + ["gt"] + ["Y"],
        )
        training_data["dataset"] = "training"

        # Generate testing data (ID)
        testing_ID_X = sample(
            config["training"]["input_distribution"],
            config["input_dimension"],
            config["testing_ID"]["sample_size"],
        )
        testing_ID_gt = equation(testing_ID_X)
        testing_ID_Y = noise(testing_ID_gt, config["noise"])
        testing_ID_data = pd.DataFrame(
            np.concatenate(
                (
                    testing_ID_X,
                    testing_ID_gt[np.newaxis, :],
                    testing_ID_Y[np.newaxis, :],
                ),
                axis=0,
            ).T,
            columns=[f"X{i}" for i in range(testing_ID_X.shape[0])] + ["gt"] + ["Y"],
        )
        testing_ID_data["dataset"] = "testing_ID"

        # Generate testing data (OOD)
        testing_OOD_X = sample(
            config["testing_OOD"]["input_distribution"],
            config["input_dimension"],
            config["testing_OOD"]["sample_size"],
        )
        testing_OOD_gt = equation(testing_OOD_X)
        testing_OOD_Y = noise(testing_OOD_gt, config["noise"])
        testing_OOD_data = pd.DataFrame(
            np.concatenate(
                (
                    testing_OOD_X,
                    testing_OOD_gt[np.newaxis, :],
                    testing_OOD_Y[np.newaxis, :],
                ),
                axis=0,
            ).T,
            columns=[f"X{i}" for i in range(testing_OOD_X.shape[0])] + ["gt"] + ["Y"],
        )
        testing_OOD_data["dataset"] = "testing_OOD"

        # Concatenate datasets
        dataset_temp = pd.concat(
            [training_data, testing_ID_data, testing_OOD_data], axis=0
        )
        dataset_temp["sample"] = i

        dataset = pd.concat([dataset, dataset_temp])

    return dataset


def sample(input_distribution, input_dimension, sample_size):
    if input_distribution["name"] == "uniform":
        return np.random.uniform(
            **input_distribution["parameter"], size=(input_dimension, sample_size)
        )
    elif input_distribution == "normal":
        return np.random.normal(
            **input_distribution["parameter"], size=(input_dimension, sample_size)
        )
    else:
        raise ValueError("Invalid input distribution")


def noise(Y, noise_config, scale=None):
    # if there is a scale parameter in noise_config, take the amplitude of Y and multiply it with the scale
    if noise_config.get("noise_fraction") is not None:
        scale = np.std(Y)
        if noise_config["name"] == "normal":
            return (
                Y
                + np.random.normal(
                    scale=noise_config["noise_fraction"] * scale, size=Y.shape
                ),
                scale,
            )
        elif noise_config["name"] == "uniform":
            return Y + np.random.uniform(
                low=-noise_config["noise_fraction"] * scale,
                high=noise_config["noise_fraction"] * scale,
                size=Y.shape,
            )
        else:
            raise ValueError("Invalid noise distribution")
    else:
        if noise_config["name"] == "normal":
            return Y + np.random.normal(**noise_config["parameter"], size=Y.shape)
        elif noise_config["name"] == "uniform":
            return Y + np.random.uniform(**noise_config["parameter"], size=Y.shape)
        else:
            raise ValueError("Invalid noise distribution")
