import numpy as np
import pandas as pd
from predictors import OLM, OLMS, OCDS, ModelType
from utils import gen_synth_data, get_predictions, index_to_features, features_to_index
import matplotlib.pyplot as plt


def run_experiment(features : int, rho : float, offset : int, permutations : int, predictor_filter = ["OLM", "OLM-S", "OCDS"]):
    losses = {}
    for i in range(permutations):
        X, y = gen_synth_data(200, rho, 914+i, features)
        predictors = {
            "OLM": OLM(X.shape[1], 0.01),
            "OLM-S": OLMS(X.shape[1], 0.01, 0.01),
            "OCDS": OCDS(X.shape[1], 0.1, 10),
        }
        predictors = {name: predictor for name, predictor in predictors.items() if name in predictor_filter}

        for name, predictor in predictors.items():
            reconstructive = False
            if name == "OLMS":
                reconstructive = True
            # predictions, coefficients = get_predictions(X, y, predictor, offset, reconstructive=reconstructive, return_coefs=True)
            predictions= get_predictions(X, y, predictor, offset, reconstructive=reconstructive)

            # print("predictions", predictions)
            # print("coefficients", coefficients)
            # plt.plot(np.cumsum((predictions[offset:]-y[offset:])**2)/np.arange(200-offset)+1, label="Predictions")
            # plt.plot(coefficients[offset:,:], alpha=0.5)
            # plt.title(f"{name} with rho={rho}")
            avg_loss = np.mean((predictions[offset:] - y[offset:]) ** 2)
            if name not in losses:
                losses[name] = []
            losses[name].append(avg_loss)
            # plt.show()

    return losses


# Correlation experiment
def correlation_experiment(features):
    experiment_results = []
    permutations = 100
    for rho in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]:
        print(f"Running experiment with rho={rho}")
        experiment = run_experiment(features, rho, 25, permutations)
        experiment_result = {}
        for name, losses in experiment.items():
            experiment_result[name] = np.mean(losses)
            experiment_result[name + "_conf"] = np.std(losses)*1.96/np.sqrt(permutations)
        experiment_result["rho"] = rho

        experiment_results.append(experiment_result)


    max_y = 0
    for predictor in ["OLM", "OLM-S", "OCDS"]:
        df = pd.DataFrame(experiment_results)
        df.to_csv(f"results_{predictor}_{features}.csv")
        plt.plot(df["rho"], df[predictor], label=predictor)
        plt.fill_between(df["rho"], df[predictor] - df[predictor + "_conf"], df[predictor] + df[predictor + "_conf"], alpha=0.5)
        if predictor in ["OLM", "OLM-S"]:
            max_y = max(max_y, max(df[predictor] + df[predictor + "_conf"]))
        # for i in range(df.shape[0]):
        #     plt.text(df["rho"][i], df[predictor][i], f"{df[predictor][i]:.2f}")
    plt.ylim(0, max_y*1.5)
    plt.grid(True, axis="y")
    plt.legend()
    plt.xlabel("rho")
    plt.ylabel("MSE")
    plt.title(f"MSE vs correlation, {features} features, 25 time step ahead predictions")
    plt.show()

def time_horizon_experiment(features):
    experiment_results = []
    permutations = 100
    for offset in [0,10,20,30,40,50,60,70,80,90,100]:
        print(f"Running experiment with offset={offset}")
        experiment = run_experiment(features, 0.99, offset, permutations)#,predictor_filter=["OLM", "OLM-S"])
        experiment_result = {}
        for name, losses in experiment.items():
            experiment_result[name] = np.mean(losses)
            experiment_result[name + "_conf"] = np.std(losses)*1.96/np.sqrt(permutations)
        experiment_result["offset"] = offset

        experiment_results.append(experiment_result)


    max_y = 0
    df = pd.DataFrame(experiment_results)
    print(df)
    for predictor in ["OLM", "OLM-S", "OCDS"]:
        plt.plot(df["offset"], df[predictor], label=predictor)
        plt.fill_between(df["offset"], df[predictor] - df[predictor + "_conf"], df[predictor] + df[predictor + "_conf"], alpha=0.5)
        max_y = max(max_y, max(df[predictor] + df[predictor + "_conf"]))
        # for i in range(df.shape[0]):
        #     plt.text(df["rho"][i], df[predictor][i], f"{df[predictor][i]:.2f}")
    plt.ylim(0, max_y*1.1)
    plt.grid(True, axis="y")
    plt.legend()
    plt.xlabel("Forecast horizon ΔT")
    plt.ylabel("MSE")
    plt.title(f"MSE vs forecast horizon, {features} features, rho=0.99")
    plt.show()

time_horizon_experiment(features=6)
correlation_experiment(features=6)
correlation_experiment(features=20)
