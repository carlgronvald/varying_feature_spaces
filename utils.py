import numpy as np
import pandas as pd
from typing import List

def index_to_features(index: int):
    """Turns an index into a list of features, by taking the unsigned binary representation."""
    features = []
    while index > 0:
        new_feature = int(np.log2(index))
        features += [new_feature]
        index -= 2**new_feature
    return features


def features_to_index(features: List[int]):
    """Turns a list of features into an index, by taking the sum of 2^feature."""
    index = 0
    for feature in features:
        index += 2**feature
    return index

def gen_synth_data(size, rho, seed, features, missing=True, return_coefs=False):
    """Generates synthetic dataset as described in the regression section of
    the thesis."""
    rng = np.random.default_rng(seed=seed)
    coefs = rng.uniform(low=0.1, high=1, size=(features))

    mean = np.zeros(features)
    covariance = np.zeros((features, features))

    # Construct covariance matrix with cyclical exponential decay
    for i in range(features):
        for j in range(features):
            dist = np.abs(i - j)
            if dist > features // 2:
                dist = features - dist

            covariance[i, j] = rho ** dist

    X = rng.multivariate_normal(mean, covariance, size=size)
    # Add constant feature
    X = np.hstack((X, np.ones((size, 1))))

    y = np.zeros(X.shape[0])
    for i in range(features):
        y += coefs[i] * X[:, i]

    y = y - np.median(y)

    # Remove missing features
    if missing:
        for t in range(X.shape[0] // 4, X.shape[0]):
            nans = rng.choice(
                range(X.shape[1] - 1),
                size=(X.shape[1] - 1) // 2,
                replace=False,
            )

            X[t, nans] = np.nan

    if return_coefs:
        return X, y, coefs
    else: 
        return X, y

def get_predictions(X, y, predictor, offset: int, reconstructive: bool = False, return_coefs=False):
    """Returns the predictions of a predictor on a dataset, doing `offset`
    time-step-ahead predictions. """
    predictions = np.zeros(X.shape[0]) * np.nan
    coefficients = None
    if return_coefs:
        coefficients = np.zeros(X.shape) * np.nan
    if reconstructive:
        for t in range(offset):
            predictor.update_reconstruction(X[t + offset, :])
    for t in range(X.shape[0]):
        if return_coefs:
            coefficients[t, :] = predictor.w
        if t + offset < X.shape[0]:
            predictions[t + offset] = predictor.predict(X[t + offset, :])
        else:
            break
        if reconstructive:
            predictor.update_reconstruction(X[t + offset, :])
            predictor.update_predictor(predictor.reconstruction_gradient(X[t,:])[0], y[t])
        else:
            predictor.fit(X[t, :], y[t])
    if return_coefs:
        return predictions, coefficients
    else:
        return predictions

def get_feature_reconstructions(X, y, predictor, offset: int):
    """Returns the feature reconstructions for a given reconstructive predictor
    and dataset."""
    predictions = np.zeros(X.shape[0]) * np.nan
    psi_X = np.zeros(X.shape) * np.nan
    for t in range(offset):
        predictor.update_reconstruction(X[t + offset, :])
    for t in range(X.shape[0]):
        psi_X[t, :], _ = predictor.reconstruction_gradient(X[t, :])
        if t + offset < X.shape[0]:
            predictions[t + offset] = predictor.predict(X[t + offset, :])
            predictor.update_reconstruction(X[t + offset, :])

        predictor.update_predictor(predictor.reconstruction_gradient(X[t,:])[0], y[t])
    return psi_X

def load_sc_dataset():
    """Loads the South Carolina wind dataset."""
    dataset = pd.read_csv(
        "c:\\users\\carlfrederik\\mine ting\\dtu\\thesis\\data\\southcarolina.csv",
        header=None,
    )
    dataset.drop(columns=[0], inplace=True)
    return dataset

def load_sc_data(seed : int, size : int = 10*24,missing=True):
    """Loads the South Carolina wind dataset. Chooses a random starting point
    based on the seed, and returns a dataset of the specified seed. The column
    is chosen based on modulo 9 of the seed, and the seed used for determining
    the starting point is the seed divided by 9, so that the seeds (9n, 9n+1,
    ..., 9n+8) return the same dataset with each of the 9 columns as
    targets."""
    chosen_column = np.abs(seed%9)+1
    seed = seed//9
    dataset = load_sc_dataset()
    rng = np.random.default_rng(seed=seed)
    start_point = rng.integers(dataset.shape[0] - size)
    dataset = (dataset - dataset.mean()) / dataset.std()
    dataset = dataset.iloc[start_point : start_point + size]
    y = dataset[chosen_column].to_numpy()
    X = dataset.to_numpy()
    X = np.vstack((X, np.zeros((1, X.shape[1]))))
    X = np.hstack((X, np.ones((X.shape[0], 1))))
    X = X[1:, :]
    if missing:
        for t in range(X.shape[0] // 2, X.shape[0]):
            nans = rng.choice(
                range(X.shape[1] - 1),
                size=(X.shape[1] - 1) // 2,
                replace=False,
            )
            X[t, nans] = np.nan
    return X,y

def load_uci_data(dataset_name):
    X = np.loadtxt(f"data/{dataset_name}_X.csv", delimiter=",")
    y = np.loadtxt(f"data/{dataset_name}_y.csv", delimiter=",")

    return X, y
