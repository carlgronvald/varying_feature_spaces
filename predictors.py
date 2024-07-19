import numpy as np
from typing import Set, List
from enum import Enum
import scipy

class ModelType(Enum):
    REGRESSION=0
    CLASSIFICATION=1

def logistic(x):
    print("Logistic")
    return 1 / (1 + np.exp(-x))

class OLM:
    """Online Linear Model."""

    def __init__(
        self,
        size: int,
        step_size: float,
        beta1: float = 0,
        beta2: float = 0,
        adaptive_step_size: bool = False,
        typ : ModelType = ModelType.REGRESSION
    ):
        self.w = np.zeros(size)
        self.step_size = step_size
        self.beta1 = beta1
        self.beta2 = beta2

        self.adaptive_step_size = adaptive_step_size
        self.sqr_gradient_sums_diag = np.ones(size)
        self.typ = typ

    def fit_predict(self, x_t: np.ndarray, y_t: float):
        observed_features = list(np.where(np.isnan(x_t) == 0)[0])
        pred = np.dot(x_t[observed_features], self.w[observed_features])
        if self.typ == ModelType.CLASSIFICATION:
            pred = logistic(pred)

        gradient = - (y_t - pred) * x_t[observed_features]
        if self.typ == ModelType.REGRESSION:
            gradient *= 2

        gradient += self.beta1 * np.sign(self.w[observed_features])
        gradient += self.beta2 * self.w[observed_features]
        final_step_size = self.step_size
        if self.adaptive_step_size:
            self.sqr_gradient_sums_diag[observed_features] = (
                self.sqr_gradient_sums_diag[observed_features] * 0.95
                + gradient * gradient
            )
            final_step_size = final_step_size * np.sqrt(
                1 / self.sqr_gradient_sums_diag[observed_features]
            )
        self.w[observed_features] -= gradient * final_step_size

        return pred

    def fit(self, x_t: np.ndarray, y_t: float):
        observed_features = list(np.where(np.isnan(x_t) == 0)[0])
        pred = np.dot(x_t[observed_features], self.w[observed_features])
        if self.typ == ModelType.CLASSIFICATION:
            pred = logistic(pred)

        gradient = - (y_t - pred) * x_t[observed_features]
        if self.typ == ModelType.REGRESSION:
            gradient *= 2
        gradient += self.beta1 * np.sign(self.w[observed_features])
        gradient += self.beta2 * self.w[observed_features]

        final_step_size = self.step_size
        if self.adaptive_step_size:
            self.sqr_gradient_sums_diag[observed_features] = (
                self.sqr_gradient_sums_diag[observed_features] * 0.95
                + gradient * gradient
            )
            final_step_size = final_step_size * np.sqrt(
                1 / self.sqr_gradient_sums_diag[observed_features]
            )
        self.w[observed_features] -= gradient * final_step_size

    def predict(self, x_t: np.ndarray):
        observed_features = list(np.where(np.isnan(x_t) == 0)[0])
        pred = np.dot(x_t[observed_features], self.w[observed_features])
        return pred

    def initialize(self, X: np.ndarray, y: np.ndarray):
        M = X.T @ X
        self.w = np.linalg.inv(M) @ X.T @ y

class OLMS:
    """Online Linear Model with Subregressions."""

    def __init__(
        self,
        size: int,
        step_size: float,
        sub_step_size: float,
        beta1: float = 0,
        beta2: float = 0,
        typ : ModelType = ModelType.REGRESSION
    ):
        self.size = size
        self.step_size = step_size
        self.sub_step_size = sub_step_size
        self.beta1 = beta1
        self.beta2 = beta2
        self.w = np.zeros(size)
        self.G = np.zeros((size, size)) * np.nan
        self.seen_features: Set[int] = set()
        self.reconstruction_errors: List[np.ndarray] = []
        self.typ = typ

    def reconstruction_gradient(self, x_t: np.ndarray):
        observed_features = np.where(np.isnan(x_t) == 0)[0]

        for feature in observed_features:
            if feature in self.seen_features:
                for seen_feature in self.seen_features:
                    if seen_feature in observed_features and np.isnan(
                        self.G[feature, seen_feature]
                    ):
                        self.G[feature, seen_feature] = 0
            elif feature not in self.seen_features:
                self.G[:, feature] = np.zeros((self.size,))
        self.seen_features = self.seen_features.union(observed_features)
        Gr = self.G[observed_features, :]

        psi_x = np.dot(x_t[observed_features], Gr)
        psi_x[np.isnan(psi_x)] = 0

        It = np.zeros((len(observed_features), x_t.shape[0]))
        for i, feature in enumerate(observed_features):
            It[i, feature] = 1

        x_diff = x_t[observed_features] - psi_x[observed_features]
        x_dot_residuals = np.dot(
            x_t[observed_features].reshape(-1, 1), x_diff.reshape(1, -1)
        )

        reconstruction_error = x_t - psi_x
        reconstruction_error[np.isnan(x_t)] = 0
        self.reconstruction_errors.append(reconstruction_error)

        gradient_G = -2 * np.dot(x_dot_residuals, It)
        gradient_G = np.clip(gradient_G, -1000, 1000)
        psi_x[observed_features] = x_t[observed_features]
        return psi_x, gradient_G

    def predictor_gradient(self, psi_x: np.ndarray, y_t: float):
        pred = np.dot(self.w, psi_x)
        if self.typ == ModelType.CLASSIFICATION:
            pred = logistic(pred)

        gradient_w = - (y_t - pred) * psi_x

        if self.typ == ModelType.REGRESSION:
            gradient_w *= 2

        gradient_w += self.beta1 * np.sign(self.w)
        gradient_w += self.beta2 * self.w
        gradient_w[np.isnan(gradient_w)] = 0
        return pred, gradient_w

    def predict(self, x_t: np.ndarray):
        psi_x, _ = self.reconstruction_gradient(x_t)
        pred, _ = self.predictor_gradient(psi_x, 0)
        return pred

    def update_reconstruction(self, x_t: np.ndarray):
        observed_features = np.where(np.isnan(x_t) == 0)[0]
        psi_x, gradient_G = self.reconstruction_gradient(x_t)
        self.G[observed_features, :] = (
            self.G[observed_features, :] - gradient_G * self.sub_step_size
        )
        for i in observed_features:
            self.G[i, i] = 0
        return psi_x

    def update_predictor(self, psi_x: np.ndarray, y_t: float):
        pred, gradient_w = self.predictor_gradient(psi_x, y_t)
        self.w -= self.step_size * gradient_w

        return pred

    def fit(self, x_t: np.ndarray, y_t: float):
        psi_x = self.update_reconstruction(x_t)
        pred = self.update_predictor(psi_x, y_t)
        return pred

    def fit_predict(self, x_t: np.ndarray, y_t: float):
        return self.fit(x_t, y_t)

    def initialize(self, X: np.ndarray, y: np.ndarray):
        M = X.T @ X
        self.w = np.linalg.inv(M) @ X.T @ y
        self.seen_features = set(list(range(self.size)))
        for i in range(self.size):
            lim_X = np.delete(X, i, axis=1)
            lim_y = X[:, i]
            M = lim_X.T @ lim_X
            lim_w = np.linalg.pinv(M) @ lim_X.T @ lim_y
            lim_w = np.concatenate((lim_w[0:i], [0], lim_w[i:]))
            self.G[i, :] = lim_w

class OLMSPartialReconstruction:
    """Online Linear Model with Subregressions. Only partially reconstructs features, i.e. only reconstructs the features that are passed as features to be reconstructed."""

    def __init__(
        self,
        size: int,
        step_size: float,
        sub_step_size: float,
        reconstructed_features : List[int],
        beta1: float = 0,
        beta2: float = 0,
        typ : ModelType = ModelType.REGRESSION,
    ):
        self.size = size
        self.step_size = step_size
        self.sub_step_size = sub_step_size
        self.beta1 = beta1
        self.beta2 = beta2
        self.w = np.zeros(size)
        self.G = np.zeros((size, size)) * np.nan
        self.seen_features: Set[int] = set()
        self.reconstruction_errors: List[np.ndarray] = []
        self.typ = typ
        self.non_reconstructed_features = list(set(range(size))-set(reconstructed_features))

    def reconstruction_gradient(self, x_t: np.ndarray):
        observed_features = np.where(np.isnan(x_t) == 0)[0]

        for feature in observed_features:
            if feature in self.seen_features:
                for seen_feature in self.seen_features:
                    if seen_feature in observed_features and np.isnan(
                        self.G[feature, seen_feature]
                    ):
                        self.G[feature, seen_feature] = 0
            elif feature not in self.seen_features:
                self.G[:, feature] = np.zeros((self.size,))
        self.seen_features = self.seen_features.union(observed_features)
        Gr = self.G[observed_features, :]

        psi_x = np.dot(x_t[observed_features], Gr)
        psi_x[np.isnan(psi_x)] = 0
        psi_x[self.non_reconstructed_features]=0

        It = np.zeros((len(observed_features), x_t.shape[0]))
        for i, feature in enumerate(observed_features):
            It[i, feature] = 1

        x_diff = x_t[observed_features] - psi_x[observed_features]
        x_dot_residuals = np.dot(
            x_t[observed_features].reshape(-1, 1), x_diff.reshape(1, -1)
        )

        reconstruction_error = x_t - psi_x
        reconstruction_error[np.isnan(x_t)] = 0
        self.reconstruction_errors.append(reconstruction_error)

        gradient_G = -2 * np.dot(x_dot_residuals, It)
        gradient_G = np.clip(gradient_G, -1000, 1000)
        psi_x[observed_features] = x_t[observed_features]
        return psi_x, gradient_G

    def predictor_gradient(self, psi_x: np.ndarray, y_t: float):
        pred = np.dot(self.w, psi_x)
        if self.typ == ModelType.CLASSIFICATION:
            pred = logistic(pred)

        gradient_w = - (y_t - pred) * psi_x

        if self.typ == ModelType.REGRESSION:
            gradient_w *= 2

        gradient_w += self.beta1 * np.sign(self.w)
        gradient_w += self.beta2 * self.w
        gradient_w[np.isnan(gradient_w)] = 0
        return pred, gradient_w

    def predict(self, x_t: np.ndarray):
        psi_x, _ = self.reconstruction_gradient(x_t)
        pred, _ = self.predictor_gradient(psi_x, 0)
        return pred

    def update_reconstruction(self, x_t: np.ndarray):
        observed_features = np.where(np.isnan(x_t) == 0)[0]
        psi_x, gradient_G = self.reconstruction_gradient(x_t)
        self.G[observed_features, :] = (
            self.G[observed_features, :] - gradient_G * self.sub_step_size
        )
        for i in observed_features:
            self.G[i, i] = 0
        return psi_x

    def update_predictor(self, psi_x: np.ndarray, y_t: float):
        pred, gradient_w = self.predictor_gradient(psi_x, y_t)
        self.w -= self.step_size * gradient_w

        return pred

    def fit(self, x_t: np.ndarray, y_t: float):
        psi_x = self.update_reconstruction(x_t)
        pred = self.update_predictor(psi_x, y_t)
        return pred

    def fit_predict(self, x_t: np.ndarray, y_t: float):
        return self.fit(x_t, y_t)

    def initialize(self, X: np.ndarray, y: np.ndarray):
        M = X.T @ X
        self.w = np.linalg.inv(M) @ X.T @ y
        self.seen_features = set(list(range(self.size)))
        for i in range(self.size):
            lim_X = np.delete(X, i, axis=1)
            lim_y = X[:, i]
            M = lim_X.T @ lim_X
            lim_w = np.linalg.pinv(M) @ lim_X.T @ lim_y
            lim_w = np.concatenate((lim_w[0:i], [0], lim_w[i:]))
            self.G[i, :] = lim_w
class OCDS:
    """Implementation of the OCDS predictor from the paper Online Learning from
    Capricious Data Streams.

    A few things have been changed: a) Classification variant uses binary
    cross-entropy loss, b) Step size is k/sqrt(t) instead of 1/sqrt(t), c) G
    matrix is initialized with zeros instead of noise since noise makes it
    extremely fragile, d) Ensembling parameter overflows extremely quickly; in
    case of overflow, use p=0.5 for ensembling parameter"""

    def __init__(
        self,
        size: int,
        base_step_size: float,
        alpha: float,
        beta1: float = 0,
        beta2: float = 0,
        typ : ModelType = ModelType.REGRESSION
    ):
        self.size = size
        self.base_step_size = base_step_size
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.w = np.zeros(size)
        self.G = np.zeros((size, size)) * np.nan
        self.seen_features: Set[int] = set()

        self.p = 0.5
        self.t = 1
        self.observed_loss = 0.0
        self.reconstructed_loss = 0.0
        self.seen_features = set()
        self.typ = typ

    def predict(self, x_t: np.ndarray):
        observed_features = np.where(np.isnan(x_t) == 0)[0]
        unobserved_features = np.where(np.isnan(x_t) == 1)[0]
        observed_feature_count = len(observed_features)
        for feature in observed_features:
            if feature in self.seen_features:  # Fill up already seen features
                for seen_feature in self.seen_features:
                    if seen_feature in observed_features and np.isnan(
                        self.G[feature, seen_feature]
                    ):
                        self.G[feature, seen_feature] = 0
            elif feature not in self.seen_features:
                self.G[:, feature] = np.zeros(
                    self.size
                )
            self.seen_features.add(feature)
        Gr = self.G[observed_features, :]

        psi_x = 1 / observed_feature_count * np.dot(x_t[observed_features], Gr)
        psi_x[np.isnan(psi_x)] = 0
        y_hat = self.p * np.dot(self.w[observed_features], x_t[observed_features]) + (
            1 - self.p
        ) * np.dot(self.w[unobserved_features], psi_x[unobserved_features])
        if self.typ == ModelType.CLASSIFICATION:
            y_hat = logistic(y_hat)
        return y_hat

    def fit(self, x_t: np.ndarray, y_t: float):
        step_size = np.max((self.base_step_size * (1 / np.sqrt(self.t+1)), 0.001))
        observed_features = np.where(np.isnan(x_t) == 0)[0]
        observed_feature_count = len(observed_features)
        for feature in observed_features:
            if feature in self.seen_features:  # Fill up already seen features
                for seen_feature in self.seen_features:
                    if seen_feature in observed_features and np.isnan(
                        self.G[feature, seen_feature]
                    ):
                        self.G[feature, seen_feature] = 0
            elif feature not in self.seen_features:
                self.G[:, feature] = np.zeros(
                    self.size
                )
            self.seen_features.add(feature)
        Gr = self.G[observed_features, :]

        psi_x = 1 / observed_feature_count * np.dot(x_t[observed_features], Gr)
        psi_x[np.isnan(psi_x)] = 0
        y_hat_obs = np.dot(self.w[observed_features], x_t[observed_features])
        y_hat_rec = np.dot(self.w, psi_x)
        y_hat = self.p * np.dot(self.w[observed_features], x_t[observed_features]) + (
            1 - self.p
        ) * np.dot(self.w, psi_x)
        if self.typ == ModelType.CLASSIFICATION:
            y_hat = logistic(y_hat)

        self.observed_loss += (y_t - y_hat_obs) ** 2
        delta_rec_loss = (y_t - y_hat_rec) ** 2

        self.reconstructed_loss += delta_rec_loss if not np.isnan(delta_rec_loss) else 0

        eta = 0.01 * np.sqrt(1 / np.log(self.t))

        self.p = np.exp(-eta * self.observed_loss) / (
            np.exp(-eta * self.observed_loss) + np.exp(-eta * self.reconstructed_loss)
        )
        if np.isnan(self.p):
            self.p = 0.5
        L = scipy.sparse.csgraph.laplacian(self.G, normed=False)

        x_t_obs_0 = x_t.copy()
        x_t_obs_0[np.isnan(x_t)] = 0

        gradient_w = (
            -(y_t - y_hat) * psi_x * (2 if self.typ == ModelType.REGRESSION else 1)
            + self.beta1 * np.sign(self.w)
            + self.beta2 * np.dot((L + L.T), self.w)
        )
        gradient_w[np.isnan(gradient_w)] = 0
        gradient_w = np.clip(gradient_w, -1000, 1000)

        full_observed_x = np.zeros(self.size)
        full_observed_x[observed_features] = x_t[observed_features]
        It = np.zeros((observed_feature_count, self.size))
        for i, feature in enumerate(observed_features):
            It[i, feature] = 1

        x_diff = x_t[observed_features] - psi_x[observed_features]
        x_dot_residuals = np.dot(
            x_t[observed_features].reshape(-1, 1), x_diff.reshape(1, -1)
        )

        gradient_G = (-2 / observed_feature_count) * (y_t - y_hat) * np.dot(
            x_t[observed_features].reshape(-1, 1), self.w.reshape(1, -1)
        ) - (2 * self.alpha / observed_feature_count) * np.dot(x_dot_residuals, It)

        gradient_G = np.clip(gradient_G, -1000, 1000)

        self.w = self.w - step_size * gradient_w
        self.G[observed_features, :] = (
            self.G[observed_features, :] - step_size * gradient_G
        )
        self.t += 1
        return y_hat

    def fit_predict(self, x_t: np.ndarray, y_t: float):
        return self.fit(x_t, y_t)
