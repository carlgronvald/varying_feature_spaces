import numpy as np
from predictors import OLM, OLMS, OCDS, ModelType
from utils import gen_synth_data, get_predictions, load_uci_data

def prepare_data(x, y, nan_fraction):
    permutation = np.random.permutation(range(x.shape[0]))
    x_prepared = x[permutation].copy()
    y_prepared = y[permutation].copy()

    for t in range(x.shape[0]):
        nans = np.random.choice(
            range(x_prepared.shape[1] - 1),
            size=int((x_prepared.shape[1] - 1) * nan_fraction),
            replace=False,
        )
        x_prepared[t, nans] = np.nan
    return x_prepared, y_prepared

def cartesian_product(*arrays):
    la = len(arrays)
    arrays = list(map(lambda x: np.array(x), arrays))
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)

def search_hyperparameters(x, y, method, parameters):
    best_accuracy = 0
    best_parameters = []

    all_parameters = cartesian_product(*list(parameters.values()))
    parameter_names = list(parameters.keys())

    use_X, use_y = prepare_data(x, y, 0.5)

    for k, parameter_set in enumerate(all_parameters):
        predictor = method(*parameter_set)
        predictions = np.zeros(use_X.shape[0])
        for t in range(use_X.shape[0]):
            predictions[t] = predictor.fit_predict(use_X[t, :], use_y[t])
        accuracy = np.mean((predictions > 0.5) == use_y)

        print(f"{k}:{parameter_set}                             ", end="\r")

        if accuracy > best_accuracy:
            print(
                "New best accuracy:",
                accuracy,
                "with parameters:",
                {k: v for k, v in zip(parameter_names, parameter_set)},
            )
            best_accuracy = accuracy
            best_parameters = {k: v for k, v in zip(parameter_names, parameter_set)}

    return best_accuracy, best_parameters

def accuracy_measurement(X, y, method, permutations=10):
    accs = []
    for i in range(permutations):
        round_X, round_y = prepare_data(X, y, 0.5)
        predictor = method()
        predictions = np.zeros(round_X.shape[0])
        for t in range(round_X.shape[0]):
            predictions[t] = predictor.fit_predict(round_X[t, :], round_y[t])
        acc = np.mean(((predictions > 0.5) == round_y))
        accs += [acc]
    return np.mean(accs), np.std(accs)

def dataset_performance(dataset, permutations=10):
    print("Examining performance on dataset ", dataset)
    X,y = load_uci_data(dataset) 
    olm = lambda step_size, beta1, beta2: OLM(X.shape[1], step_size, beta1, beta2, typ=ModelType.CLASSIFICATION)
    olms = lambda step_size_1, step_size_2, beta1, beta2: OLMS(X.shape[1], step_size_1, step_size_2, beta1, beta2, typ=ModelType.CLASSIFICATION)
    ocds = lambda step_size, alpha, beta1, beta2: OCDS(X.shape[1], step_size, alpha, beta1, beta2, typ=ModelType.CLASSIFICATION)

    print(" ---- SEARCHING OLM HYPERPARAMETERS ---- ")
    _, olm_hyperparams = search_hyperparameters(X, y, olm, {"step_size": [0.1, 0.01, 0.001], "beta1": [0, 0.001, 0.01], "beta2": [0, 0.001, 0.01]})
    print(" ---- SEARCHING OLMS HYPERPARAMETERS ---- ")
    _, olms_hyperparams = search_hyperparameters(X, y, olms, {"step_size_1": [0.1, 0.01, 0.001], "step_size_2": [0.1, 0.01, 0.001], "beta1": [0, 0.001, 0.01], "beta2": [0, 0.001, 0.01]})
    print(" ---- SEARCHING OCDS HYPERPARAMETERS ---- ")
    _, ocds_hyperparams = search_hyperparameters(X, y, ocds, {"step_size": [0.1, 0.01, 0.001], "alpha": [10, 1, 0.1, 0.01], "beta1": [0, 0.001, 0.01], "beta2": [0, 0.001, 0.01]})

    olm_acc, olm_std = accuracy_measurement(X, y, lambda: olm(**olm_hyperparams), permutations=permutations)
    olms_acc, olms_std = accuracy_measurement(X, y, lambda: olms( **olms_hyperparams), permutations=permutations)
    ocds_acc, ocds_std = accuracy_measurement(X, y, lambda: ocds( **ocds_hyperparams), permutations=permutations)

    print(f"OLM accuracy: {olm_acc} += {olm_std*1.96/np.sqrt(permutations)} over {permutations} permutations")
    print(f"OLMS accuracy: {olms_acc} += {olms_std*1.96/np.sqrt(permutations)} over {permutations} permutations")
    print(f"OCDS accuracy: {ocds_acc} += {ocds_std*1.96/np.sqrt(permutations)} over {permutations} permutations")

    return {
        "OLM": (olm_acc, olm_std),
        "OLMS": (olms_acc, olms_std),
        "OCDS": (ocds_acc, ocds_std),
        "permutations": permutations,
        "dataset": dataset,
    }

datasets = [
    "ionosphere",
    "german",
    "australian",
    "wdbc",
    "wbc",
    "magic04",
    "kr-vs-kp",
    "credit-a",
]

if __name__ == "__main__":
    all_data = {}
    for dataset in datasets:
        all_data[dataset] = dataset_performance(dataset)
