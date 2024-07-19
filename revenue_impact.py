import numpy as np
from utils import get_predictions, gen_synth_data, features_to_index, index_to_features
from market_tests import get_shapleys
from predictors import OLMSPartialReconstruction
from shapley import shapley
import matplotlib.pyplot as plt
from typing import Dict, List
import json

def index_or_none(lst, v) -> int | None:
    try:
        return lst.index(v)
    except ValueError:
        return None

def index_rerepresent(lst, index):
    """Given a list of feature indices index and a list of present features
    lst, it finds the locations of all the present features in the list of
    feature indices."""
    final_list = []
    for i in index:
        if index_or_none(lst, i) is not None:
            final_list += [index_or_none(lst, i)]
    return final_list

class MethodMaker:
    def __init__(self, index):
        self.index = index

    def __call__(self, k):
        reconstructed_features = index_rerepresent(k, index_to_features(self.index))
        return OLMSPartialReconstruction(len(k), 0.01, 0.01, reconstructed_features)



def grouped_barplot(data : Dict[str, List[float]], group_names : List[str], errors : Dict[str, List[float]] = {}, title = ""):
    x = np.arange(len(data[list(data.keys())[0]]))
    width = 1/(len(data)+1)
    offset=0

    _,ax = plt.subplots(figsize=(10,5))
    for i, variable in enumerate(data):
        if i in errors:
            ax.bar(x + offset, data[variable], width=width, label=variable, yerr=errors[i], capsize=5)
        else:
            ax.bar(x + offset, data[variable], width=width, label=variable)
        offset += width

    new_group_names = []
    for i,group_name in enumerate(group_names):
        sum = np.sum([data[variable][i] for variable in data.keys()])
        new_group_names += [f"{group_name}\nsum impact {sum:.2f}"]

    ax.set_xticks(x + width * (len(data.keys())-1)/2, new_group_names)
    ax.set_ylabel("Revenue Impact")
    ax.legend(loc="upper right", ncols=2)
    ax.set_title(title)

def get_feature_payments(X,y,method):
    feature_shapleys = get_shapleys(X, y, method, 25, [X.shape[1]-1])
    # print("Feature shapleys:",feature_shapleys )
    feature_payments = []
    for i in range(X.shape[1]-1):
        feature_payments += [feature_shapleys[i]]
        # print(f"Feature {i+1}: {feature_payments[i]}")
    return feature_payments

features=3
methods = {}
for index in range(0,2**features):
    methods[f"coalition_{index}"] = MethodMaker(index)
iterations = 200
rho=0.1

cross_method_feature_shapleys = {}
for i in range(iterations):
    X,y = gen_synth_data(200, rho, i+31233, features)
    for method in methods:
        print("Running method", method)
        if method not in cross_method_feature_shapleys:
            cross_method_feature_shapleys[method] = []
        cross_method_feature_shapleys[method] += [np.array(get_feature_payments(X,y,methods[method]))]



# json.dump(cross_method_feature_shapleys, open("all_feature_shapleys.json", "w"))
for method in cross_method_feature_shapleys:
    cross_method_feature_shapleys[method] = np.array(cross_method_feature_shapleys[method])
sum_rec_gains = np.zeros((iterations,features,features))



for iteration in range(iterations):
    print("")
    print("Iteration", iteration)
    reconstruction_gains = {k : [] for k in range(features)}
    for feature in range(features):
        payoffs = {}
        # print(cross_method_feature_shapleys["coalition_0"] - cross_method_feature_shapleys["coalition_1"])
        for i in range(0, 2**features):
            # print(cross_method_feature_shapleys[f"coalition_{i}"].shape)
            # print(cross_method_feature_shapleys[f"coalition_{i}"][run,feature, :])
            payoffs[i] = np.nansum(cross_method_feature_shapleys[f"coalition_{i}"][iteration,feature, :])
        central_features = []
        shapleys = {}
        for reconstructed_feature in range(features):
            shapleys[reconstructed_feature] = shapley(payoffs, reconstructed_feature, central_features, features)
            reconstruction_gains[reconstructed_feature] += [-shapleys[reconstructed_feature]]
        print("Feature", feature)
        # print("Payoffs:", payoffs)
        print("Shapleys:", shapleys)
    sum_rec_gains[iteration] = np.array([reconstruction_gains[i] for i in range(features)])

confidence_intervals = {}
for i in range(features):
    arr = []
    for j in range(features):
        arr += [np.std(sum_rec_gains[:,i,j])/np.sqrt(iterations)]
    confidence_intervals[i] = arr

reconstruction_gains = np.mean(sum_rec_gains, axis=0)
reconstruction_gains = {f"Reconstructing feature {k}" : reconstruction_gains[k] for k in range(features)}

grouped_barplot(reconstruction_gains, [f"Revenue of feature {k}" for k in range(features)], title=f"Revenue Impact of Reconstructing Features, rho={rho}\nSum revenue increase: {np.sum(list(reconstruction_gains.values())):.2f}", errors=confidence_intervals)
plt.show()
