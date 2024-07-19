import numpy as np
from utils import get_predictions, gen_synth_data, features_to_index, index_to_features, load_sc_data
from market_tests import get_shapleys
from predictors import OLMSPartialReconstruction
from shapley import shapley
import matplotlib.pyplot as plt
from typing import Dict, List
import json
import time


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
    def __init__(self, features):
        if type(features) == int:
            features = index_to_features(features)
        self.features = features

    def __call__(self, k):
        reconstructed_features = index_rerepresent(k, self.features)
        return OLMSPartialReconstruction(len(k), 0.01, 0.01, reconstructed_features)


def get_feature_payments(X,y,method):
    feature_shapleys = get_shapleys(X, y, method, 25 if dataset==0 else 0, [X.shape[1]-1])
    # print("Feature shapleys:",feature_shapleys )
    feature_payments = []
    for i in range(X.shape[1]-1):
        feature_payments += [feature_shapleys[i]]
        # print(f"Feature {i+1}: {feature_payments[i]}")
    return feature_payments

iterations = 90
dataset = 1
features = None
rho = None
if dataset==0:
    features=3
    rho = 0.9
elif dataset==1:
    features=9

methods = {}
methods["grand_coalition"] = MethodMaker(2**features-1)
for feature in range(0,features):
    reconstructed_features = list(range(features))
    reconstructed_features.remove(feature)
    methods[f"mean_impute_{feature}"] = MethodMaker(reconstructed_features)

total_work = iterations*len(methods)
done_work = 0
start_time = time.time()

def seconds_to_timestamp(seconds):
    minutes = seconds//60
    seconds = seconds%60
    return f"{int(minutes):02d}:{int(seconds):02d}"

# cross_method_feature_shapleys = {}
# for i in range(iterations):
#     print(f" --- Iteration {i+1} of {iterations} --- ")
#     X,y = None, None
#     if dataset==0:
#         X,y = gen_synth_data(200, rho, i+31233, features)
#     elif dataset==1:
#         X,y = load_sc_data(i)
#     for method in methods:
#         print("Running method", method)
#         if method not in cross_method_feature_shapleys:
#             cross_method_feature_shapleys[method] = []
#         payment = get_feature_payments(X,y,methods[method])
#         cross_method_feature_shapleys[method] += [np.array(payment)]
#         done_work += 1
#         print(f"{done_work/total_work*100:.2f}% ({seconds_to_timestamp(time.time()-start_time)})")
#
#     projected_time = (time.time()-start_time)*(total_work-done_work)/done_work
#     print(f"Projected time: {seconds_to_timestamp(projected_time)} ({seconds_to_timestamp(time.time()-start_time)} so far)")
#
#
#
# for method in cross_method_feature_shapleys:
#     cross_method_feature_shapleys[method] = np.array(cross_method_feature_shapleys[method])
#     np.save(f"cross_method_feature_shapleys_{method}_dataset_{dataset}.npy", cross_method_feature_shapleys[method])

cross_method_feature_shapleys = {}
for method in methods:
    cross_method_feature_shapleys[method] = np.load(f"cross_method_feature_shapleys_{method}_dataset_{dataset}.npy")
reconstructed_payments = np.array(cross_method_feature_shapleys["grand_coalition"])
payment_difference = np.zeros((iterations, features))

for feature in range(features):
    print(reconstructed_payments.shape)
    feature_rec = reconstructed_payments[:, feature, :]
    feature_imp = cross_method_feature_shapleys[f"mean_impute_{feature}"][:, feature, :]
    payment_difference[:, feature] = np.nansum(feature_imp - feature_rec, axis=1)
    print(f"Feature {feature+1}:", feature_rec, feature_imp)


payment_difference_mean = np.mean(payment_difference, axis=0)
payment_difference_conf = np.std(payment_difference, axis=0)*1.96/np.sqrt(iterations)

plt.bar(range(features), payment_difference_mean, yerr=payment_difference_conf)
title = "Difference in revenue when mean imputing (MI) for individual features\n"
if dataset==0:
    title += f"rho={rho}"
elif dataset==1:
    title += "South Carolina dataset"
plt.title(title)

plt.ylabel("<- MI pays less | MI pays more ->")
plt.xticks(np.arange(features), [f"Feature {i+1}" for i in range(features)], rotation=30)
plt.tight_layout()
plt.savefig(f"mean_impute_difference_south_carolina.pdf")
plt.show()
