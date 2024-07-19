import numpy as np
from utils import load_sc_data, get_predictions
from predictors import OLM, OLMS, ModelType
import matplotlib.pyplot as plt
import seaborn as sns

losses = {}
seeds = 100*9
for seed in range(seeds):
    print(f"Seed {seed}                  ", end="\r")
    X,y = load_sc_data(seed=seed, size=10*24, missing=True)
    olm = OLM(X.shape[1], 0.01)
    olms = OLMS(X.shape[1], 0.01, 0.01)

    predictors = {
        "OLM": olm,
        "OLM-S": olms
    }

    for name, predictor in predictors.items():
        reconstructive = False
        if name == "OLM-S":
            reconstructive = True
        predictions = get_predictions(X, y, predictor, offset=0, reconstructive=reconstructive)
        if name not in losses:
            losses[name] = []
        losses[name].append(np.mean((predictions[60:]-y[60:])**2))
    #     plt.plot(np.cumsum((predictions-y)**2)/np.arange(240)+1, label=name)
    # plt.legend()
    # plt.show()

results = {}
results["OLM"] = np.mean(losses["OLM"])
results["OLM_conf"] = np.std(losses["OLM"])*1.96/np.sqrt(seeds)
results["OLM-S"] = np.mean(losses["OLM-S"])
results["OLM-S_conf"] = np.std(losses["OLM-S"])*1.96/np.sqrt(seeds)

plt.bar([0], [results["OLM"]], yerr=[results["OLM_conf"]], label="OLM")
plt.bar([1], [ results["OLM-S"]], yerr=[ results["OLM-S_conf"]], label="OLM-S")
plt.xticks([0,1], ["OLM", "OLM-S"])
for i, name in enumerate(["OLM", "OLM-S"]):
    plt.text(i, results[name]/2, f"{results[name]:.3f} ± {results[name+'_conf']:.3f}", ha="center", va="bottom")
plt.ylabel("MSE")
plt.title("South Carolina wind dataset regression performance")
plt.show()



