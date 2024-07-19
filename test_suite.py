from predictors import OLM, OLMS, OCDS, ModelType
from utils import gen_synth_data, get_predictions
import matplotlib.pyplot as plt
import numpy as np

def test_olm_convergence():
    X,y, true_coefs = gen_synth_data(size=200, rho=0.0, seed=914, features=6, missing=False, return_coefs=True)
    olm = OLM(X.shape[1], 0.01)
    predictions, coefs = get_predictions(X, y, olm, offset=0, reconstructive=False, return_coefs=True)
    plt.plot(coefs)
    for i in range(6):
        plt.axhline(true_coefs[i], linestyle="--")
    plt.show()

def test_olms_convergence():
    X,y, true_coefs = gen_synth_data(size=200, rho=0.0, seed=913, features=6, missing=False, return_coefs=True)
    olms = OLMS(X.shape[1], 0.01,0.01)
    predictions, coefs = get_predictions(X, y, olms, offset=0, reconstructive=False, return_coefs=True)
    plt.plot(coefs)
    for i in range(6):
        plt.axhline(true_coefs[i], linestyle="--")
    plt.show()

def test_olms_on_missing():
    X,y, true_coefs = gen_synth_data(size=400, rho=0.99, seed=912, features=6, missing=True, return_coefs=True)
    olms = OLMS(X.shape[1], 0.01,0.01)
    predictions, coefs = get_predictions(X, y, olms, offset=0, reconstructive=False, return_coefs=True)
    plt.plot(coefs)
    for i in range(6):
        plt.axhline(true_coefs[i], linestyle="--")
    plt.show()

    plt.plot(np.cumsum((predictions-y)**2)/np.arange(400)+1)
    plt.show()

test_olms_on_missing()
