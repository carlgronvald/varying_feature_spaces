import numpy as np
from utils import index_to_features, features_to_index, get_predictions, gen_synth_data, load_sc_data
from shapley import calculate_losses, shapley
from predictors import OLMS, OLM
import matplotlib.pyplot as plt
import scipy

def get_shapleys(X, y, method, offset, central_features, return_regressions=False):
    def run_regressions(X, y, central_features, predictor_lambda):
        regressions = {}
        for i in range(1, 2**X.shape[1]):
            features = index_to_features(i)
            if not set(central_features).issubset(set(features)):
                continue
            regressions[i] = get_predictions(X[:,features], y, predictor_lambda(features), offset)
        return regressions
    
    regressions = run_regressions(X, y, central_features, method)
    losses = calculate_losses(regressions, y, central_features, X.shape[1])
    feature_shapleys = {}
    for i in range(X.shape[1]):
        feature_shapleys[i] = shapley(losses, i, central_features, X.shape[1])

    if return_regressions:
        return feature_shapleys, regressions
    return feature_shapleys

def run_market(X,y,method,offset,central_features, title):
    shapleys, regressions = get_shapleys(X, y, method, offset, central_features, return_regressions=True)
    central_loss = ((regressions[features_to_index(central_features)] - y)**2)
    grand_loss = ((regressions[2**X.shape[1]-1] - y)**2)
    print(f"Central loss: {central_loss}, Grand loss: {grand_loss}, Improvement: {central_loss-grand_loss}")
    for i in shapleys.keys():
        if i in central_features:
            continue
        plt.plot(np.cumsum(np.nan_to_num(shapleys[i])), label=f"Feature {i+1}")
    plt.legend()
    plt.title(title + f"\nTotal market loss improvement: {np.nansum(central_loss) - np.nansum(grand_loss):.2f}\nTotal payments: {np.sum([np.sum(np.nan_to_num(shapleys[i])) for i in shapleys.keys()]):.2f}")
    plt.tight_layout()
    plt.axhline(0, color="black", linestyle="-", linewidth=0.5)
    plt.axvline(50, color="black", linestyle="--")
    plt.show()

    grand_regression, coefficients = get_predictions(X, y, method(np.arange(X.shape[1])), offset, return_coefs=True)
    labels = [f"Feature {i+1}" for i in range(X.shape[1]-1)]
    labels += ["Constant"]
    plt.plot(coefficients, label=labels)
    plt.legend()
    plt.title("OLM-S regression coefficients, grand coalition regression (rho = 0.9)")
    plt.axhline(0, color="black", linestyle="-", linewidth=0.5)
    plt.axvline(50, color="black", linestyle="--")
    plt.show()

def payout_difference(data_generator, method1,method2, offset, central_features, iterations, title, method_names, warmup=0):

    sum_payments_1 = {}
    sum_payments_2 = {}

    for iteration in range(iterations):
        X,y = data_generator(iteration)
        print("Iteration", iteration+1, "of", iterations, "         ", end="\r")
        shapleys1 = get_shapleys(X, y, method1, offset, central_features)
        shapleys2 = get_shapleys(X, y, method2, offset, central_features)
        for i in shapleys1.keys():
            if i in central_features:
                continue
            if i not in sum_payments_1:
                sum_payments_1[i] = []
                sum_payments_2[i] = []
            sum_payments_1[i] += [np.sum(np.nan_to_num(shapleys1[i][warmup:]))]
            sum_payments_2[i] += [np.sum(np.nan_to_num(shapleys2[i][warmup:]))]

    print(sum_payments_1)
    for feature in sum_payments_1.keys():
        feature_payments_1 = np.array(sum_payments_1[feature])
        feature_payments_2 = np.array(sum_payments_2[feature])
        test =scipy.stats.ttest_rel(feature_payments_1, feature_payments_2)
        print(f"Feature {feature+1}: {test}" )

    feature_payments_1 = [np.mean(sum_payments_1[i]) for i in range(X.shape[1]-1)]
    feature_payments_2 = [np.mean(sum_payments_2[i]) for i in range(X.shape[1]-1)]

    sum_diff = np.mean(np.array(list(sum_payments_1.values())) - np.array(list(sum_payments_2.values())))

    plt.bar(np.arange(X.shape[1]-1)*3, feature_payments_1, label=method_names[0])
    plt.bar(np.arange(X.shape[1]-1)*3+1, feature_payments_2, label=method_names[1])
    plt.legend()
    plt.title(title + f"\nAverage loss reduction: {-sum_diff:.2f}")
    plt.xticks(np.arange(X.shape[1]-1)*3+0.5, [f"Feature {i+1}" for i in range(X.shape[1]-1)], rotation=30)
    plt.show()

def show_missing_payments():
    missing_payments_olm = []
    present_payments_olm = []
    missing_payments_olms = []
    present_payments_olms = []


    iterations = 20
    rho = 0.95

    for i in range(iterations):
        print("Iteration", i+1, "of", iterations, "         ", end="\r")
        X,y = gen_synth_data(200, rho, i+34000, 3)
        shapleys_olm = get_shapleys(X, y, lambda k: OLM(len(k), 0.01), 25, [3])
        shapleys_olms = get_shapleys(X, y, lambda k: OLMS(len(k), 0.01, 0.01), 25, [3])
        shapleys_olm = np.array([shapleys_olm[i] for i in range(X.shape[1]-1)])
        shapleys_olms = np.array([shapleys_olms[i] for i in range(X.shape[1]-1)])
        # print(shapleys_olm.T)
        # print(X)
        missing_payments_olm += [np.nan_to_num(shapleys_olm[np.isnan(X[:,:-1].T)])]
        present_payments_olm += [np.nan_to_num(shapleys_olm[~np.isnan(X[:,:-1].T)])]
        missing_payments_olms += [np.nan_to_num(shapleys_olms[np.isnan(X[:,:-1].T)])]
        present_payments_olms += [np.nan_to_num(shapleys_olms[~np.isnan(X[:,:-1].T)])]

    print(missing_payments_olms)
    print(np.mean(missing_payments_olm))
    missing_payments_olm = np.array(missing_payments_olm)
    missing_payments_olms = np.array(missing_payments_olms)
    print(np.nanmean(missing_payments_olms<0))
    plt.bar([0], np.nanmean(missing_payments_olm<0), label="OLM")
    plt.bar([1], np.nanmean(missing_payments_olms<0), label="OLM-S")
    plt.legend()
    plt.title(f"Fraction of payments to missing features that are negative, rho={rho}")
    plt.ylabel("Fraction")
    plt.xticks([0,1], ["OLM", "OLM-S"])
    plt.tight_layout()
    plt.show()


    #print(len(missing_payments_olm))
    print(np.sum([len(missing_payments_olm[i]) for i in range(len(missing_payments_olm))]))
    missing_count_olm = np.sum([len(missing_payments_olm[i]) for i in range(len(missing_payments_olm))])
    missing_count_olms = np.sum([len(missing_payments_olms[i]) for i in range(len(missing_payments_olms))])
    plt.bar([0], np.nanmean(missing_payments_olm), label="OLM",yerr=np.nanstd(missing_payments_olm) * 1.96 / np.sqrt(missing_count_olm))
    plt.bar([1], np.nanmean(missing_payments_olms), label="OLM-S",yerr=np.nanstd(missing_payments_olms) * 1.96 / np.sqrt(missing_count_olms))

    plt.legend()
    plt.title(f"Average single time step payment for missing features, rho={rho}")
    plt.ylabel("")
    plt.xticks([0,1], ["OLM", "OLM-S"])
    plt.axhline(0, color="black", linestyle="-", linewidth=0.5)
    plt.tight_layout()
    plt.show()

def show_negative_payments():
    payments_olm = []
    payments_olms = []

    iterations = 100
    rho = 0.95

    for i in range(iterations):
        print("Iteration", i+1, "of", iterations, "         ", end="\r")
        X,y = gen_synth_data(200, rho, i+34000, 3)
        shapleys_olm = get_shapleys(X, y, lambda k: OLM(len(k), 0.01), 25, [3])
        shapleys_olms = get_shapleys(X, y, lambda k: OLMS(len(k), 0.01, 0.01), 25, [3])
        shapleys_olm = np.array([shapleys_olm[i] for i in range(X.shape[1]-1)])
        shapleys_olms = np.array([shapleys_olms[i] for i in range(X.shape[1]-1)])
        # print(shapleys_olm.T)
        # print(X)
        payments_olm += [shapleys_olm[~np.isnan(shapleys_olm)]]
        payments_olms += [shapleys_olms[~np.isnan(shapleys_olm)]]

    payments_olm = np.array(payments_olm)
    payments_olms = np.array(payments_olms)
    print(np.nanmean(payments_olms<0))
    plt.bar([0], np.nanmean(payments_olm<0), label="OLM")
    plt.bar([1], np.nanmean(payments_olms<0), label="OLM-S")
    plt.legend()
    plt.title(f"Fraction of feature payments that are negative, rho={rho}")
    plt.ylabel("Fraction")
    plt.xticks([0,1], ["OLM", "OLM-S"])
    plt.tight_layout()
    plt.show()

    print(np.sum([len(payments_olm[i]) for i in range(len(payments_olm))]))
    iteration_sums_olm = [np.sum(payments_olm[i, payments_olm[i,:]<0]) for i in range(payments_olm.shape[0])]
    iteration_sums_olms = [np.sum(payments_olms[i, payments_olms[i,:]<0]) for i in range(payments_olms.shape[0])]
    plt.bar([0], np.nanmean(iteration_sums_olm), label="OLM",yerr=np.nanstd(iteration_sums_olm) * 1.96 / np.sqrt(iterations))
    plt.bar([1], np.nanmean(iteration_sums_olms), label="OLM-S",yerr=np.nanstd(iteration_sums_olm) * 1.96 / np.sqrt(iterations))

    plt.legend()
    plt.title(f"Average total negative payments, rho={rho}")
    plt.ylabel("")
    plt.xticks([0,1], ["OLM", "OLM-S"])
    plt.axhline(0, color="black", linestyle="-", linewidth=0.5)
    plt.tight_layout()
    plt.show()

def show_present_payments():
    missing_payments_olm = []
    present_payments_olm = []
    missing_payments_olms = []
    present_payments_olms = []


    iterations = 20
    rho = 0.95

    for i in range(iterations):
        print("Iteration", i+1, "of", iterations, "         ", end="\r")
        X,y = gen_synth_data(200, rho, i+34000, 3)
        shapleys_olm = get_shapleys(X, y, lambda k: OLM(len(k), 0.01), 25, [3])
        shapleys_olms = get_shapleys(X, y, lambda k: OLMS(len(k), 0.01, 0.01), 25, [3])
        shapleys_olm = np.array([shapleys_olm[i] for i in range(X.shape[1]-1)])
        shapleys_olms = np.array([shapleys_olms[i] for i in range(X.shape[1]-1)])
        # print(shapleys_olm.T)
        # print(X)
        present_payments_olm += [np.nan_to_num(shapleys_olm[~np.isnan(X[:,:-1].T)])]
        present_payments_olms += [np.nan_to_num(shapleys_olms[~np.isnan(X[:,:-1].T)])]

    print("average present OLM payment:", np.nanmean(present_payments_olm))
    print("average present OLMS payment:", np.nanmean(present_payments_olms))

if __name__ == "__main__":
    # X,y = gen_synth_data(200, 0.9, 1, 6)
    # run_market(X, y, lambda k: OLMS(len(k), 0.01, 0.01), 25, [6], title="Cumulative feature payments, OLM-S regression (rho = 0.9)")

    # datagen = lambda k : gen_synth_data(200, 0.9, 1013+k, 3)
    # payout_difference(datagen, lambda k: OLM(len(k), 0.01), lambda k: OLMS(len(k), 0.01, 0.01), 25, [3], 100, "Market payout using OLM and OLM-S (rho=0.9)", ["OLM", "OLM-S"])

    datagen = lambda k : load_sc_data(k, 240)
    payout_difference(datagen, lambda k: OLM(len(k), 0.01), lambda k: OLMS(len(k), 0.01, 0.01), 0, [9], 90, "Market payout using OLM and OLM-S for the South Carolina dataset", ["OLM", "OLM-S"], warmup=0) #WARMUP DOESN'T MATTER SINCE OLM=OLM-S FOR ALL FEATURES PRESENT

    # show_missing_payments()
    show_present_payments()
    # show_negative_payments()
