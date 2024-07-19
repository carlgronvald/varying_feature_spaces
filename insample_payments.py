import numpy as np

import matplotlib.pyplot as plt
from utils import index_to_features, features_to_index
from shapley import  shapley

def OLR(X, y):
    #I = np.eye(X.shape[1])
    coefs = np.linalg.pinv(X.T @ X) @ (X.T @ y)
    coefs = coefs.reshape(1,-1)
    loss = ((y - X @ coefs.T) ** 2)
    return coefs, loss

def OLM_insample(X,y):
    X_used = np.nan_to_num(X)
    return OLR(X_used, y)

def OLMS_insample(X,y):
    X_used = X.copy()
    for i in range(X.shape[1]):
        nans = np.isnan(X[:,i])
        coefs,_ = OLM_insample(np.delete(X_used[~nans,:], i, axis=1), X_used[~nans,i])
        X_used[nans,i] = coefs @ np.delete(np.nan_to_num(X_used[nans,:]), i, axis=1).T
    return OLR(X_used, y)

def insample_payment(observations=6, width=2, missing_from=1, rho=0.8, method = OLM_insample, return_means = True, mean_impute_f2=False, randomly_missing=False, missing_ratio=0.1):
    all_shapleys = {}
    iterations = 1000

    losses= {}
    for j in range(2**width):
        all_shapleys[j] = np.zeros((iterations, observations))
    for i in range(iterations):
        print(f"iteration {i+1} of {iterations}                              ", end="\r")
        sigma = np.zeros((width,width))
        for k in range(width):
            for j in range(width):
                dist = np.abs(k - j)
                if dist > width // 2:
                    dist = width - dist
                sigma[k, j] = rho ** dist
        true_x = np.random.multivariate_normal(np.zeros(width), sigma, size=observations)
        true_coefs = np.random.uniform(0.1, 1, width)
        true_y = true_x @ true_coefs

        x_missing = true_x.copy()
        if randomly_missing:
            for t in range(observations):
                for j in range(width):
                    if np.random.rand() < missing_ratio:
                        x_missing[t,j] = np.nan
        else:
            x_missing[missing_from:,1] = np.nan
            x_missing[1,0] = np.nan
            x_missing[0:106,2] = np.nan
            x_missing[4:5,0] = np.nan
            # x_missing[2,0] = np.nan
            # x_missing[3,0] = np.nan
            # x_missing[2,2] = np.nan
            # x_missing[3,2] = np.nan
            #x_missing[3,2] = np.nan

        if mean_impute_f2:
            x_missing[:,1] = np.nan_to_num(x_missing[:,1])


        for j in range(2**true_x.shape[1]):
            features = index_to_features(j)
            _,loss = method(x_missing[:,features], true_y.reshape(-1,1))
            losses[j] = loss

        for j in range(true_x.shape[1]):
            all_shapleys[j][i,:] = shapley(losses, j, [], true_x.shape[1])[:,0]

    print(np.array(all_shapleys[0]).shape)
    print(np.mean(all_shapleys[0],axis=0), np.mean(all_shapleys[1],axis=0))
    print(np.std(all_shapleys[0],axis=0), np.std(all_shapleys[1],axis=0))
    # feature_shapleys = {}
    # # print(sum_losses)
    # mean_losses = {k: np.mean(v,axis=0) for k,v in sum_losses.items()}
    # print(mean_losses)
    # std_losses = {k: np.std(v,axis=0) for k,v in sum_losses.items()}
    # print(std_losses)
    #
    # for i in range(2):
    #     feature_shapleys[i] = shapley(mean_losses, i, [], 2)
    if return_means:
        return [np.mean(all_shapleys[0],axis=0), np.mean(all_shapleys[1],axis=0)], [np.std(all_shapleys[0],axis=0)*1.96/np.sqrt(iterations), np.std(all_shapleys[1],axis=0)*1.96/np.sqrt(iterations)]
    else:
        shapley_list = []
        for i in range(width):
            shapley_list.append(np.array(all_shapleys[i]))
        return np.array(shapley_list)

def insample_payment_randomly_missing(observations=6, width=2, rho=0.8, method = OLM_insample, mean_impute_f2=False, missing_ratio=0.1):
    all_shapleys_present = {}
    all_shapleys_missing = {}
    iterations = 1000

    losses_2_present= {}
    losses_2_missing= {}
    for j in range(2**width):
        all_shapleys_present[j] = np.zeros((iterations, observations))
        all_shapleys_missing[j] = np.zeros((iterations, observations))
    for i in range(iterations):
        print(f"iteration {i+1} of {iterations}                              ", end="\r")
        sigma = np.zeros((width,width))
        for k in range(width):
            for j in range(width):
                dist = np.abs(k - j)
                if dist > width // 2:
                    dist = width - dist
                sigma[k, j] = rho ** dist
        true_x = np.random.multivariate_normal(np.zeros(width), sigma, size=observations)
        true_coefs = np.random.uniform(0.1, 1, width)
        true_y = true_x @ true_coefs

        x_missing = true_x.copy()
        for t in range(observations):
            for j in range(width):
                if np.random.rand() < missing_ratio:
                    x_missing[t,j] = np.nan
        x_missing[observations-1,1] = np.nan # Let last observation go missing to ensure at least 1 missing value

        if mean_impute_f2:
            x_missing[:,1] = np.nan_to_num(x_missing[:,1])


        f2_present = ~np.isnan(x_missing[:,1])
        for j in range(1,2**true_x.shape[1]):
            features = index_to_features(j)
            if len(features) == 0:
                _,losses_2_present[j] = method(np.zeros((np.sum(f2_present)*1,2)), true_y[f2_present].reshape(-1,1))
                _,losses_2_missing[j] = method(np.zeros((np.sum(~f2_present)*1,2)), true_y[~f2_present].reshape(-1,1))
            else:
                _,losses_2_present[j] = method(x_missing[f2_present,:][:,features], true_y[f2_present].reshape(-1,1))
                _,losses_2_missing[j] = method(x_missing[~f2_present,:][:,features], true_y[~f2_present].reshape(-1,1))

        iteration_present_shapleys = np.zeros((np.sum(f2_present)*1,true_x.shape[1]))
        iteration_missing_shapleys = np.zeros((np.sum(~f2_present)*1,true_x.shape[1]))
        for j in range(true_x.shape[1]):
            iteration_present_shapleys[:,j] = shapley(losses_2_present, j, [], true_x.shape[1])[:,0] #shapley returns of shape (observations, 1), so just removing extraneous dimension
            iteration_missing_shapleys[:,j] = shapley(losses_2_missing, j, [], true_x.shape[1])[:,0]
        iteration_present_shapleys_full = np.zeros((observations,true_x.shape[1]))*np.nan
        iteration_missing_shapleys_full = np.zeros((observations,true_x.shape[1]))*np.nan
        iteration_present_shapleys_full[f2_present,:] = iteration_present_shapleys
        iteration_missing_shapleys_full[~f2_present,:] = iteration_missing_shapleys
        for j in range(true_x.shape[1]):
            all_shapleys_present[j][i,:] = iteration_present_shapleys_full[:,j]
            all_shapleys_missing[j][i,:] = iteration_missing_shapleys_full[:,j]

    print(np.array(all_shapleys_present[0]).shape)
    print("present", np.nanmean(all_shapleys_present[0],axis=0), np.nanmean(all_shapleys_present[1],axis=0))
    print("missing", np.nanmean(all_shapleys_missing[0],axis=0), np.nanmean(all_shapleys_missing[1],axis=0))
    print(all_shapleys_missing)
    print("proportion of missing values", np.mean(~np.isnan(all_shapleys_missing[0])))
    # print(np.std(all_shapleys_present[0],axis=0), np.std(all_shapleys_present[1],axis=0))
    # feature_shapleys = {}
    # # print(sum_losses)
    # mean_losses = {k: np.mean(v,axis=0) for k,v in sum_losses.items()}
    # print(mean_losses)
    # std_losses = {k: np.std(v,axis=0) for k,v in sum_losses.items()}
    # print(std_losses)
    #
    # for i in range(2):
    #     feature_shapleys[i] = shapley(mean_losses, i, [], 2)
    # print(all_shapleys_present)
    return (
        [np.nanmean(all_shapleys_present[0],axis=0), np.nanmean(all_shapleys_present[1],axis=0)],
        [np.nanstd(all_shapleys_present[0],axis=0)*1.96/np.sqrt(iterations), np.nanstd(all_shapleys_present[1],axis=0)*1.96/np.sqrt(iterations)],
        [np.nanmean(all_shapleys_missing[0],axis=0), np.nanmean(all_shapleys_missing[1],axis=0)],
        [np.nanstd(all_shapleys_missing[0],axis=0)*1.96/np.sqrt(iterations), np.nanstd(all_shapleys_missing[1],axis=0)*1.96/np.sqrt(iterations)]
    )
def plot_payments():
    means, confs = insample_payment(observations=8, width=3, missing_from=7, method=OLMS_insample)
    print(means[0].shape)
    #plt.bar(np.arange(len(means[1][:,0]))*3, means[1][:,0])#, yerr=confs[0][:,0], capsize=3)
    plt.bar(np.arange(len(means[1][:,0]))*3+1, means[1][:,0], yerr=confs[1][:,0], capsize=3)
    plt.axhline(0, color="black", linestyle="-", linewidth=0.5)
    plt.title("Feature 2 in-sample payment, OLM model, rho=0.8\nFeature 2 missing on time step 8")
    plt.xticks(np.arange(len(means[1][:,0]))*3+1, [f"{i+1}" for i in range(means[1].shape[0])])
    plt.xlabel("Time Step")
    plt.ylabel("Payment")
    plt.show()

def plot_payments_randomly_missing():
    rho=0.999
    means_present, confs_present, means_missing, confs_missing = insample_payment_randomly_missing(observations=800, rho=0.99, width=8, method=OLM_insample, missing_ratio=0.99)
    print(means_present)
    print(means_missing)
    print("present mean payment:",np.nanmean(means_present))
    print("missing mean payment:",np.mean(means_missing))
    #plt.bar(np.arange(len(means[1][:,0]))*3, means[1][:,0])#, yerr=confs[0][:,0], capsize=3)
    plt.bar(np.arange(len(means_present[1][:]))*3, means_present[1][:], yerr=confs_present[1][:], capsize=3)
    plt.bar(np.arange(len(means_missing[1][:]))*3+1, means_missing[1][:], yerr=confs_missing[1][:], capsize=3)
    plt.axhline(0, color="black", linestyle="-", linewidth=0.5)
    plt.title(f"Feature 2 in-sample payment, OLM model, rho={rho}")
    # plt.xticks(np.arange(len(means[1][:,0]))*3+1, [f"{i+1}" for i in range(means[1].shape[0])])
    plt.xlabel("Time Step")
    plt.ylabel("Payment")
    plt.show()

def plot_payment_difference():
    missing_from = 3
    rho=0.95
    payments_rec = insample_payment(observations=180, width=3, rho=rho, missing_from=missing_from, method=OLMS_insample, return_means=False)
    payments_mi = insample_payment(observations=180, width=3, rho=rho, missing_from=missing_from, method=OLMS_insample, return_means=False, mean_impute_f2=True)
    diff = payments_rec - payments_mi
    revenue_diff = np.sum(diff[1,:,:], axis=1)
    
    plt.bar([0], np.mean(revenue_diff), yerr=np.std(revenue_diff)*1.96/np.sqrt(revenue_diff.shape[0]))
   
    missing_from=147
    payments_rec = insample_payment(observations=180, width=3, missing_from=missing_from, method=OLMS_insample, return_means=False)
    payments_mi = insample_payment(observations=180, width=3, missing_from=missing_from, method=OLMS_insample, return_means=False, mean_impute_f2=True)
    diff = payments_rec - payments_mi
    revenue_diff = np.sum(diff[1,:,:], axis=1)

    plt.bar([1], np.mean(revenue_diff), yerr=np.std(revenue_diff)*1.96/np.sqrt(revenue_diff.shape[0]))
    #plt.bar(np.arange(len(means[1][:,0]))*3, means[1][:,0])#, yerr=confs[0][:,0], capsize=3)
    # plt.bar(np.arange(missing_from)*3+1, mean_diff[:missing_from], capsize=3, yerr=mean_err[:missing_from])
    # plt.bar(np.arange(missing_from, len(mean_diff))*3+1, mean_diff[missing_from:], capsize=3, color="white", edgecolor="C0", hatch="////", yerr=mean_err[missing_from:])
    plt.axhline(0, color="black", linestyle="-", linewidth=0.5)
    plt.title(f"Feature 2 in-sample payment\nmean imputing vs reconstructing feature 2\nrho={rho}\nFeature 2 missing on time step 8")
    # plt.xticks(np.arange(len(means[1][:,0]))*3+1, [f"{i+1}" for i in range(means[1].shape[0])])
    plt.ylabel("<- MI pays less | MI pays more ->")

    plt.xlabel("Time Step")
    plt.tight_layout()
    plt.show()
plot_payments_randomly_missing()
# plot_payment_difference()
