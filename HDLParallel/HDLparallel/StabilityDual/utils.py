import numpy as np

def compVecImpurity(x):
    giniIndex = 0
    unique, counts = np.unique(x, return_counts=True)
    for i in range(len(unique)):
        pi = counts[i]/len(x)
        giniIndex += pi*(1-pi)
    return giniIndex




# We have a matrix, regPred that is n x k, where n is the number of observations, and k the number of experiments
# Each row of regPred stores the set of predictions made on observation i over the k experiments
# We go row by row, computing the impurity of the i^th observation, and then average
# We do the same thing for the set of predictions from the stable method, whose predictions are stored in stablePred

def total_gini(preds):
    n = len(preds)
    # check that preds is indeed observations * number of experiment
    assert preds.shape[0] > preds.shape[1]
    results = np.zeros(n)
    for i in range(n):
        results[i] += compVecImpurity(preds[i])/n
    return np.mean(results)
