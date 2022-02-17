# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 17:22:44 2022

@author: shazn
"""

import pandas as pd
from math import comb
from itertools import product
from copy import deepcopy


# read in data
hdl_data = pd.read_csv("results_mnist_full.csv")

pd.unique(hdl_data.stable)
pd.unique(hdl_data.robust_test)
pd.unique(hdl_data.l0)


# choose the testing robustness 0.01

hdl_data = hdl_data[hdl_data.robust_test == 1e-2]


## calculate the features of the shap values - so we have 6 binary categories for SHAP, essentially 2^6 models that we are comparing
hdl_data["robust_bin"] = (hdl_data.robust >= 1e-2).astype(int)
hdl_data["sparsity_bin"] = (hdl_data.l0 > 0).astype(int)
hdl_data["l2_bin"] = (hdl_data.l2 > 0).astype(int)
hdl_data["lr_bin"] = (hdl_data.lr > 1e-4).astype(int)
hdl_data["batchsize_bin"] = (hdl_data.batch_size > 64).astype(int)
hdl_data["stable_bin"] = hdl_data.stable.astype(int)

# calculate some output values

hdl_data["log_stability"] = hdl_data.logit_stability.apply(lambda x: sum([float(y) for y in x[1:-1].split()]))
hdl_data["sparsity"] = (hdl_data.W1_non_zero + hdl_data.W2_non_zero + hdl_data.W3_non_zero) / 3

features = ["robust_bin","sparsity_bin","l2_bin","lr_bin","batchsize_bin","stable_bin"]

measures = ["log_stability", "gini_stability", "sparsity", "avg_test_acc", "adv_acc"]

total_columns = features + measures

hdl_data = hdl_data[total_columns]

# group the dataset so that each of the 2^6 models gets one line

hdl_data_grouped = hdl_data.groupby(features).median().reset_index()

type_value_dict = {}
feature_shap_dict = {}
shap_values_features_measures = []

# give a name to each of the 2^6 models (e.g. "1,0,0,0,0,1")
hdl_data_grouped["type"] = hdl_data_grouped[features].apply(
    lambda x: ','.join(x.dropna().astype(str)),
    axis=1
)

#prepare the combinations
combis = [','.join(x) for x in list(product(['1','0'], repeat=len(features)))]

for measure in measures:
    values = hdl_data_grouped[measure].values
    type_value_dict = dict(zip(hdl_data_grouped.type,values))
    for i, feature in enumerate(features):
        # we calcualte SHAP for each feature
        feature_shap = 0
        total_combs = [comb for comb in combis if comb[2*i] == "0"]
        # we sum over all models without that certain feature
        for combi in total_combs:
            # the main SHAP value calculation, see wikipedia for why this is calculated as such
            num_comb = sum([int(x) for x in combi.split(",")])
            combi_n = list(combi)
            combi_n[2*i] = "1"
            combi_n = ''.join(combi_n)
            feature_shap += 1 / (len(features) * comb(len(features)-1, num_comb)) * (type_value_dict[combi_n]-type_value_dict[combi])
        feature_shap_dict[feature] = feature_shap
    column_naming = 'Gain_' + measure
    shap_values_features = pd.DataFrame(feature_shap_dict.items(), columns=['Feature', column_naming])
    shap_values_features = shap_values_features.set_index('Feature')
    shap_values_features_measures.append(shap_values_features)

shap_values_features_measures_output = pd.concat(shap_values_features_measures, axis=1)
shap_values_features_measures_output.to_csv("shap_values_features_measures.csv")
