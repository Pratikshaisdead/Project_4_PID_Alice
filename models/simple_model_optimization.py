#%% Importing Libraries
import numpy as np
import pandas as pd 

import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import sklearn
from sklearn.utils import shuffle 
from sklearn import preprocessing as pp
from sklearn.metrics import confusion_matrix
import joblib

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier

import uproot
import time
from datetime import datetime

import optuna 
import optuna.visualization.matplotlib as vis 
#%% Loading Data
folder = r'D:/OneDrive/Documents/Universiteit Utrecht/Masters/Computational Aspects of Machine Learning/PR4/'

full_train_df = pd.read_csv(folder+r"models/data/full_train_df.csv")
full_train_trg = pd.read_csv(folder+r"models/data/full_train_trg.csv")
full_test_df = pd.read_csv(folder+r"models/data/full_test_df.csv")
full_test_trg = pd.read_csv(folder+r"models/data/full_test_trg.csv")

#%% Optimize models
dtc = DecisionTreeClassifier()
params = dtc.get_params()
print("Hyper-parameters for dtc: ")

def objective(trial, X_train = full_train_df, y_train = full_train_trg, X_test = full_test_df, y_test = full_test_trg): 
    # ccp_alpha = trial.suggest_float("ccp_alpha", 1e-8, 1e-2)
    max_depth = trial.suggest_int("max_depth", 2, 50)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)

    model = DecisionTreeClassifier(ccp_alpha=3.052311750001142e-06, 
                                   max_depth=max_depth,
                                   min_samples_leaf=min_samples_leaf,
                                   random_state=0,
                                  )
    
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    size = model.tree_.node_count    
    return acc

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=500)
joblib.dump(study, folder+r"models/optimization/dtc_study_optuna_acc2")
# study = joblib.load(folder+r"models/optimization/dtc_study_optuna_AccSize")

#%%
study_df = study.trials_dataframe()

vis.plot_optimization_history(
    study,
    target=lambda t: t.values[0],
    target_name="Accuracy"
)

#%%
# pareto = study.best_trials
# for t in pareto:
#     print("values:", t.values, "params:", t.params)

best_acc = max(study.trials, key=lambda t: t.values[0])
print(best_acc.values, best_acc.params)

clf = DecisionTreeClassifier() 
clf.fit(full_train_df, full_train_trg)
print("Default DTC accuracy:", clf.score(full_test_df, full_test_trg))

# plt.figure()
# plt.ion()
# plt.plot(study_df["params_ccp_alpha"], study_df["value"], "o")
# plt.xlabel("ccp_alpha")
# plt.ylabel("Accuracy")
# plt.title("DTC Hyperparameter Optimization")


#%%
plt.plot([1,2,3,4,5], [1,2,3,4,5], "o")

# %%
# ccp_alpha_lst = np.linspace(0, 10, 50)
# test_scores = []
# i=0

# for ccp_alpha in ccp_alpha_lst:
#     i = i + 1
#     model = DecisionTreeClassifier(ccp_alpha=ccp_alpha)
#     model.fit(full_train_df, full_train_trg)
#     test_score = model.score(full_test_df, full_test_trg)
#     test_scores.append(test_score)
#     print(i)

# plt.plot(ccp_alpha_lst, test_scores)
# plt.xlabel("ccp_alpha")
# plt.ylabel("Test Score")
# plt.title("DTC")