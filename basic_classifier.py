#%% Importing libraries

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
import optuna
import time
from datetime import datetime

#%% Open ROOT files

folder = r'D:/OneDrive/Documents/Universiteit Utrecht/Masters/Computational Aspects of Machine Learning/PR4/'
kaons   = uproot.open(folder+r"Kaons.root")
pions   = uproot.open(folder+r"Pions.root")
protons = uproot.open(folder+r"Protons.root")
electrons = uproot.open(folder+r"Electrons.root")
deuterons = uproot.open(folder+r"Deuterons.root")

#%% Creating DataFrames from ROOT files
def extract_data(particle):
    myTree = particle["data;1"]
    particleBranch = myTree.arrays()
    
    dEdxITS = np.asarray(particleBranch['dEdxITS'])
    dEdxTPC = particleBranch['dEdxTPC']
    p = particleBranch['p']
    pT = particleBranch['pT']
    eta = particleBranch['eta']
    phi = particleBranch['phi']
    
    particle_df = pd.DataFrame({
    'dEdxITS': dEdxITS,
    'dEdxTPC': dEdxTPC,
    'p': p,
    'pT': pT,
    'eta': eta,
    'phi': phi
    })
    
    return particle_df

def build_dataframe(particles_dic):
    frames = []
    for name, df in particles_dic.items():
        temp = df.copy()
        temp["particle"] = name     # add label column
        frames.append(temp)
    return pd.concat(frames, ignore_index=True)

def cutoff(df, variable, cutoff):
    low_df = df[df[variable]<cutoff]
    high_df = df[df[variable]>=cutoff]
    return low_df, high_df
    
particles = {"kaons": extract_data(kaons),
             "pions": extract_data(pions),
             "protons": extract_data(protons),
             "electrons": extract_data(electrons),
             "deuterons": extract_data(deuterons)}

particle_df = build_dataframe(particles)

# low_momentum_df, high_momentum_df = cutoff(particle_df, "pT", 1.0)

#%% Plot data from ROOT files

def plot_from_df(variable_x, variable_y, df,save_appendix=""):
    start = time.time()
    x_arr = df[variable_x]
    y_arr = df[variable_y]
    labels = df["particle"]

    plt.ion()
    fig,ax = plt.subplots(1,2,figsize=(18,6))
    
    for particle, subdf in df.groupby("particle"):
        ax[0].scatter(subdf[variable_x], subdf[variable_y], s=0.3, alpha = 0.5, label=particle)
        
    ax[0].legend(markerscale=10, loc = "upper right")
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_ylabel(variable_y)
    ax[0].set_xlabel(variable_x)
    ax[0].grid(alpha=0.4)
    ax[0].grid(which="minor", alpha=0.4)
    ax[0].set_axisbelow(True)
    ax[0].minorticks_on()
    ax[0].set_ylim(10, 2000)
    ax[0].set_xlim(0.1,70)
    
    # pT_bins = np.logspace(np.min(pT_arr), np.max(pT_arr), 100)
    # dEdxTPC_bins = np.logspace(np.log(np.min(dEdxTPC_arr)+0.1), np.log(np.max(dEdxTPC_arr)), 100)
    
    x_bins = np.logspace(np.log10(np.min(x_arr)), np.log10(np.max(x_arr)), 150)
    y_bins = np.logspace(np.log10(np.min(y_arr+10)),np.log10(np.max(y_arr)),150)
    
    hists = ax[1].hist2d(x_arr, y_arr, bins=[x_bins, y_bins], density=True, norm=LogNorm(), cmap="jet")
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    ax[1].set_ylim(10, np.max(y_arr))
    ax[1].grid(alpha=0.4)
    ax[1].grid(which="minor", alpha=0.4)
    ax[1].set_axisbelow(True)
    ax[1].minorticks_on()
    ax[1].set_ylim(10, 2000)
    ax[1].set_ylabel(variable_y)
    ax[1].set_xlabel(variable_x)
    ax[1].set_xlim(0.1,70)
    # fig.colorbar(hists[3], ax=ax[1]) 
    
    fig.suptitle(variable_x+" VS "+variable_y)
    file_name = folder+variable_x+"_"+variable_y+save_appendix+".png"
    plt.savefig(file_name,dpi=500)

    print("Amount of data points plotted: {:.2e}".format(len(y_arr)))
    end = time.time()
    print(end-start)

# plot_from_df("pT", "dEdxTPC", particle_df)
# plot_from_df("pT", "dEdxTPC", low_momentum_df,"_LowMomentum")
# plot_from_df("pT", "dEdxTPC", high_momentum_df,"_HighMomentum")

#%% Train models

def particle2int(x, inverse=False):
    if inverse: 
        x[x == 0] = "kaons"
        x[x == 1] = "pions"
        x[x == 2] = "protons"
        x[x == 3] = "electrons"
        x[x == 4] = "deuterons"
    elif not inverse: 
        x[x == "kaons"] = 0 
        x[x == "pions"] = 1
        x[x == "protons"] = 2
        x[x == "electrons"] = 3
        x[x == "deuterons"] = 4
    return x

def training_data(df, training_var, train_per):
    #Splitting the DataFrame into a low momentum and a high momentum DataFrame
    low_df, high_df = cutoff(df, "pT", 0.5)
    lim_low = int(len(low_df)*train_per)
    lim_high = int(len(high_df)*train_per)
    
    #Selecting the data used for training and for testing
    low_df = shuffle(low_df.copy())
    low_train = low_df[:lim_low]
    low_test = low_df[lim_low:]

    high_df = shuffle(high_df.copy())
    high_train = high_df[:lim_high]
    high_test = high_df[lim_high:]

    full_train = shuffle(pd.concat([low_train, high_train], axis=0))
    full_test = shuffle(pd.concat([low_test, high_test], axis=0))
    
    low_train_df = low_train[training_var]
    low_train_trg = low_train["particle"]
    low_test_df = low_test[training_var]
    low_test_trg = low_test["particle"]
    
    high_train_df = high_train[training_var]
    high_train_trg = high_train["particle"]
    high_test_df = high_test[training_var]
    high_test_trg = high_test["particle"]
    
    full_train_df = full_train[training_var]
    full_train_trg = full_train["particle"]
    full_test_df = full_test[training_var]
    full_test_trg = full_test["particle"]
    
    return low_train_df, low_train_trg, low_test_df, low_test_trg, \
        high_train_df, high_train_trg, high_test_df, high_test_trg, \
        full_train_df, full_train_trg, full_test_df, full_test_trg

def train_model(train_df, train_target, model, name):
    print(f"Training {name} starting")
    start = time.time()
    clf = model.fit(train_df, train_target)
    end = time.time()
    np.save(r"D:/OneDrive/Documents/Universiteit Utrecht/Masters/Computational Aspects of Machine Learning/PR4/models/"+f"{name}.npy", clf)
    print("Training complete")
    print(f"Duration: {(end-start)/60:.0f} min and {(end-start)%60:.1f} sec")
    print("--------------------------------------------------")
    return clf

train_models = False

if train_models:
    dtc = DecisionTreeClassifier(random_state=0)
    rfc = RandomForestClassifier(random_state=0)
    gbc = GradientBoostingClassifier(random_state=0)
    abc = AdaBoostClassifier(n_estimators=100, random_state=0)
    mlpc = MLPClassifier(random_state=1, max_iter=300)
    
    cls_list = [dtc, rfc, gbc, abc, mlpc]
    title_list = ["dtc",  "rfc", "gbc", "abc", "mlpc"]
    models_full = {}
    models_low = {}
    models_high = {}
    
    low_train_df, low_train_trg, low_test_df, low_test_trg, \
    high_train_df, high_train_trg, high_test_df, high_test_trg, \
    full_train_df, full_train_trg, full_test_df, full_test_trg = training_data(particle_df, ["pT", "dEdxTPC", "p", "dEdxITS", "eta", "phi"], 0.9)
    print("Training started at :", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("Training data created")
    print("--------------------------------------------------")
    
    for i in range(0,len(cls_list)):
        clf_low = train_model(low_train_df, low_train_trg, cls_list[i], f"{title_list[i]}_LowMomentum")
        clf_high = train_model(high_train_df, high_train_trg, cls_list[i], f"{title_list[i]}_HighMomentum")
        clf = train_model(full_train_df, full_train_trg, cls_list[i], f"{title_list[i]}_FullMomentum")
        models_low[title_list[i]] = clf_low
        models_high[title_list[i]] = clf_high
        models_full[title_list[i]] = clf
        
    joblib.dump(models_low, folder+r"models/trained_models_LowMomentum")
    joblib.dump(models_high, folder+r"models/trained_models_HighMomentum")
    joblib.dump(models_full, folder+r"models/trained_models_FullMomentum")
    
    low_train_df.to_csv(folder+r"models/data/low_train_df.csv", header=True, index=False)
    low_train_trg.to_csv(folder+r"models/data/low_train_trg.csv", header=True, index=False)
    low_test_df.to_csv(folder+r"models/data/low_test_df.csv", header=True, index=False)
    low_test_trg.to_csv(folder+r"models/data/low_test_trg.csv", header=True, index=False)
    
    high_train_df.to_csv(folder+r"models/data/high_train_df.csv", header=True, index=False)
    high_train_trg.to_csv(folder+r"models/data/high_train_trg.csv", header=True, index=False)
    high_test_df.to_csv(folder+r"models/data/high_test_df.csv", header=True, index=False)
    high_test_trg.to_csv(folder+r"models/data/high_test_trg.csv", header=True, index=False)
    
    full_train_df.to_csv(folder+r"models/data/full_train_df.csv", header=True, index=False)
    full_train_trg.to_csv(folder+r"models/data/full_train_trg.csv", header=True, index=False)
    full_test_df.to_csv(folder+r"models/data/full_test_df.csv", header=True, index=False)
    full_test_trg.to_csv(folder+r"models/data/full_test_trg.csv", header=True, index=False)

    print("All models trained and saved")
    print("Training ended at   :", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
#%% Load trained models and load training/testing data
models_low = joblib.load(folder+r"models/trained_models_LowMomentum") 
models_high = joblib.load(folder+r"models/trained_models_HighMomentum")
models_full = joblib.load(folder+r"models/trained_models_FullMomentum")

low_train_df = pd.read_csv(folder+r"models/data/low_train_df.csv")
low_train_trg = pd.read_csv(folder+r"models/data/low_train_trg.csv")
low_test_df = pd.read_csv(folder+r"models/data/low_test_df.csv")
low_test_trg = pd.read_csv(folder+r"models/data/low_test_trg.csv")

high_train_df = pd.read_csv(folder+r"models/data/high_train_df.csv")
high_train_trg = pd.read_csv(folder+r"models/data/high_train_trg.csv")
high_test_df = pd.read_csv(folder+r"models/data/high_test_df.csv")
high_test_trg = pd.read_csv(folder+r"models/data/high_test_trg.csv")

full_train_df = pd.read_csv(folder+r"models/data/full_train_df.csv")
full_train_trg = pd.read_csv(folder+r"models/data/full_train_trg.csv")
full_test_df = pd.read_csv(folder+r"models/data/full_test_df.csv")
full_test_trg = pd.read_csv(folder+r"models/data/full_test_trg.csv")

#%% Plotting training results 

model_names = list(models_full.keys())
particles = list(full_test_trg["particle"].unique())

for i in model_names:
    model_full = models_full[i]
    model_low = models_low[i]
    model_high = models_high[i]
    
    acc_full = model_full.score(full_test_df, full_test_trg)
    acc_low = model_low.score(low_test_df, low_test_trg)
    acc_high = model_high.score(high_test_df, high_test_trg)
    
    print(f"Model: {i}")
    print(f"Full momentum test accuracy: {acc_full:.4f}")
    print(f"Low momentum test accuracy: {acc_low:.4f}")
    print(f"High momentum test accuracy: {acc_high:.4f}")
    print("--------------------------------------------------")

    plt.ion()   
    fig, ax = plt.subplots(1,3,figsize=(24,6))
    c_matrix_low = confusion_matrix(low_test_trg, model_low.predict(low_test_df), labels = particles, normalize='true')
    c_matrix_high = confusion_matrix(high_test_trg, model_high.predict(high_test_df), labels = particles, normalize='true')
    c_matrix_full = confusion_matrix(full_test_trg, model_full.predict(full_test_df), labels = particles, normalize='true')

    sklearn.metrics.ConfusionMatrixDisplay(c_matrix_low).plot(cmap="Blues", ax=ax[0], colorbar=False)
    ax[0].set_xticklabels(particles, rotation=45)
    ax[0].set_yticklabels(particles, rotation=45)
    ax[0].set_title(f"Low Momentum\nTest Accuracy: {acc_low:.2f}")

    sklearn.metrics.ConfusionMatrixDisplay(c_matrix_high).plot(cmap="Blues", ax=ax[1], colorbar=False)
    ax[1].set_xticklabels(particles, rotation=45)
    ax[1].set_yticklabels([], rotation=45)
    ax[1].set_title(f"High Momentum\nTest Accuracy: {acc_high:.2f}")

    sklearn.metrics.ConfusionMatrixDisplay(c_matrix_full).plot(cmap="Blues", ax=ax[2], colorbar=False)
    ax[2].set_xticklabels(particles, rotation=45)
    ax[2].set_yticklabels([], rotation=45)
    ax[2].set_title(f"Full Momentum\nTest Accuracy: {acc_full:.2f}")

    fig.suptitle(f"Confusion Matrices for {i}")
    plt.savefig(folder+f"models/Confusion_Matrix_{i}.png", dpi=500)

        # ax[0].set_xticklabels("True Label")
        # ax[0].set_ybels("Predicted Label")
        # ax[1] = sklearn.metrics.ConfusionMatrixDisplay(c_matrix_high, display_labels=particles).plot(cmap="Blues", ax=ax[1])
        # ax[1].set_title(f"High Momentum\nTest Accuracy: {acc_high:.2f}")
        # ax[2] = sklearn.metrics.ConfusionMatrixDisplay(c_matrix_full, display_labels=particles).plot(cmap="Blues", ax=ax[2])
        # ax[2].set_title(f"Full Momentum\nTest Accuracy: {acc_full:. 2f}")
        # plt.suptitle(f"Confusion Matrices for {i}")     
        # plt.savefig(folder+f"models/Confusion_Matrix_{i}.png", dpi=500)



# i = "dtc"
# model_full = models_full[i]
# model_low = models_low[i]
# model_high = models_high[i]

# prediction_full = model_full.predict(full_test_df)
# prediction_low = model_low.predict(low_test_df)
# prediction_high = model_high.predict(high_test_df)

# prediction_comb = 

# results_full

# for i in range(0,len(cls_list)): 
#     # i = 0
#     start = time.time() 
#     classifier = cls_list[i]
#     clf = classifier.fit(trn_df, trn_target)
#     acc_test = clf.score(test_df, test_target)
    
#     end = time.time()
#     model_dic[title_list[i]] = clf
    
#     plot_df_trn = trn_df.copy()
#     plot_df_trn["particle"] = trn_target
#     plot_df_test = test_df.copy()
#     plot_df_test["particle"] = test_target
#     print(f"Training duration: {end-start:.2f}")
    
#     for particle, subdf in plot_df_trn.groupby("particle"):
#         ax[0,i].scatter(subdf['pT'], subdf['dEdxTPC'], s=0.3, alpha = 0.5, label=particle)
#     ax[0,i].legend(markerscale=10, loc = "upper right")
#     ax[0,i].set_xscale("log")
#     ax[0,i].set_yscale("log")
#     if i==0:
#         ax[0,i].set_ylabel('dEdxTPC')
#     ax[0,i].grid(alpha=0.4)
#     ax[0,i].grid(which="minor", alpha=0.4)
#     ax[0,i].set_axisbelow(True)
#     ax[0,i].minorticks_on()
#     ax[0,i].set_ylim(10, 2000)
#     ax[0,i].set_xlim(0.1,70)
#     ax[0,i].set_title(title_list[i])
    
#     for particle, subdf in plot_df_test.groupby("particle"):
#         ax[1,i].scatter(subdf['pT'], subdf['dEdxTPC'], s=0.3, alpha = 0.5, label=particle)
#     ax[1,i].legend(markerscale=10, loc = "upper right")
#     ax[1,i].set_xscale("log")
#     ax[1,i].set_yscale("log")
#     ax[1,i].set_ylabel('dEdxTPC')
#     ax[1,i].set_xlabel('pT')
#     ax[1,i].grid(alpha=0.4)
#     ax[1,i].grid(which="minor", alpha=0.4)
#     ax[1,i].set_axisbelow(True)
#     ax[1,i].minorticks_on()
#     ax[1,i].set_ylim(10, 2000)
#     ax[1,i].set_xlim(0.1,70)
#     ax[1,i].set_title(f"Test accuracy: {acc_test:.2f}")
    
#     print(f"{title_list[i]} complete")
