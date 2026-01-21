#%% Importing libraries
import numpy as np
import pandas as pd 

import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
plt.style.use("default")

import sklearn
from sklearn.utils import shuffle 
from sklearn import preprocessing as pp
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import joblib

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier

import uproot
import optuna
import time
from datetime import datetime
from pathlib import Path
import os 

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

def cutoff_func(df, variable, cutoff):
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
#%% Plot histogram of data as function of variable 
def plot_histogram(variable, df, bins=100): 
    plt.ion()
    fig, ax = plt.subplots(figsize=(8,6))
    
    
    ax.hist(df[variable], bins=bins, alpha=0.5, density=True)
        
    ax.set_xlabel(variable)
    ax.set_ylabel("Density")
    ax.set_title(f"Histogram of {variable}")
    ax.legend()
    # ax.grid(alpha=0.4)
    ax.grid(which="minor", alpha=0.4)
    ax.set_axisbelow(True)
    ax.minorticks_on()
    # ax.set_xscale("log")
    ax.set_yscale("log")
    
    plt.show()

plot_histogram("pT", particle_df, bins=200)
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

plot_from_df("pT", "dEdxTPC", particle_df)
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

def training_data(df, training_var, train_per, cutoff):
    #Splitting the DataFrame into a low momentum and a high momentum DataFrame
    low_df, high_df = cutoff_func(df, "pT", cutoff)
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
    print(f"Time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S")) 
    start = time.time()
    clf = model.fit(train_df, train_target)
    end = time.time()
    np.save(r"D:/OneDrive/Documents/Universiteit Utrecht/Masters/Computational Aspects of Machine Learning/PR4/models/"+f"{name}.npy", clf)
    print("Training complete")
    print(f"Duration: {(end-start)/60:.0f} min and {(end-start)%60:.1f} sec")
    print("--------------------------------------------------")
    return clf

train_models = True
cutoff_arr = np.linspace(0.2, 0.6, 21)  # Cutoff values to try

if train_models:
    for k_cutoff in range(0,len(cutoff_arr)): 
        cutoff_pT = cutoff_arr[k_cutoff]
        print(f"Cutoff value: {cutoff_pT:.2f} GeV/c")

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
        full_train_df, full_train_trg, full_test_df, full_test_trg = training_data(particle_df, 
                                                                                   ["pT", "dEdxTPC", "p", "dEdxITS", "eta", "phi"], 
                                                                                   0.2, # Training percentage
                                                                                   cutoff_pT)
        print("Training started at :", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print("Training data created")
        print("--------------------------------------------------")
        
        for i in range(0,len(cls_list)):
        # for i in range(0,1):
            clf_low = train_model(low_train_df, low_train_trg, cls_list[i], f"{title_list[i]}_LowMomentum_Cutoff={cutoff_pT:.3f}")
            clf_high = train_model(high_train_df, high_train_trg, cls_list[i], f"{title_list[i]}_HighMomentum_Cutoff={cutoff_pT:.3f}")
            if k_cutoff==0: 
                clf = train_model(full_train_df, full_train_trg, cls_list[i], f"{title_list[i]}_FullMomentum")
            models_low[title_list[i]] = clf_low
            models_high[title_list[i]] = clf_high
            models_full[title_list[i]] = clf
        
        cutoff_folder = f"cutoff={cutoff_pT:.3f}/"
        Path(folder+r"models/cutoff/"+cutoff_folder).mkdir(parents=True, exist_ok=True)
        print("Saving trained models and data...")

        joblib.dump(models_low, folder+r"models/cutoff/"+cutoff_folder+r"/trained_models_LowMomentum")
        joblib.dump(models_high, folder+r"models/cutoff/"+cutoff_folder+r"/trained_models_HighMomentum")
        if k_cutoff ==0:
            joblib.dump(models_full, folder+r"models/cutoff/"+cutoff_folder+r"/trained_models_FullMomentum")
        
        # Saving training and testing data 
        # Commented out to to speed up testing when not retraining training data for cutoff 

        # low_train_df.to_csv(folder+r"models/data/low_train_df.csv", header=True, index=False)
        # low_train_trg.to_csv(folder+r"models/data/low_train_trg.csv", header=True, index=False)
        # low_test_df.to_csv(folder+r"models/data/low_test_df.csv", header=True, index=False)
        # low_test_trg.to_csv(folder+r"models/data/low_test_trg.csv", header=True, index=False)
        
        # high_train_df.to_csv(folder+r"models/data/high_train_df.csv", header=True, index=False)
        # high_train_trg.to_csv(folder+r"models/data/high_train_trg.csv", header=True, index=False)
        # high_test_df.to_csv(folder+r"models/data/high_test_df.csv", header=True, index=False)
        # high_test_trg.to_csv(folder+r"models/data/high_test_trg.csv", header=True, index=False)
        
            full_train_df.to_csv(folder+r"models/cutoff/data/full_train_df.csv", header=True, index=False)
            full_train_trg.to_csv(folder+r"models/cutoff/data/full_train_trg.csv", header=True, index=False)
            full_test_df.to_csv(folder+r"models/cutoff/data/full_test_df.csv", header=True, index=False)
            full_test_trg.to_csv(folder+r"models/cutoff/data/full_test_trg.csv", header=True, index=False)

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
    ax[0].set_title(f"Low Momentum\nTest Accuracy: {acc_low:.4f}")

    sklearn.metrics.ConfusionMatrixDisplay(c_matrix_high).plot(cmap="Blues", ax=ax[1], colorbar=False)
    ax[1].set_xticklabels(particles, rotation=45)
    ax[1].set_yticklabels([], rotation=45)
    ax[1].set_title(f"High Momentum\nTest Accuracy: {acc_high:.4f}")

    sklearn.metrics.ConfusionMatrixDisplay(c_matrix_full).plot(cmap="Blues", ax=ax[2], colorbar=False)
    ax[2].set_xticklabels(particles, rotation=45)
    ax[2].set_yticklabels([], rotation=45)
    ax[2].set_title(f"Full Momentum\nTest Accuracy: {acc_full:.4f}")

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

#%% Combine models 

def combine_models(low_model, high_model, test_df, test_trg, cutoff):
    # Mask for high and low momentum
    low_mask = test_df["pT"]<cutoff
    high_mask = test_df["pT"]>=cutoff

    # Predictions
    low_predictions = low_model.predict(test_df[low_mask])
    high_predictions = high_model.predict(test_df[high_mask])

    # True labels
    low_true = test_trg[low_mask]
    high_true = test_trg[high_mask]

    # Combine in the SAME order
    predictions = np.concatenate([low_predictions, high_predictions])
    true_labels = np.concatenate([low_true, high_true])

    accuracy = accuracy_score(true_labels, predictions)
    accuracy_low = accuracy_score(low_true, low_predictions)
    accuracy_high = accuracy_score(high_true, high_predictions)

    return predictions, true_labels, accuracy, accuracy_low, accuracy_high


folder = r'D:/OneDrive/Documents/Universiteit Utrecht/Masters/Computational Aspects of Machine Learning/PR4/'

cutoff_arr = [
    float(name.split("=")[1])
    for name in os.listdir(folder+"models/cutoff/")
    if name.startswith("cutoff=")
]
cutoff_arr = sorted(cutoff_arr)

# Arrays to store accuracies
dtc_arr = np.zeros((len(cutoff_arr),3))
rfc_arr = np.zeros((len(cutoff_arr),3))
gbc_arr = np.zeros((len(cutoff_arr),3))
abc_arr = np.zeros((len(cutoff_arr),3))
mlpc_arr = np.zeros((len(cutoff_arr),3))

#load data
full_test_df = pd.read_csv(folder+r"models/cutoff/data/full_test_df.csv")
full_test_trg = pd.read_csv(folder+r"models/cutoff/data/full_test_trg.csv")
full_test_df = full_test_df[:200000] # Limit size for faster testing
full_test_trg = full_test_trg[:200000] # Limit size for faster testing

for i in range(len(cutoff_arr)):
    cutoff_pT = cutoff_arr[i]
    #Load models 
    low_models = joblib.load(folder+r"models/cutoff/" + f"cutoff={cutoff_pT:.3f}/trained_models_LowMomentum")
    high_models = joblib.load(folder+r"models/cutoff/" + f"cutoff={cutoff_pT:.3f}/trained_models_HighMomentum")
    
    # Combine models and get accuracy
    for model_name in low_models.keys():
        predictions, true_labels, accuracy, accuracy_low, accuracy_high = combine_models(
            low_models[model_name],
            high_models[model_name],
            full_test_df,
            full_test_trg,
            cutoff=cutoff_pT)
        
        if model_name == "dtc":
            dtc_arr[i,0] = accuracy
            dtc_arr[i,1] = accuracy_low
            dtc_arr[i,2] = accuracy_high
            print("dtc accuracy at cutoff {:.3f}: {:.4f}".format(cutoff_pT, accuracy))
        elif model_name == "rfc":
            rfc_arr[i,0] = accuracy
            rfc_arr[i,1] = accuracy_low
            rfc_arr[i,2] = accuracy_high
            print("rfc accuracy at cutoff {:.3f}: {:.4f}".format(cutoff_pT, accuracy))
        elif model_name == "gbc":
            gbc_arr[i,0] = accuracy
            gbc_arr[i,1] = accuracy_low
            gbc_arr[i,2] = accuracy_high
            print("gbc accuracy at cutoff {:.3f}: {:.4f}".format(cutoff_pT, accuracy))
        elif model_name == "abc":
            abc_arr[i,0] = accuracy
            abc_arr[i,1] = accuracy_low
            abc_arr[i,2] = accuracy_high
            print("abc accuracy at cutoff {:.3f}: {:.4f}".format(cutoff_pT, accuracy))
        elif model_name == "mlpc":
            mlpc_arr[i,0] = accuracy
            mlpc_arr[i,1] = accuracy_low
            mlpc_arr[i,2] = accuracy_high
            print("mlpc accuracy at cutoff {:.3f}: {:.4f}".format(cutoff_pT, accuracy))

    print("--------------------------------------------------")

#%%
models_full = joblib.load(r"D:\OneDrive\Documents\Universiteit Utrecht\Masters\Computational Aspects of Machine Learning\PR4\models\cutoff\cutoff=0.200\trained_models_FullMomentum")
# Accuracy default models
for model_name in models_full.keys():
    model = models_full[model_name]
    accuracy = model.score(full_test_df, full_test_trg)
    print(f"{model_name} default model accuracy: {accuracy:.4f}")

plt.ion()
fig, ax = plt.subplots(figsize=(10,6))
ax.plot(cutoff_arr, dtc_arr[:,0], label="Decision Tree Classifier", color="tab:blue", marker="o")
ax.plot(cutoff_arr, dtc_arr[:,1], label="Decision Tree Classifier Low Momentum", color="tab:blue", marker="x", alpha=0.3)
ax.plot(cutoff_arr, dtc_arr[:,2], label="Decision Tree Classifier High Momentum", color="tab:blue", marker="^", alpha=0.3)

ax.plot(cutoff_arr, rfc_arr[:,0], label="Random Forest Classifier", color="tab:orange", marker="o")
ax.plot(cutoff_arr, rfc_arr[:,1], label="Random Forest Classifier Low Momentum", color="tab:orange", marker="x", alpha=0.3)
ax.plot(cutoff_arr, rfc_arr[:,2], label="Random Forest Classifier High Momentum", color="tab:orange", marker="^", alpha=0.3)

ax.plot(cutoff_arr, gbc_arr[:,0], label="Gradient Boosting Classifier", color="tab:green", marker="o")
ax.plot(cutoff_arr, gbc_arr[:,1], label="Gradient Boosting Classifier Low Momentum", color="tab:green", marker="x", alpha=0.3)
ax.plot(cutoff_arr, gbc_arr[:,2], label="Gradient Boosting Classifier High Momentum", color="tab:green", marker="^", alpha=0.3)

ax.plot(cutoff_arr, abc_arr[:,0], label="AdaBoost Classifier", color="tab:red", marker="o")
ax.plot(cutoff_arr, abc_arr[:,1], label="AdaBoost Classifier Low Momentum", color="tab:red", marker="x", alpha=0.3)
ax.plot(cutoff_arr, abc_arr[:,2], label="AdaBoost Classifier High Momentum", color="tab:red", marker="^", alpha=0.3)

ax.plot(cutoff_arr, mlpc_arr[:,0], label="Multi-layer Perceptron Classifier", color="tab:purple", marker="o")
ax.plot(cutoff_arr, mlpc_arr[:,1], label="Multi-layer Perceptron Classifier Low Momentum", color="tab:purple", marker="x", alpha=0.3)
ax.plot(cutoff_arr, mlpc_arr[:,2], label="Multi-layer Perceptron Classifier High Momentum", color="tab:purple", marker="^", alpha=0.3)

# ax.set_xscale("log")
ax.set_xlabel("Cutoff momentum (GeV/c)")
ax.set_ylabel("Test Accuracy")
ax.set_title("Combined Model Test Accuracy vs Cutoff Momentum")
ax.legend()
plt.show()
#%%



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

# %%
