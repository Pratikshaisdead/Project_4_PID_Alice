import uproot
import numpy as np
import pandas as pd 
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from matplotlib.colors import LogNorm
from sklearn.mixture import GaussianMixture  
import time
from sklearn.utils import shuffle
from sklearn import preprocessing as pp
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier

dtc = DecisionTreeClassifier(random_state=0)
rfc = RandomForestClassifier(random_state=0)
gbc = GradientBoostingClassifier(random_state=0)
abc = AdaBoostClassifier(n_estimators=100, random_state=0)
mlpc = MLPClassifier(random_state=1, max_iter=300)
gpc = GaussianProcessClassifier()


pions = uproot.open("data\Pions.root")
kaons = uproot.open("data\Kaons.root")
deuterons = uproot.open("data\Deuterons.root")
electrons = uproot.open("data\Electrons.root")
protons = uproot.open("data\Protons.root")

# Open Root files and extract data

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

low_momentum_df, high_momentum_df = cutoff(particle_df, "pT", 1.0)

# Plots

def plot_from_df(variable_x, variable_y, df,save_appendix=""):
    start = time.time()
    x_arr = df[variable_x]
    y_arr = df[variable_y]
    labels = df["particle"]
    
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
    
    folder = r'Plots\energy_loss'
    fig.suptitle(variable_x+" VS "+variable_y)
    file_name = folder+variable_x+"_"+variable_y+save_appendix+".png"
    plt.savefig(file_name,dpi=500)

    print("Amount of data points plotted: {:.2e}".format(len(y_arr)))
    end = time.time()
    print(end-start)

# plot_from_df("pT", "dEdxTPC", particle_df)
# plot_from_df("pT", "dEdxTPC", low_momentum_df,"_LowMomentum")
plot_from_df("pT", "dEdxTPC", high_momentum_df,"_HighMomentum")

# Simple Classifier

def training_data(df, training_var, classifier, train_per = 0.8):
    #Turns the data into 
    if classifier not in ["kaons", "pions", "protons", "electrons", "deuterons"]:
        print("Error: select one of following particles: ")
        print("kaons, pions, protons, electrons, deuterons")
    subdf = df[training_var]
    target = df["particle"]==classifier
    subdf, target = shuffle(subdf,target)
    training_df = subdf[:int(len(subdf)*train_per)]
    training_target = target[:int(len(target)*train_per)]
    test_df = subdf[int(len(subdf)*train_per):]
    test_target = target[int(len(target)*train_per):]
    return training_df, training_target, test_df, test_target

trn_df, trn_target, test_df, test_target = training_data(particle_df, ["pT", "dEdxTPC"], "electrons", 0.9)

dtc = DecisionTreeClassifier(random_state=0)
rfc = RandomForestClassifier(random_state=0)
gbc = GradientBoostingClassifier(random_state=0)
abc = AdaBoostClassifier(n_estimators=100, random_state=0)
mlpc = MLPClassifier(random_state=1, max_iter=300)
gpc = GaussianProcessClassifier()
cls_list = [dtc, rfc, gbc, abc, mlpc]
title_list = ["dtc",  "rfc", "gbc", "abc", "mlpc"]

fig, ax = plt.subplots(2,5,figsize=(25,12))

for i in range(0,len(cls_list)): 
    classifier = cls_list[i]
    clf = classifier.fit(trn_df, trn_target)
    acc_test = clf.score(test_df, test_target)
    
    plot_df_trn = trn_df
    plot_df_trn["particle"] = trn_target
    plot_df_test = test_df
    plot_df_test["particle"] = test_target
    
    for particle, subdf in plot_df_trn.groupby("particle"):
        ax[0,i].scatter(subdf['pT'], subdf['dEdxTPC'], s=0.3, alpha = 0.5, label=particle)
    ax[0,i].legend(markerscale=10, loc = "upper right")
    ax[0,i].set_xscale("log")
    ax[0,i].set_yscale("log")
    ax[0,i].set_ylabel('dEdxTPC')
    ax[0,i].set_xlabel('pT')
    ax[0,i].grid(alpha=0.4)
    ax[0,i].grid(which="minor", alpha=0.4)
    ax[0,i].set_axisbelow(True)
    ax[0,i].minorticks_on()
    ax[0,i].set_ylim(10, 2000)
    ax[0,i].set_xlim(0.1,70)
    ax[0,i].set_title(title_list[i])
    
    for particle, subdf in plot_df_test.groupby("particle"):
        ax[1,i].scatter(subdf['pT'], subdf['dEdxTPC'], s=0.3, alpha = 0.5, label=particle)
    ax[1,i].legend(markerscale=10, loc = "upper right")
    ax[1,i].set_xscale("log")
    ax[1,i].set_yscale("log")
    ax[1,i].set_ylabel('dEdxTPC')
    ax[1,i].set_xlabel('pT')
    ax[1,i].grid(alpha=0.4)
    ax[1,i].grid(which="minor", alpha=0.4)
    ax[1,i].set_axisbelow(True)
    ax[1,i].minorticks_on()
    ax[1,i].set_ylim(10, 2000)
    ax[1,i].set_xlim(0.1,70)
    ax[1,i].set_title(f"Test accuracy: {acc_test:.2f}")
    
    print(f"{title_list[i]} complete")


