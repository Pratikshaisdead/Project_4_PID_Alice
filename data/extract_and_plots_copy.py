import uproot
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
#import time

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

print(particles)