#classnames in kaons {'data;1': 'TTree'}
#TTrees in directory: ['index', 'pT', 'eta', 'phi', 'p', 'dEdxITS', 'NclusterPIDTPC', 'dEdxTPC', 'ITSclsMap', 'NclusterTPC', 'NcrossedRowsTPC', 'NFindableTPC']

#Explaination of the branches in the Data:
#pT - transverse momenta of the particles
#eta - pseudorapidity
#phi - phi of the track, in radians withing [0, 2pi)
#dEdxITS - the energy loss in the ITS
#ITSclsMap - 
#NclusterPIDTPC - 
#NclusterTPC - number of TPC clusters
#NcrossedRowsTPC - number of crossed TPC rows
#NFindableTPC - findable TPC clusters for this track geometry
#dEdxTPC - energy loss in the TPC



import uproot
import matplotlib.pyplot as plt


kaons = uproot.open("data/Kaons.root")
pions = uproot.open("data/Pions.root")
protons = uproot.open("data/Protons.root")
electrons = uproot.open("data/Electrons.root")
deuterons = uproot.open("data/Deuterons.root")

myKaonTree = kaons["data;1"]
myPionTree = pions["data;1"]
myProtonTree = protons["data;1"]
myElectronTree = electrons["data;1"]
myDeuteronTree = deuterons["data;1"]

kaonBranches = myKaonTree.arrays()
pionBranches = myPionTree.arrays()
protonBranches = myProtonTree.arrays()
electronBranches = myElectronTree.arrays()
deuteronBranches = myDeuteronTree.arrays()


yk_00 = kaonBranches ['dEdxITS']
yk_11 = kaonBranches ['dEdxTPC']
xk = kaonBranches ['pT']

ypi_00 = pionBranches['dEdxITS']
ypi_11 = pionBranches['dEdxTPC']
xpi = pionBranches ['pT']

yp_00 = protonBranches['dEdxITS']
yp_11 = protonBranches['dEdxTPC']
xp = protonBranches ['pT']

ye_00 = electronBranches['dEdxITS']
ye_11 = electronBranches ['dEdxTPC']
xe = electronBranches['pT']

yd_00 = deuteronBranches['dEdxITS']
yd_11 = deuteronBranches['dEdxTPC']
xd = deuteronBranches['pT']


fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(10, 5))


ax[0].scatter(xpi, ypi_00, 0.1, c='purple', alpha = 0.4, label= 'pions')
ax[0].scatter(xp, yp_00, 0.1, c='green', alpha = 0.4, label= 'protons')
ax[0].scatter(xk, yk_00, 0.1, c='blue', alpha = 0.4, label= 'kaons')
ax[0].scatter(xd, yd_00, 0.1, c='darkred', alpha = 0.4, label='deutrons')
ax[0].scatter(xe, ye_00, 0.1, c='red', alpha = 0.4, label='electrons')

#ax[0].set_title('dEdxITS')
ax[0].set_ylabel(r'Energy loss $\mathrm{d}E/\mathrm{d}x$ (Gev/m) at ITS', fontsize = 12)
ax[0].set_xlabel(r'Momentum p (GeV/c)', fontsize  = 12)

ax[0].set_xscale('log')
ax[0].set_yscale('log')
ax[0].grid(alpha = 0.4)
ax[0].grid(which = "minor", alpha = 0.4)
ax[0].set_axisbelow(True)
ax[0].minorticks_on()
ax[0].tick_params(axis='both', which='major', labelsize=10)
ax[0].tick_params(axis='both', which='minor', labelsize=10)
#ax[0].set_yticks(fontsize = 10)
#ax[0].set_xticks(fontsize = 10)
ax[0].legend(markerscale=10)



ax[1].scatter(xpi, ypi_11, 0.1, c='purple', alpha = 0.44, label='pions')
ax[1].scatter(xp, yp_11, 0.1, c='green', alpha = 0.44, label='protons')
ax[1].scatter(xk, yk_11, 0.1, c='blue', alpha = 0.44, label='kaons')
ax[1].scatter(xd, yd_11, 0.1, c='darkred', alpha = 0.4, label='deutrons')
ax[1].scatter(xe, ye_11, 0.1, c='red', alpha = 0.4, label='electrons')

#ax[1].set_title('$\mathrm{d}E\mathrm{d}x TPC$')
ax[1].set_ylabel(r'Energy loss $\mathrm{d}E/\mathrm{d}x$ (Gev/m) at TPC', fontsize = 12)
ax[1].set_xlabel(r'Momentum p (GeV/c)', fontsize = 12)

ax[1].set_xscale('log')
ax[1].set_yscale('log')
ax[1].grid(alpha = 0.4)
ax[1].grid(which = "minor", alpha = 0.4)
ax[1].set_axisbelow(True)
ax[1].minorticks_on()
ax[1].tick_params(axis='both', which='major', labelsize=10)
ax[1].tick_params(axis='both', which='minor', labelsize=10)
ax[1].legend(markerscale=10)


fig.tight_layout(pad = 5)

plt.show()
