import uproot
import matplotlib.pyplot as plt

# --- 1. Load Data Directly ---
# We use library='np' so we can plot immediately without extra conversion steps
kaons     = uproot.open("data\Kaons.root")["data;1"].arrays(library='np')
pions     = uproot.open("data\Pions.root")["data;1"].arrays(library='np')
protons   = uproot.open("data\Protons.root")["data;1"].arrays(library='np')
electrons = uproot.open("data\Electrons.root")["data;1"].arrays(library='np')
deuterons = uproot.open("data\Deuterons.root")["data;1"].arrays(library='np')

# --- 2. Configuration for Histogram Plots ---
# Defining all particles and their colors centrally for the histograms
HIST_PARTICLES = [
    ('Protons',   protons,   'blue'),
    ('Pions',     pions,     'green'),
    ('Electrons', electrons, 'red'),
    ('Kaons',     kaons,     'orange'),  # Added
    ('Deuterons', deuterons, 'purple'), # Added
]

# --- 3. Create Plot Layout ---
fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(10, 8), layout="constrained")

# Define common look for scatters
scatter_style = {'s': 0.1, 'alpha': 0.4} 

# ---------------------------------------------------------
# Plot [0,0]: pT vs dEdxITS (Scatter)
# ---------------------------------------------------------
ax00 = axes[0,0]
ax00.scatter(pions['pT'],     pions['dEdxITS'],     c='navy', **scatter_style)
ax00.scatter(protons['pT'],   protons['dEdxITS'],   c='navy', **scatter_style)
ax00.scatter(kaons['pT'],     kaons['dEdxITS'],     c='navy', **scatter_style)
ax00.scatter(deuterons['pT'], deuterons['dEdxITS'], c='navy', label='other', **scatter_style)
ax00.scatter(electrons['pT'], electrons['dEdxITS'], c='red',  label='electrons', **scatter_style)

ax00.set_title('dEdxITS')
ax00.set_ylabel('dEdxITS')
ax00.set_xlabel('pT')
ax00.set_xscale('log')
ax00.set_yscale('log')
ax00.legend(markerscale=10)

# ---------------------------------------------------------
# Plot [0,1]: NcrossedRowsTPC (Histogram - All Particles)
# ---------------------------------------------------------
ax01 = axes[0,1]
for name, data_obj, color in HIST_PARTICLES:
    ax01.hist(data_obj['NcrossedRowsTPC'], bins=150, color=color, label=name, alpha=0.7)

ax01.set_title('NcrossedRowsTPC')
ax01.set_ylabel('NcrossedRowsTPC Counts')
ax01.set_yscale('log')
ax01.legend()

# ---------------------------------------------------------
# Plot [1,0]: NclusterPIDTPC (Histogram - All Particles)
# ---------------------------------------------------------
ax10 = axes[1,0]
for name, data_obj, color in HIST_PARTICLES:
    ax10.hist(data_obj['NclusterPIDTPC'], bins=150, color=color, label=name, alpha=0.7)

ax10.set_title('NclusterPIDTPC')
ax10.set_ylabel('NclusterPIDTPC Counts')
ax10.set_xlabel('pT')
ax10.legend()

# ---------------------------------------------------------
# Plot [1,1]: pT vs dEdxTPC (Scatter)
# ---------------------------------------------------------
ax11 = axes[1,1]
ax11.scatter(pions['pT'],     pions['dEdxTPC'],     c='navy', **scatter_style)
ax11.scatter(protons['pT'],   protons['dEdxTPC'],   c='navy', **scatter_style)
ax11.scatter(kaons['pT'],     kaons['dEdxTPC'],     c='navy', **scatter_style)
ax11.scatter(deuterons['pT'], deuterons['dEdxTPC'], c='navy', label='other', **scatter_style)
ax11.scatter(electrons['pT'], electrons['dEdxTPC'], c='red',  label='electrons', **scatter_style)

ax11.set_title('dEdxTPC')
ax11.set_ylabel('dEdxTPC')
ax11.set_xlabel('pT')
ax11.set_xscale('log')
ax11.set_yscale('log')
ax11.legend(markerscale=10)

# ---------------------------------------------------------
# Global Styling (Grid)
# ---------------------------------------------------------
for ax in axes.flat:
    ax.grid(alpha=0.4)
    ax.grid(which="minor", alpha=0.4)
    ax.set_axisbelow(True)
    ax.minorticks_on()

plt.show()
