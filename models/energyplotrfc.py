import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

# 1. Path Setup
base_dir = Path(__file__).resolve().parent
models_dir = base_dir / "models"
export_dir = models_dir / "data"

# 2. Load RFC Model and Data
# Change 'final_rfc_model.joblib' to whatever your RFC filename is
rfc_model_path = models_dir / "best_trained_rfc1_full.joblib" 
rfc = joblib.load(rfc_model_path)

# Using 'full' data for the visualization
X_test = pd.read_csv(export_dir / "full_test_df.csv")
y_test = pd.read_csv(export_dir / "full_test_trg.csv").values.ravel().astype(str)

# 3. Generate Predictions
# RFC usually returns the class name (string) if it was trained on strings
y_pred = rfc.predict(X_test).astype(str)

# 4. Create Boolean Masks
# Compare strings directly
correct_mask = (y_test == y_pred)
incorrect_mask = (y_test != y_pred)

# 5. Extract features
pt = X_test['pT']
its = X_test['dEdxITS']
tpc = X_test['dEdxTPC']

# 6. Plotting
fig, ax = plt.subplots(1, 2, figsize=(15, 6), layout="constrained")

# ITS Plot
ax[0].scatter(pt[correct_mask], its[correct_mask], s=0.5, c='navy', alpha=0.2, label='Correct')
ax[0].scatter(pt[incorrect_mask], its[incorrect_mask], s=1.5, c='red', alpha=0.7, label='Incorrect')
ax[0].set_title("RFC: dEdxITS vs pT")
ax[0].set_ylabel("dEdxITS")

# TPC Plot
ax[1].scatter(pt[correct_mask], tpc[correct_mask], s=0.5, c='navy', alpha=0.2, label='Correct')
ax[1].scatter(pt[incorrect_mask], tpc[incorrect_mask], s=1.5, c='red', alpha=0.7, label='Incorrect')
ax[1].set_title("RFC: dEdxTPC vs pT")
ax[1].set_ylabel("dEdxTPC")

# Formatting for both
for a in ax:
    a.set_xlabel("pT")
    a.set_xscale('log')
    a.set_yscale('log')
    a.grid(True, which="both", alpha=0.3)
    a.legend(markerscale=8)

plt.show()
