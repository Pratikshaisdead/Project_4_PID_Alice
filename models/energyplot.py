import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

# 1. Path Setup
base_dir = Path(__file__).resolve().parent
#models_dir = base_dir / "models"
#export_dir = models_dir / "data"

# 2. Load Label Encoder and Data
le = LabelEncoder()
full_trg = pd.read_csv(base_dir / "full_train_trg.csv")
le.fit(full_trg.values.ravel().astype(str))

# Using the 'full' region for the example
reg = "full"
model = joblib.load(base_dir / f"final_fixed_MLPC_{reg}.joblib")
print(model)
X_test = pd.read_csv(base_dir / f"{reg}_test_df.csv")
y_test_raw = pd.read_csv(base_dir / f"{reg}_test_trg.csv").values.ravel().astype(str)
y_test = le.transform(y_test_raw)

# 3. Generate Predictions
y_pred = model.predict(X_test).astype(str)

# 4. Create Boolean Masks for Correct/Incorrect
correct_mask = (y_test == y_pred)
incorrect_mask = (y_test != y_pred)

# 5. Extract specific features for plotting
# Note: Ensure these column names match your CSV headers exactly
pt = X_test['pT']
dedx_its = X_test['dEdxITS']
dedx_tpc = X_test['dEdxTPC']

# 6. Plotting
fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(14, 6), layout="constrained")

# Titles and Labels
titles = ['dEdxITS vs pT', 'dEdxTPC vs pT']
y_datas = [dedx_its, dedx_tpc]
y_labels = ['dEdxITS', 'dEdxTPC']

for i in range(2):
    # Plot Correct Identifications (Navy)
    ax[i].scatter(pt[correct_mask], y_datas[i][correct_mask], 
                  s=0.5, c='navy', alpha=0.3, label='Correctly Identified')
    
    # Plot False Identifications (Red) - Plotted second so they stay on top
    ax[i].scatter(pt[incorrect_mask], y_datas[i][incorrect_mask], 
                  s=1.0, c='red', alpha=0.8, label='False Identification')

    ax[i].set_title(titles[i])
    ax[i].set_ylabel(y_labels[i])
    ax[i].set_xlabel('pT')
    ax[i].set_xscale('log')
    ax[i].set_yscale('log')
    ax[i].grid(True, which="both", alpha=0.3)
    ax[i].legend(markerscale=10)

plt.show()
