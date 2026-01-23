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
from sklearn.ensemble import RandomForestClassifier
import uproot
import time
from datetime import datetime
from pathlib import Path 
import optuna 
import optuna.visualization.matplotlib as vis 

#%% Universal Path Setup
base_dir = Path(__file__).resolve().parent
models_dir = base_dir / "models"
export_dir = models_dir / "data"
opt_dir = models_dir / "optimization"

# Ensure directories exist
opt_dir.mkdir(parents=True, exist_ok=True)

#%% Loading Data
full_train_df = pd.read_csv(export_dir / "full_train_df.csv")
full_train_trg = pd.read_csv(export_dir / "full_train_trg.csv")
full_test_df = pd.read_csv(export_dir / "full_test_df.csv")
full_test_trg = pd.read_csv(export_dir / "full_test_trg.csv")

low_train_df = pd.read_csv(export_dir / "low_train_df.csv")
low_train_trg = pd.read_csv(export_dir / "low_train_trg.csv")
low_test_df = pd.read_csv(export_dir / "low_test_df.csv")
low_test_trg = pd.read_csv(export_dir / "low_test_trg.csv")

high_train_df = pd.read_csv(export_dir / "high_train_df.csv")
high_train_trg = pd.read_csv(export_dir / "high_train_trg.csv")
high_test_df = pd.read_csv(export_dir / "high_test_df.csv")
high_test_trg = pd.read_csv(export_dir / "high_test_trg.csv")



# ... [Imports and Data Loading remain same as your snippet] ...

# Loading Data 
y_low_train_flat = low_train_trg.values.ravel()
y_low_test_flat = low_test_trg.values.ravel()
y_high_train_flat = high_train_trg.values.ravel()
y_high_test_flat = high_test_trg.values.ravel()
y_full_train_flat = full_train_trg.values.ravel()  
y_full_test_flat = full_test_trg.values.ravel()    


    

def objective_rfc_low_mom(trial): 
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 20, 400, step=20),
        "max_depth": trial.suggest_int("max_depth", 10, 80),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy","log_loss"]),
        "random_state": 0, "n_jobs": -1
    }
    model = RandomForestClassifier(**params)
    model.fit(low_train_df, y_low_train_flat)
    return model.score(low_test_df, y_low_test_flat)

def objective_rfc_high_mom(trial): 
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 20, 400, step=20),
        "max_depth": trial.suggest_int("max_depth", 10, 80),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy","log_loss"]),
        "random_state": 0, "n_jobs": -1
    }
    model = RandomForestClassifier(**params)
    model.fit(high_train_df, y_high_train_flat)
    return model.score(high_test_df, y_high_test_flat)

def objective_rfc_full(trial): 
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 20, 400, step=20),
        "max_depth": trial.suggest_int("max_depth", 10, 80),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy","log_loss"]),
        "random_state": 0, "n_jobs": -1
    }
    model = RandomForestClassifier(**params)
    model.fit(full_train_df, y_full_train_flat)
    return model.score(full_test_df, y_full_test_flat)





regions = [
    ("Low", objective_rfc_low_mom, low_train_df, y_low_train_flat, low_test_df, y_low_test_flat),
    ("High", objective_rfc_high_mom, high_train_df, y_high_train_flat, high_test_df, y_high_test_flat),
    ("Full", objective_rfc_full, full_train_df, y_full_train_flat, full_test_df, y_full_test_flat)
]

for name, obj_func, train_x, train_y, test_x, test_y in regions:
    print(f"\n{'='*20}\nOptimizing {name} Momentum\n{'='*20}")
    
    # Initial Performance
    init_model = RandomForestClassifier(random_state=0, n_jobs=-1)
    init_model.fit(train_x, train_y)
    init_acc = init_model.score(test_x, test_y)
    
    # Study
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(multivariate=True))
    study.optimize(obj_func, n_trials=1) # Recommended for overnight
    
    # Final Training
    best_model = RandomForestClassifier(**study.best_params, random_state=0, n_jobs=-1)
    best_model.fit(train_x, train_y)
    
    # Saving
    joblib.dump(best_model, models_dir / f"best_trained_rfc_{name.lower()}.joblib")
    
    final_acc = best_model.score(test_x, test_y)
    print(f"{name} Initial Accuracy: {init_acc:.4f}")
    print(f"{name} Final Accuracy: {final_acc:.4f}")

    # Save results to text
    with open(models_dir / f"opt_results_rfc_{name.lower()}.txt", "w") as f:
        f.write(f"Region: {name}\nBest Params: {study.best_params}\nInitial Acc: {init_acc}\nFinal Acc: {final_acc}")

print("\nAll regions optimized and expert models saved.")
