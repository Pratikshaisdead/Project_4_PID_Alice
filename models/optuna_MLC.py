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
from sklearn.neural_network import MLPClassifier
from joblib import parallel_backend
from sklearn.preprocessing import LabelEncoder

# Universal Path Setup
base_dir = Path(__file__).resolve().parent
models_dir = base_dir / "models"
export_dir = models_dir / "data"
opt_dir = models_dir / "optimization"



# Ensure directories exist
opt_dir.mkdir(parents=True, exist_ok=True)

mlpc = MLPClassifier(random_state=1, max_iter=300)

# Loading Data
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

le = LabelEncoder()

# Flattening targets
y_low_train_flat = le.fit_transform(low_train_trg.values.ravel())
y_low_test_flat = le.transform(low_test_trg.values.ravel())

y_high_train_flat = le.fit_transform(high_train_trg.values.ravel())
y_high_test_flat = le.transform(high_test_trg.values.ravel())

y_full_train_flat = le.fit_transform(full_train_trg.values.ravel())
y_full_test_flat = le.transform(full_test_trg.values.ravel()) 

# Objectives
def objective_MLPC_low_mom(trial): 
    momentum_val = trial.suggest_float("momentum", 0.1, 0.9)
    batch_size_val = trial.suggest_categorical("batch_size", [32, 64, 128])    
    params = {
        "hidden_layer_sizes": (100, 50),
        "batch_size": batch_size_val,
        "momentum": momentum_val,
        "solver": "sgd",
        "learning_rate": "adaptive", 
        "early_stopping": True,     
        "validation_fraction": 0.1,         
        "random_state": 0,
    }
    model = MLPClassifier(**params, max_iter=300)
    model.fit(low_train_df, y_low_train_flat)
    return model.score(low_test_df, y_low_test_flat)

def objective_MLPC_high_mom(trial): 
    momentum_val = trial.suggest_float("momentum", 0.1, 0.9)
    batch_size_val = trial.suggest_categorical("batch_size", [32, 64, 128])    
    params = {
        "hidden_layer_sizes": (100, 50),
        "batch_size": batch_size_val,
        "momentum": momentum_val,
        "solver": "sgd",
        "learning_rate": "adaptive", 
        "early_stopping": True,     
        "validation_fraction": 0.1,         
        "random_state": 0,
    }
    model = MLPClassifier(**params, max_iter=300)
    model.fit(high_train_df, y_high_train_flat)
    return model.score(high_test_df, y_high_test_flat)

def objective_MLPC_full(trial): 
    momentum_val = trial.suggest_float("momentum", 0.1, 0.9)
    batch_size_val = trial.suggest_categorical("batch_size", [32, 64, 128])    
    params = {
        "hidden_layer_sizes": (100, 50),
        "batch_size": batch_size_val,
        "momentum": momentum_val,
        "solver": "sgd",
        "learning_rate": "adaptive", 
        "early_stopping": True,     
        "validation_fraction": 0.1,         
        "random_state": 0,
    }
    model = MLPClassifier(**params, max_iter=300)
    model.fit(full_train_df, y_full_train_flat)
    return model.score(full_test_df, y_full_test_flat)


regions = [
    ("Low", objective_MLPC_low_mom, low_train_df, y_low_train_flat, low_test_df, y_low_test_flat),
    ("High", objective_MLPC_high_mom, high_train_df, y_high_train_flat, high_test_df, y_high_test_flat),
    ("Full", objective_MLPC_full, full_train_df, y_full_train_flat, full_test_df, y_full_test_flat)
]

for name, obj_func, train_x, train_y, test_x, test_y in regions:
    print(f"\n{'='*20}\nOptimizing {name} Momentum\n{'='*20}")
    
    # Initial Performance 
    init_model = MLPClassifier(hidden_layer_sizes=(100, 50), solver='sgd', random_state=1, max_iter=300)
    init_model.fit(train_x, train_y)
    init_acc = init_model.score(test_x, test_y)
    
    # Study
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(multivariate=True))
    
    # Run optimization
    with parallel_backend("threading", n_jobs=-1):
        study.optimize(obj_func, n_trials=24)

    # Final Training 
   
    #best_model = MLPClassifier(**study.best_params, random_state=0)
    best_model.fit(train_x, train_y)
    best_model = MLPClassifier(
        **study.best_params,
        hidden_layer_sizes=(100, 50),
        solver="sgd",
        learning_rate="adaptive",
        max_iter=500,        
        random_state=0
    )
    best_model.fit(train_x, train_y)
    
    
    joblib.dump(best_model, models_dir / f"best_trained_MLPC_{name.lower()}.joblib")
    
    final_acc = best_model.score(test_x, test_y)
    print(f"{name} Initial Accuracy: {init_acc:.4f}")
    print(f"{name} Final Accuracy: {final_acc:.4f}")

    # Save results to text
    with open(models_dir / f"opt_MLPC_results_{name.lower()}.txt", "w") as f:
        f.write(f"Region: {name}\nBest Params: {study.best_params}\nInitial Acc: {init_acc}\nFinal Acc: {final_acc}")

print("\nAll regions optimized and expert models saved.")

#%% Plotting confusion matrix of best models 
best_model = joblib.load(best_model, models_dir / f"best_trained_MLPC_{name.lower()}.joblib")

full_train_df = pd.read_csv(export_dir / "full_train_df.csv")
full_train_trg = pd.read_csv(export_dir / "full_train_trg.csv")
full_test_df = pd.read_csv(export_dir / "full_test_df.csv")
full_test_trg = pd.read_csv(export_dir / "full_test_trg.csv")

particles = list(full_test_trg["particle"].unique())

# Creating confusion matrix using sklearn.metrics.confusion_matrix
c_matrix_train =confusion_matrix(full_train_trg, model_full.predict(full_train_df), labels = particles, normalize='true')
c_matrix_test = confusion_matrix(full_test_trg, model_full.predict(full_test_df), labels = particles, normalize='true')

acc_train = best_model.score(full_train_df, full_train_trg)
acc_test = best_model.score(full_test_df, full_test_trg)

# Plotting confusion matrix for testing and training data
plt.ion()   
fig, ax = plt.subplots(1,2,figsize=(24,6))

sklearn.metrics.ConfusionMatrixDisplay(c_matrix_train).plot(cmap="Blues", ax=ax[0], colorbar=False)
ax[0].set_xticklabels(particles, rotation=45)
ax[0].set_yticklabels(particles, rotation=45)
ax[0].set_title(f"Training Accuracy: {acc_train:.4f}")

sklearn.metrics.ConfusionMatrixDisplay(c_matrix_test).plot(cmap="Blues", ax=ax[0], colorbar=False)
ax[0].set_xticklabels(particles, rotation=45)
ax[0].set_yticklabels(particles, rotation=45)
ax[0].set_title(f"Testing Accuracy: {acc_test:.4f}")

plt.show()
