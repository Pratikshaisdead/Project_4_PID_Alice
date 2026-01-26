import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder


base_dir = Path(__file__).resolve().parent
models_dir = base_dir / "models"
export_dir = models_dir / "data"

def run_evaluation():

    print("Initializing Label Encoder...")
    le = LabelEncoder()
    try:
        full_trg_path = export_dir / "full_train_trg.csv"
        full_trg = pd.read_csv(full_trg_path)

        le.fit(full_trg.values.ravel().astype(str))
    except FileNotFoundError:
        print(f"Error: Could not find {full_trg_path}. Check your paths.")
        return

    regions = ["low", "high", "full"]

    for reg in regions:
        print(f"\n{'='*50}")
        print(f"EVALUATING: {reg.upper()} MOMENTUM MODEL")
       # print(f{'='*50}")
        

        model_name = f"final_fixed_MLPC_{reg}.joblib"
        model_path = models_dir / model_name
        
        if not model_path.exists():
            print(f"Skipping: {model_name} not found in {models_dir}")
            continue
            
        model = joblib.load(model_path)

        try:
            X_test = pd.read_csv(export_dir / f"{reg}_test_df.csv")
            y_test_raw = pd.read_csv(export_dir / f"{reg}_test_trg.csv").values.ravel().astype(str)
        except FileNotFoundError as e:
            print(f"Data file missing for {reg}: {e}")
            continue


        y_test = le.transform(y_test_raw)
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        print(f"Overall Test Accuracy: {acc*100:.2f}%")
  
        present_labels = np.unique(y_test)
        present_names = [le.classes_[i] for i in present_labels]
        
        print("\nDetailed Classification Report:")
        report = classification_report(
            y_test, 
            y_pred, 
            labels=present_labels, 
            target_names=present_names, 
            digits=4
        )
        print(report)

e
        output_file = models_dir / f"evaluation_report_{reg}.txt"
        with open(output_file, "w") as f:
            f.write(f"Evaluation for {reg} momentum\n")
            f.write(f"Accuracy: {acc*100:.2f}%\n\n")
            f.write(report)
        print(f"Report saved to: {output_file.name}")

if __name__ == "__main__":
    run_evaluation()
