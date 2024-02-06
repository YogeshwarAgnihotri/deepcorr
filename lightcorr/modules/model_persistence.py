import joblib
import os

def save_model(model, run_folder_path):
    # Save model for later evaluation or prediction making
    joblib.dump(model, os.path.join(run_folder_path, 'model.joblib'))