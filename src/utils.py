import os
import sys
# Add project root to path to resolve src imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pickle
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    """
    Save an object to a file using pickle
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        
        logging.info(f"Object saved successfully at: {file_path}")

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    """
    Evaluate multiple models with hyperparameter tuning
    """
    try:
        report = {}

        for model_name, model in models.items():
            model_params = param.get(model_name, {})
            
            # Hyperparameter tuning if parameters are provided
            if model_params:
                logging.info(f"Tuning hyperparameters for {model_name}")
                gs = GridSearchCV(
                    model, 
                    model_params, 
                    cv=3, 
                    n_jobs=-1, 
                    verbose=0,
                    scoring='r2'
                )
                gs.fit(X_train, y_train)
                model.set_params(**gs.best_params_)
                logging.info(f"Best parameters for {model_name}: {gs.best_params_}")
            
            # Train the model
            logging.info(f"Training {model_name}")
            model.fit(X_train, y_train)
            
            # Make predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Calculate R² scores
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            
            logging.info(f"{model_name} - Train R²: {train_model_score:.4f}, Test R²: {test_model_score:.4f}")
            
            # Store test score in report
            report[model_name] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    """
    Load an object from a file using pickle
    """
    try:
        with open(file_path, "rb") as file_obj:
            obj = pickle.load(file_obj)
        logging.info(f"Object loaded successfully from: {file_path}")
        return obj
    except Exception as e:
        raise CustomException(e, sys)

if __name__ == "__main__":
    print("Utils module loaded successfully!")