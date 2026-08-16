import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        self.model_path = os.path.join('artifacts', 'model.pkl')
        self.preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
        
        # Check if model exists
        if not os.path.exists(self.model_path) or not os.path.exists(self.preprocessor_path):
            logging.warning("Model or preprocessor not found. Please train the model first.")
            print("\n⚠️ WARNING: Model not found. Please run the training pipeline first.")
            print("Run: python -m src.components.data_ingestion\n")
        
    def predict(self, features):
        try:
            # Load model and preprocessor
            model = load_object(self.model_path)
            preprocessor = load_object(self.preprocessor_path)
            
            # Transform features
            scaled_data = preprocessor.transform(features)
            
            # Make prediction
            prediction = model.predict(scaled_data)
            
            return prediction
            
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self,
                 gender: str,
                 race_ethnicity: str,
                 parental_level_of_education: str,
                 lunch: str,
                 test_preparation_course: str,
                 reading_score: int,
                 writing_score: int):
        
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)