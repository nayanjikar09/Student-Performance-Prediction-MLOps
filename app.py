import os
import sys
import pandas as pd
import numpy as np
from flask import Flask, request, render_template, jsonify
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.logger import logging
from src.exception import CustomException
import subprocess
import importlib

# Add project root to path
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Check if model and preprocessor exist
def check_and_train_model():
    """Check if model and preprocessor exist, if not, run the training pipeline"""
    model_path = os.path.join('artifacts', 'model.pkl')
    preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
    
    if not os.path.exists(model_path) or not os.path.exists(preprocessor_path):
        logging.info("Model or preprocessor not found. Running training pipeline...")
        print("="*60)
        print("🔄 TRAINING PIPELINE INITIATED")
        print("="*60)
        
        try:
            # Import and run the data ingestion pipeline
            from src.components.data_ingestion import DataIngestion
            from src.components.data_transformation import DataTransformation
            from src.components.model_trainer import ModelTrainer
            
            # Step 1: Data Ingestion
            logging.info("Step 1: Data Ingestion")
            print("\n📊 Step 1: Data Ingestion")
            obj = DataIngestion()
            train_data, test_data = obj.initiate_data_ingestion()
            print("✅ Data Ingestion completed!")
            
            # Step 2: Data Transformation
            logging.info("Step 2: Data Transformation")
            print("\n🔧 Step 2: Data Transformation")
            data_transformation = DataTransformation()
            train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)
            print("✅ Data Transformation completed!")
            
            # Step 3: Model Training
            logging.info("Step 3: Model Training")
            print("\n🤖 Step 3: Model Training")
            modeltrainer = ModelTrainer()
            r2_score = modeltrainer.initiate_model_trainer(train_arr, test_arr)
            print(f"✅ Model Training completed! R² Score: {r2_score:.4f}")
            
            print("\n" + "="*60)
            print("✅ TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*60)
            
            return True
            
        except Exception as e:
            logging.error(f"Training pipeline failed: {str(e)}")
            print(f"\n❌ Training pipeline failed: {str(e)}")
            return False
    else:
        logging.info("Model and preprocessor already exist.")
        print("✅ Model and preprocessor already exist. Skipping training...")
        return True

# Initialize Flask app
application = Flask(__name__)
app = application

# Check and train model before starting the server
with app.app_context():
    if not check_and_train_model():
        print("\n⚠️ WARNING: Could not train model. Please check the errors above.")
        print("The application may not work properly without trained models.")

@app.route('/')
def index():
    """Home page with input form"""
    try:
        return render_template('index.html')
    except:
        return render_template('home.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    """Handle prediction requests"""
    if request.method == 'GET':
        try:
            return render_template('home.html')
        except:
            return render_template('index.html')
    else:
        try:
            # Get data from form
            data = CustomData(
                gender=request.form.get('gender'),
                race_ethnicity=request.form.get('ethnicity'),
                parental_level_of_education=request.form.get('parental_level_of_education'),
                lunch=request.form.get('lunch'),
                test_preparation_course=request.form.get('test_preparation_course'),
                reading_score=int(request.form.get('reading_score')),
                writing_score=int(request.form.get('writing_score'))
            )
            
            # Convert to dataframe
            pred_df = data.get_data_as_data_frame()
            logging.info(f"Prediction input data: {pred_df}")
            
            # Make prediction
            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)
            
            # Return results
            try:
                return render_template('home.html', results=round(results[0], 2))
            except:
                return render_template('index.html', results=round(results[0], 2))
            
        except Exception as e:
            logging.error(f"Error in prediction: {str(e)}")
            try:
                return render_template('home.html', error="An error occurred during prediction. Please try again.")
            except:
                return render_template('index.html', error="An error occurred during prediction. Please try again.")

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    try:
        # Get JSON data
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['gender', 'race_ethnicity', 'parental_level_of_education', 
                          'lunch', 'test_preparation_course', 'reading_score', 'writing_score']
        
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400
        
        # Create CustomData object
        custom_data = CustomData(
            gender=data['gender'],
            race_ethnicity=data['race_ethnicity'],
            parental_level_of_education=data['parental_level_of_education'],
            lunch=data['lunch'],
            test_preparation_course=data['test_preparation_course'],
            reading_score=int(data['reading_score']),
            writing_score=int(data['writing_score'])
        )
        
        # Convert to dataframe
        pred_df = custom_data.get_data_as_data_frame()
        
        # Make prediction
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        
        # Return JSON response
        return jsonify({
            'prediction': round(results[0], 2),
            'status': 'success'
        })
        
    except Exception as e:
        logging.error(f"API error: {str(e)}")
        return jsonify({'error': str(e), 'status': 'failed'}), 500

@app.route('/retrain', methods=['POST'])
def retrain_model():
    """API endpoint to retrain the model"""
    try:
        logging.info("Manual retraining initiated")
        print("\n🔄 Manual retraining initiated...")
        
        from src.components.data_ingestion import DataIngestion
        from src.components.data_transformation import DataTransformation
        from src.components.model_trainer import ModelTrainer
        
        # Step 1: Data Ingestion
        obj = DataIngestion()
        train_data, test_data = obj.initiate_data_ingestion()
        
        # Step 2: Data Transformation
        data_transformation = DataTransformation()
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)
        
        # Step 3: Model Training
        modeltrainer = ModelTrainer()
        r2_score = modeltrainer.initiate_model_trainer(train_arr, test_arr)
        
        return jsonify({
            'status': 'success',
            'message': 'Model retrained successfully',
            'r2_score': round(r2_score, 4)
        })
        
    except Exception as e:
        logging.error(f"Retraining failed: {str(e)}")
        return jsonify({
            'status': 'failed',
            'error': str(e)
        }), 500

@app.route('/status', methods=['GET'])
def check_status():
    """Check if model and preprocessor exist"""
    model_path = os.path.join('artifacts', 'model.pkl')
    preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
    
    model_exists = os.path.exists(model_path)
    preprocessor_exists = os.path.exists(preprocessor_path)
    
    return jsonify({
        'model_exists': model_exists,
        'preprocessor_exists': preprocessor_exists,
        'status': 'ready' if model_exists and preprocessor_exists else 'needs_training'
    })

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🎓 STUDENT PERFORMANCE PREDICTION SYSTEM")
    print("="*60)
    print("\n🚀 Starting Flask server...")
    print("📊 Web UI: http://localhost:5000")
    print("📡 API Endpoint: http://localhost:5000/api/predict")
    print("🔄 Retrain API: http://localhost:5000/retrain")
    print("📈 Status API: http://localhost:5000/status")
    print("\n" + "="*60 + "\n")
    
    app.run(host="0.0.0.0", port=5000, debug=True)