#!/usr/bin/env python3
"""
Evaluation script for ML pipeline.
Loads trained model and test data, evaluates the model, and saves metrics.
"""

import argparse
import os
import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_test_data(test_data_path):
    """Load test data from preprocessed .npz file."""
    try:
        logger.info(f"Loading test data from {test_data_path}")
        data = np.load(test_data_path, allow_pickle=True)
        
        # Check if data includes train/test split
        if 'X_test' in data and 'y_test' in data:
            X_test = data['X_test']
            y_test = data['y_test']
            logger.info(f"Test data loaded. X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
            return X_test, y_test
        else:
            # If no train/test split, use a portion of data for testing
            X = data['X']
            feature_names = data['feature_names'] if 'feature_names' in data else None
            
            # Use 20% for testing
            test_size = int(0.2 * X.shape[0])
            X_test = X[-test_size:]
            
            logger.info(f"No explicit test set found. Using {test_size} samples for testing.")
            logger.info(f"X_test shape: {X_test.shape}")
            
            # Return X_test as both features and targets (for autoencoder evaluation)
            return X_test, X_test
            
    except Exception as e:
        logger.error(f"Error loading test data: {str(e)}")
        raise

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and calculate various metrics."""
    try:
        logger.info("Evaluating model on test data")
        
        # Get predictions
        y_pred_prob = model.predict(X_test)
        
        # Initialize metrics dictionary
        metrics = {
            "test_samples": X_test.shape[0],
            "model_evaluation": {}
        }
        
        # Use model.evaluate for basic metrics
        eval_results = model.evaluate(X_test, y_test, verbose=0)
        metrics["model_evaluation"] = {
            "loss": float(eval_results[0]),
            "accuracy": float(eval_results[1])
        }
        
        # Check shape of predictions to determine task type
        if y_pred_prob.shape[1] > 1:
            # Multi-class classification
            task_type = "multiclass"
            y_pred = np.argmax(y_pred_prob, axis=1)
            
            # Convert one-hot encoded y_test to class labels if needed
            if len(y_test.shape) > 1 and y_test.shape[1] > 1:
                y_test_labels = np.argmax(y_test, axis=1)
            else:
                y_test_labels = y_test
            
            # Calculate metrics
            metrics.update({
                "task_type": task_type,
                "accuracy": float(accuracy_score(y_test_labels, y_pred)),
                "precision_macro": float(precision_score(y_test_labels, y_pred, average='macro')),
                "recall_macro": float(recall_score(y_test_labels, y_pred, average='macro')),
                "f1_macro": float(f1_score(y_test_labels, y_pred, average='macro')),
                "num_classes": int(y_pred_prob.shape[1])
            })
            
            # Generate confusion matrix
            cm = confusion_matrix(y_test_labels, y_pred)
            metrics["confusion_matrix"] = cm.tolist()
            
        elif y_pred_prob.shape[1] == 1:
            # Binary classification
            task_type = "binary"
            y_pred = (y_pred_prob > 0.5).astype(int).flatten()
            
            # Calculate metrics
            metrics.update({
                "task_type": task_type,
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "precision": float(precision_score(y_test, y_pred)),
                "recall": float(recall_score(y_test, y_pred)),
                "f1": float(f1_score(y_test, y_pred)),
                "auc_roc": float(roc_auc_score(y_test, y_pred_prob.flatten()))
            })
            
            # Generate confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            metrics["confusion_matrix"] = cm.tolist()
            
        else:
            # Regression or other task
            task_type = "regression"
            y_pred = y_pred_prob
            
            # Calculate metrics (MSE, MAE, etc.)
            mse = np.mean(np.square(y_test - y_pred))
            mae = np.mean(np.abs(y_test - y_pred))
            
            metrics.update({
                "task_type": task_type,
                "mse": float(mse),
                "mae": float(mae),
                "rmse": float(np.sqrt(mse))
            })
        
        logger.info(f"Evaluation complete. Task type: {task_type}")
        return metrics
    
    except Exception as e:
        logger.error(f"Error evaluating model: {str(e)}")
        raise

def save_metrics(metrics, metrics_path):
    """Save evaluation metrics to a JSON file."""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        logger.info(f"Metrics saved to {metrics_path}")
    
    except Exception as e:
        logger.error(f"Error saving metrics: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Evaluate model for ML pipeline')
    parser.add_argument('--model-path', required=True, help='Path to trained model (.h5 file)')
    parser.add_argument('--test-data', required=True, help='Path to test data (.npz file)')
    parser.add_argument('--metrics-path', required=True, help='Path to save evaluation metrics (.json file)')
    args = parser.parse_args()
    
    try:
        # Load model
        logger.info(f"Loading model from {args.model_path}")
        model = load_model(args.model_path)
        
        # Load test data
        X_test, y_test = load_test_data(args.test_data)
        
        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test)
        
        # Save metrics
        save_metrics(metrics, args.metrics_path)
        
        logger.info("Evaluation process completed successfully")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()
