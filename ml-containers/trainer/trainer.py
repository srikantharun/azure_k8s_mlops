#!/usr/bin/env python3
"""
Training script for ML pipeline.
Loads preprocessed data, trains a model, and saves it.
"""

import argparse
import os
import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure GPU memory growth to avoid OOM errors
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logger.info(f"GPU memory growth enabled for {len(gpus)} GPUs")
    except RuntimeError as e:
        logger.warning(f"Error setting GPU memory growth: {str(e)}")

def load_data(input_path):
    """Load preprocessed data."""
    try:
        logger.info(f"Loading data from {input_path}")
        data = np.load(input_path, allow_pickle=True)
        
        # Check if data includes train/test split
        if 'X_train' in data and 'y_train' in data:
            X_train = data['X_train']
            X_test = data['X_test']
            y_train = data['y_train']
            y_test = data['y_test']
            feature_names = data['feature_names'].tolist() if 'feature_names' in data else None
            
            logger.info(f"Data loaded. X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
            return X_train, X_test, y_train, y_test, feature_names
        else:
            # If no train/test split, use all data as X
            X = data['X']
            feature_names = data['feature_names'].tolist() if 'feature_names' in data else None
            logger.info(f"Data loaded. X shape: {X.shape}")
            return X, None, None, None, feature_names
            
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def build_model(input_shape, output_shape=1):
    """Build a neural network model."""
    try:
        # Check if output is binary or multi-class
        if output_shape == 1:
            output_activation = 'sigmoid'
            loss = 'binary_crossentropy'
        else:
            output_activation = 'softmax'
            loss = 'sparse_categorical_crossentropy'
        
        model = Sequential([
            Dense(128, activation='relu', input_shape=(input_shape,)),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(output_shape, activation=output_activation)
        ])
        
        model.compile(
            optimizer='adam',
            loss=loss,
            metrics=['accuracy']
        )
        
        logger.info(f"Model built. Input shape: {input_shape}, Output shape: {output_shape}")
        model.summary(print_fn=logger.info)
        return model
    
    except Exception as e:
        logger.error(f"Error building model: {str(e)}")
        raise

def train_model(model, X_train, y_train, X_test, y_test, model_path):
    """Train the model and save it."""
    try:
        # Create directory for model if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Set up callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ModelCheckpoint(
                filepath=model_path,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train the model
        logger.info("Starting model training")
        history = model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=2
        )
        
        # Save the final model
        model.save(model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Return training history for parameter file
        return {
            "epochs": len(history.history['loss']),
            "final_accuracy": float(history.history['accuracy'][-1]),
            "final_val_accuracy": float(history.history['val_accuracy'][-1]),
            "final_loss": float(history.history['loss'][-1]),
            "final_val_loss": float(history.history['val_loss'][-1])
        }
    
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        raise

def save_params(params, params_path):
    """Save training parameters to a JSON file."""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(params_path), exist_ok=True)
        
        with open(params_path, 'w') as f:
            json.dump(params, f, indent=4)
        
        logger.info(f"Parameters saved to {params_path}")
    
    except Exception as e:
        logger.error(f"Error saving parameters: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Train model for ML pipeline')
    parser.add_argument('--input-path', required=True, help='Path to preprocessed data (.npz file)')
    parser.add_argument('--model-path', required=True, help='Path to save trained model (.h5 file)')
    parser.add_argument('--params-path', required=True, help='Path to save training parameters (.json file)')
    args = parser.parse_args()
    
    try:
        # Load data
        X_train, X_test, y_train, y_test, feature_names = load_data(args.input_path)
        
        # Determine output shape based on y_train
        if y_train is not None:
            # Check if y_train is one-hot encoded or just labels
            if len(y_train.shape) > 1:
                output_shape = y_train.shape[1]  # One-hot encoded
            else:
                output_shape = len(np.unique(y_train))  # Just labels
                if output_shape == 2:  # Binary classification
                    output_shape = 1
        else:
            # If no target, assume unsupervised task (e.g., autoencoder)
            output_shape = X_train.shape[1]
            y_train = X_train
            y_test = X_test
        
        # Build model
        model = build_model(X_train.shape[1], output_shape)
        
        # Train model
        training_history = train_model(model, X_train, y_train, X_test, y_test, args.model_path)
        
        # Save parameters
        params = {
            "training": training_history,
            "model": {
                "input_shape": X_train.shape[1],
                "output_shape": output_shape,
                "feature_names": feature_names
            },
            "data": {
                "training_samples": X_train.shape[0],
                "validation_samples": X_test.shape[0] if X_test is not None else 0
            }
        }
        save_params(params, args.params_path)
        
        logger.info("Training process completed successfully")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()
