#!/usr/bin/env python3
"""
Deployment script for ML pipeline.
Loads trained model, quantizes it to 8-bit, and uploads to Azure Blob Storage.
"""

import argparse
import os
import json
import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow_model_optimization as tfmot
from azure.storage.blob import BlobServiceClient
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def quantize_model(model, quantization_type="int8"):
    """Quantize the model to reduce its size."""
    try:
        logger.info(f"Quantizing model to {quantization_type}")
        
        if quantization_type == "int8":
            # Clone the model first
            clone = tf.keras.models.clone_model(model)
            clone.set_weights(model.get_weights())
            
            # Apply quantization aware training
            quantize_model = tfmot.quantization.keras.quantize_model
            
            # Apply quantization to the model
            q_aware_model = quantize_model(clone)
            
            # Compile the quantized model
            q_aware_model.compile(
                optimizer='adam',
                loss=model.loss,
                metrics=['accuracy']
            )
            
            # Convert the model to TFLite
            converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            quantized_tflite_model = converter.convert()
            
            # Save TFLite model
            tflite_model_path = "/tmp/quantized_model.tflite"
            with open(tflite_model_path, 'wb') as f:
                f.write(quantized_tflite_model)
            
            logger.info(f"Quantized TFLite model saved to {tflite_model_path}")
            
            # Also save quantized Keras model for compatibility
            keras_model_path = "/tmp/quantized_model.h5"
            q_aware_model.save(keras_model_path)
            
            logger.info(f"Quantized Keras model saved to {keras_model_path}")
            
            return tflite_model_path, keras_model_path
        else:
            raise ValueError(f"Unsupported quantization type: {quantization_type}")
            
    except Exception as e:
        logger.error(f"Error quantizing model: {str(e)}")
        raise

def upload_to_blob_storage(file_path, container_name, blob_name):
    """Upload a file to Azure Blob Storage."""
    try:
        # Get connection string from environment
        connection_string = os.environ.get('AZURE_STORAGE_CONNECTION_STRING')
        if not connection_string:
            raise ValueError("AZURE_STORAGE_CONNECTION_STRING environment variable not set")
        
        # Create blob service client
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service_client.get_container_client(container_name)
        
        # Upload file
        with open(file_path, "rb") as data:
            blob_client = container_client.upload_blob(name=blob_name, data=data, overwrite=True)
            
        logger.info(f"Uploaded {file_path} to container {container_name} as {blob_name}")
        
        # Return the blob URL
        account_name = blob_service_client.account_name
        blob_url = f"https://{account_name}.blob.core.windows.net/{container_name}/{blob_name}"
        return blob_url
        
    except Exception as e:
        logger.error(f"Error uploading to blob storage: {str(e)}")
        raise

def upload_model_artifacts(model_path, params_path, metrics_path, output_container, model_name):
    """Upload model and associated artifacts to Azure Blob Storage."""
    try:
        # Load model
        logger.info(f"Loading model from {model_path}")
        model = load_model(model_path)
        
        # Quantize model
        tflite_path, quantized_h5_path = quantize_model(model)
        
        # Create model info dictionary
        upload_info = {
            "original_model": None,
            "quantized_model_h5": None,
            "quantized_model_tflite": None,
            "params": None,
            "metrics": None
        }
        
        # Upload original model
        original_model_blob = f"{model_name}.h5"
        upload_info["original_model"] = upload_to_blob_storage(
            model_path, output_container, original_model_blob
        )
        
        # Upload quantized models
        quantized_h5_blob = f"{model_name}-quantized.h5"
        upload_info["quantized_model_h5"] = upload_to_blob_storage(
            quantized_h5_path, output_container, quantized_h5_blob
        )
        
        tflite_blob = f"{model_name}-quantized.tflite"
        upload_info["quantized_model_tflite"] = upload_to_blob_storage(
            tflite_path, output_container, tflite_blob
        )
        
        # Upload params file
        if params_path and os.path.exists(params_path):
            params_blob = f"{model_name}-params.json"
            upload_info["params"] = upload_to_blob_storage(
                params_path, output_container, params_blob
            )
        
        # Upload metrics file
        if metrics_path and os.path.exists(metrics_path):
            metrics_blob = f"{model_name}-metrics.json"
            upload_info["metrics"] = upload_to_blob_storage(
                metrics_path, output_container, metrics_blob
            )
        
        # Create and upload model info file
        info_path = "/tmp/model-info.json"
        with open(info_path, 'w') as f:
            json.dump(upload_info, f, indent=4)
        
        info_blob = f"{model_name}-info.json"
        upload_to_blob_storage(info_path, output_container, info_blob)
        
        logger.info(f"All model artifacts uploaded successfully to {output_container}")
        return upload_info
        
    except Exception as e:
        logger.error(f"Error uploading model artifacts: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Deploy model for ML pipeline')
    parser.add_argument('--model-path', required=True, help='Path to trained model (.h5 file)')
    parser.add_argument('--params-path', help='Path to training parameters (.json file)')
    parser.add_argument('--metrics-path', help='Path to evaluation metrics (.json file)')
    parser.add_argument('--output-container', required=True, help='Azure Blob container for output')
    parser.add_argument('--quantize', choices=['int8', 'float16'], default='int8', help='Quantization type')
    args = parser.parse_args()
    
    try:
        # Extract model name from path
        model_name = os.path.splitext(os.path.basename(args.model_path))[0]
        
        # Upload model and artifacts
        upload_info = upload_model_artifacts(
            args.model_path,
            args.params_path,
            args.metrics_path,
            args.output_container,
            model_name
        )
        
        logger.info("Deployment process completed successfully")
        logger.info(f"Quantized model available at: {upload_info['quantized_model_h5']}")
        
    except Exception as e:
        logger.error(f"Deployment failed: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()
