#!/usr/bin/env python3
"""
Preprocessing script for ML pipeline.
Downloads dataset from Azure Blob, processes it, and saves the result.
"""

import argparse
import os
import numpy as np
import pandas as pd
from azure.storage.blob import BlobServiceClient
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_blob(input_url):
    """Download blob from Azure Storage."""
    try:
        # Extract container and blob name from URL
        parts = input_url.replace('https://', '').split('/')
        storage_account = parts[0].split('.')[0]
        container_name = parts[1]
        blob_name = '/'.join(parts[2:])
        
        # Get connection string from environment
        connection_string = os.environ.get('AZURE_STORAGE_CONNECTION_STRING')
        if not connection_string:
            raise ValueError("AZURE_STORAGE_CONNECTION_STRING environment variable not set")
        
        # Create blob service client
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service_client.get_container_client(container_name)
        blob_client = container_client.get_blob_client(blob_name)
        
        # Download blob
        temp_file = '/tmp/dataset.csv'
        logger.info(f"Downloading blob {blob_name} to {temp_file}")
        with open(temp_file, "wb") as download_file:
            download_file.write(blob_client.download_blob().readall())
        
        return temp_file
    except Exception as e:
        logger.error(f"Error downloading blob: {str(e)}")
        raise

def preprocess_data(input_file, output_path):
    """Preprocess the data and save the result."""
    try:
        logger.info(f"Loading data from {input_file}")
        df = pd.read_csv(input_file)
        
        logger.info(f"Original data shape: {df.shape}")
        
        # Basic preprocessing
        # 1. Handle missing values
        df = df.dropna()
        logger.info(f"Data shape after dropping NA: {df.shape}")
        
        # 2. Convert categorical features if needed
        # Example: One-hot encoding categorical columns
        # categorical_cols = ['category1', 'category2']
        # df = pd.get_dummies(df, columns=categorical_cols)
        
        # 3. Split features and target (adjust column names as needed)
        if 'target' in df.columns:
            X = df.drop('target', axis=1)
            y = df['target']
        else:
            # If no target column, use all data as features
            X = df
            y = None
        
        # 4. Normalize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 5. Split into train/test sets if target exists
        if y is not None:
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
            
            # Save preprocessed data
            logger.info(f"Saving processed data to {output_path}")
            np.savez_compressed(
                output_path,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                feature_names=X.columns.tolist(),
                scaler_mean=scaler.mean_,
                scaler_scale=scaler.scale_
            )
        else:
            # If no target, just save the preprocessed features
            logger.info(f"No target column found. Saving processed features to {output_path}")
            np.savez_compressed(
                output_path,
                X=X_scaled,
                feature_names=X.columns.tolist(),
                scaler_mean=scaler.mean_,
                scaler_scale=scaler.scale_
            )
            
        logger.info("Preprocessing completed successfully")
        
    except Exception as e:
        logger.error(f"Error preprocessing data: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Preprocess dataset for ML pipeline')
    parser.add_argument('--input-url', required=True, help='Azure Blob URL of the input dataset')
    parser.add_argument('--output-path', required=True, help='Path to save processed data')
    args = parser.parse_args()
    
    try:
        # Download the dataset
        input_file = download_blob(args.input_url)
        
        # Create directory for output if it doesn't exist
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        
        # Preprocess the data
        preprocess_data(input_file, args.output_path)
        
        logger.info(f"Preprocessing complete. Results saved to {args.output_path}")
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()
