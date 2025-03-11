#!/bin/bash
# Script to set up Azure Storage for the ML pipeline

# Variables - Replace with your values
RESOURCE_GROUP="my-ml-resources"
LOCATION="eastus"
STORAGE_ACCOUNT_NAME="mlstorage$(date +%s)"
INPUT_CONTAINER_NAME="datasets"
OUTPUT_CONTAINER_NAME="models"

# Create resource group if it doesn't exist
echo "Creating resource group $RESOURCE_GROUP..."
az group create --name $RESOURCE_GROUP --location $LOCATION

# Create storage account
echo "Creating storage account $STORAGE_ACCOUNT_NAME..."
az storage account create \
    --name $STORAGE_ACCOUNT_NAME \
    --resource-group $RESOURCE_GROUP \
    --location $LOCATION \
    --sku Standard_LRS \
    --kind StorageV2 \
    --enable-hierarchical-namespace false

# Get storage account key
STORAGE_KEY=$(az storage account keys list --resource-group $RESOURCE_GROUP --account-name $STORAGE_ACCOUNT_NAME --query "[0].value" -o tsv)
echo "Storage account key: $STORAGE_KEY"

# Create containers
echo "Creating input container $INPUT_CONTAINER_NAME..."
az storage container create \
    --name $INPUT_CONTAINER_NAME \
    --account-name $STORAGE_ACCOUNT_NAME \
    --account-key $STORAGE_KEY

echo "Creating output container $OUTPUT_CONTAINER_NAME..."
az storage container create \
    --name $OUTPUT_CONTAINER_NAME \
    --account-name $STORAGE_ACCOUNT_NAME \
    --account-key $STORAGE_KEY

# Create connection string
CONNECTION_STRING=$(az storage account show-connection-string \
    --name $STORAGE_ACCOUNT_NAME \
    --resource-group $RESOURCE_GROUP \
    --query connectionString \
    --output tsv)

echo "Storage Account:   $STORAGE_ACCOUNT_NAME"
echo "Input Container:   $INPUT_CONTAINER_NAME"
echo "Output Container:  $OUTPUT_CONTAINER_NAME"
echo "Connection String: $CONNECTION_STRING"

# Create K8s secret
echo "Creating secret for Kubernetes..."
cat > azure-storage-credentials.yaml << EOF
apiVersion: v1
kind: Secret
metadata:
  name: azure-storage-credentials
  namespace: default
type: Opaque
stringData:
  accountKey: "$STORAGE_KEY"
  connectionString: "$CONNECTION_STRING"
EOF

echo "Credentials saved to azure-storage-credentials.yaml"
echo "Apply with: kubectl apply -f azure-storage-credentials.yaml"
