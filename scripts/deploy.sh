#!/bin/bash
# Script to deploy the Azure Blob Kubernetes Operator and Argo Workflows

# Check if kubectl is installed
if ! command -v kubectl &> /dev/null; then
    echo "kubectl is not installed. Please install it first."
    exit 1
fi

# Check if current context is set
CURRENT_CONTEXT=$(kubectl config current-context 2>/dev/null)
if [ $? -ne 0 ]; then
    echo "No Kubernetes context found. Please configure kubectl."
    exit 1
fi

echo "Using Kubernetes context: $CURRENT_CONTEXT"
echo "Starting deployment..."

# Install Argo Workflows CRDs
echo "Installing Argo Workflows..."
kubectl create namespace argo || true
kubectl apply -n argo -f https://github.com/argoproj/argo-workflows/releases/download/v3.4.7/install.yaml

# Wait for Argo controller to be ready
echo "Waiting for Argo Workflows controller to be ready..."
kubectl -n argo wait --for=condition=Available deployment/argo-server --timeout=300s
kubectl -n argo wait --for=condition=Available deployment/workflow-controller --timeout=300s

# Install Azure Blob Watcher CRD
echo "Installing Azure Blob Watcher CRD..."
kubectl apply -f config/crd/bases/mlops.sertel.com_azureblobwatchers.yaml

# Create namespace for operator
kubectl create namespace azure-blob-operator || true

# Deploy Azure credentials secret
echo "Deploying Azure credentials secret..."
if [ -f "azure-storage-credentials.yaml" ]; then
    kubectl apply -f azure-storage-credentials.yaml
else
    echo "Warning: azure-storage-credentials.yaml not found."
    echo "Make sure to create and apply this secret before using the operator."
fi

# Deploy the operator
echo "Deploying Azure Blob Kubernetes Operator..."
kubectl apply -f config/deployment.yaml

# Check status
echo "Checking operator status..."
kubectl -n azure-blob-operator get pods

echo ""
echo "Deployment complete!"
echo ""
echo "To create an AzureBlobWatcher resource:"
echo "  kubectl apply -f examples/azureblobwatcher.yaml"
echo ""
echo "To check triggered workflows:"
echo "  kubectl get workflows"
echo ""
echo "To access Argo Workflows UI:"
echo "  kubectl -n argo port-forward svc/argo-server 2746:2746"
echo "  Then open: https://localhost:2746"
