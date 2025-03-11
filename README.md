# Azure Blob Kubernetes Operator for ML Pipelines

This project provides a Kubernetes operator that monitors Azure Blob Storage for new datasets and triggers ML pipelines using Argo Workflows. It's designed for MLOps workflows where new data automatically triggers training, evaluation, and deployment of machine learning models.

## Architecture Overview

![Operator Architecture](https://via.placeholder.com/800x400?text=Azure+Blob+Operator+Architecture)

The system architecture consists of:

1. **Azure Blob Storage**: Stores input datasets and output models
2. **Azure Blob Kubernetes Operator**: Monitors blob storage for new datasets
3. **Argo Workflows**: Orchestrates the ML pipeline steps
4. **ML Processing Containers**: Docker images for each stage of the ML pipeline

When a new dataset is uploaded to Azure Blob Storage, the operator detects it and creates an Argo Workflow that processes the data through the following steps:

1. **Preprocessing**: Clean and prepare the raw data
2. **Training**: Train the model using the preprocessed data
3. **Evaluation**: Evaluate model performance with test data
4. **Deployment**: Quantize to 8-bit and upload the model to the output container

## Prerequisites

- Kubernetes cluster (v1.19+)
- kubectl
- Azure CLI
- Docker (for building images)
- Go 1.20+ (for development)

## Setup Instructions

### 1. Set up Azure Storage

Run the provided setup script to create Azure Storage resources:

```bash
chmod +x scripts/azure-setup.sh
./scripts/azure-setup.sh
```

This script will:
- Create a resource group
- Create a storage account
- Create input and output containers
- Generate a Kubernetes secret with the credentials

### 2. Install Argo Workflows

Install Argo Workflows to your Kubernetes cluster:

```bash
kubectl create namespace argo
kubectl apply -n argo -f https://github.com/argoproj/argo-workflows/releases/download/v3.4.7/install.yaml
```

### 3. Build and Deploy the Operator

Build the operator image:

```bash
# Update the image name in the Makefile
make docker-build docker-push IMG=yourusername/azure-blob-k8s-operator:latest
```

Deploy the operator:

```bash
kubectl apply -f config/deployment.yaml
```

### 4. Create an AzureBlobWatcher Resource

Apply the example AzureBlobWatcher resource:

```bash
kubectl apply -f examples/azureblobwatcher.yaml
```

## Usage

### Upload a Dataset

Upload a CSV file to your Azure Storage input container:

```bash
az storage blob upload --account-name yourstorageaccount --container-name datasets --name raw/mydata.csv --file ./mydata.csv --auth-mode key --account-key your-account-key-here
```

### Monitor the Workflow

The operator will detect the new file and create an Argo Workflow. You can monitor the workflow using:

```bash
kubectl get workflows
kubectl describe workflow ml-pipeline-raw-mydata.csv
```

Access the Argo Workflows UI:

```bash
kubectl -n argo port-forward svc/argo-server 2746:2746
```

Then navigate to https://localhost:2746 in your browser.

### Check the Output

Once the workflow completes, the quantized model will be uploaded to the output container. You can download it using:

```bash
az storage blob download --account-name yourstorageaccount --container-name models --name model-raw-mydata.csv-quantized.h5 --file ./model-quantized.h5 --auth-mode key --account-key your-account-key-here
```

## Customizing the ML Pipeline

### Container Images

You'll need to create Docker images for each step in the pipeline:

1. **Preprocessor**: Clean and prepare data
2. **Trainer**: Train the model
3. **Evaluator**: Evaluate model performance
4. **Deployer**: Quantize and upload the model

Example Dockerfile for preprocessor:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

RUN pip install azure-storage-blob pandas numpy scikit-learn

COPY preprocess.py /app/

CMD ["python", "preprocess.py"]
```

### Workflow Template

You can customize the Argo Workflow template in the AzureBlobWatcher resource to:
- Add more processing steps
- Change resource requirements
- Use different container images
- Modify the pipeline logic

## Development

### Project Structure

```
├── api/                    # API definitions
├── config/                 # Kubernetes configuration
├── controllers/            # Controller logic
├── examples/               # Example resources
├── scripts/                # Utility scripts
├── main.go                 # Main entry point
├── Dockerfile              # Operator container image
└── Makefile                # Build commands
```

### Building and Testing

Build the operator:

```bash
make build
```

Run tests:

```bash
make test
```

Run the operator locally:

```bash
make run
```

## Troubleshooting

### Common Issues

1. **Secret not found**: Ensure the Azure credentials secret exists in the same namespace as the AzureBlobWatcher resource
2. **Permission errors**: Check RBAC rules and service account permissions
3. **Blob detection issues**: Verify the blobPrefix and blobPattern settings

### Checking Logs

```bash
kubectl logs -n azure-blob-operator deployment/azure-blob-operator
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
