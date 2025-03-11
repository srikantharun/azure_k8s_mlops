# Azure Blob Kubernetes Operator

A Kubernetes operator that monitors Azure Blob Storage for new datasets and triggers ML pipelines when new data is detected. This operator is particularly useful for ML workflows where new data should automatically trigger training, evaluation, or prediction jobs.

## Features

- Monitor Azure Blob Storage containers for new or updated blobs
- Filter blobs by prefix and/or regular expressions
- Automatically create Kubernetes resources (Jobs, Workflows) when new data is detected
- Configurable polling interval
- Metrics and status reporting
- Support for SAS tokens or account keys for authentication

## Architecture

![Operator Architecture](https://via.placeholder.com/800x400?text=Azure+Blob+Operator+Architecture)

The operator:
1. Watches for AzureBlobWatcher custom resources
2. Polls Azure Blob Storage at specified intervals
3. Detects new or updated blobs
4. Creates Kubernetes resources based on templates
5. Tracks triggered jobs in the AzureBlobWatcher status

## Installation

### Prerequisites

- Kubernetes cluster (v1.19+)
- kubectl
- Helm (optional)

### Install with Helm

```bash
helm repo add azure-blob-operator https://yourusername.github.io/azure-blob-k8s-operator/charts
helm install azure-blob-operator azure-blob-operator/azure-blob-operator
```

### Install with kubectl

```bash
kubectl apply -f https://raw.githubusercontent.com/yourusername/azure-blob-k8s-operator/main/config/install/install.yaml
```

## Usage

### 1. Create Azure Storage Credentials Secret

Create a Kubernetes secret with your Azure Storage credentials:

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: azure-storage-credentials
  namespace: default
type: Opaque
data:
  # Base64 encoded credentials - replace with actual values
  accountKey: <base64-encoded-account-key>
  # Or alternatively use SAS token
  # sasToken: <base64-encoded-sas-token>
```

### 2. Create an AzureBlobWatcher Resource

```yaml
apiVersion: mlops.example.com/v1alpha1
kind: AzureBlobWatcher
metadata:
  name: dataset-watcher
  namespace: default
spec:
  storageAccount: "mystorageaccount"
  containerName: "datasets"
  credentialsSecret: "azure-storage-credentials"
  
  # Optional: Filter blobs
  blobPrefix: "processed/"
  blobPattern: ".*\\.csv$"
  pollingIntervalSeconds: 60
  
  # Job template
  jobTemplate:
    kind: "Job"
    template: |
      apiVersion: batch/v1
      kind: Job
      metadata:
        name: process-${BLOB_NAME}
      spec:
        template:
          spec:
            containers:
            - name: processor
              image: myregistry/data-processor:latest
              env:
              - name: DATASET_URL
                value: "${BLOB_URL}"
              - name: DATASET_NAME
                value: "${BLOB_NAME}"
            restartPolicy: Never
        backoffLimit: 2
```

### 3. Monitor AzureBlobWatcher Status

```bash
kubectl get azureblobwatcher
kubectl describe azureblobwatcher dataset-watcher
```

## Integration with ML Pipelines

The operator can trigger various Kubernetes resources when new data is detected:

### Basic Job Example

For simple processing tasks, create a Job template:

```yaml
jobTemplate:
  kind: "Job"
  template: |
    apiVersion: batch/v1
    kind: Job
    metadata:
      name: process-data-${BLOB_NAME}
    spec:
      template:
        spec:
          containers:
          - name: processor
            image: myregistry/ml-processor:latest
            env:
            - name: INPUT_DATA_URL
              value: "${BLOB_URL}"
          restartPolicy: Never
```

### Argo Workflow Example

For more complex ML pipelines, trigger an Argo Workflow:

```yaml
jobTemplate:
  kind: "Workflow"
  template: |
    apiVersion: argoproj.io/v1alpha1
    kind: Workflow
    metadata:
      name: ml-pipeline-${BLOB_NAME}
    spec:
      entrypoint: ml-pipeline
      templates:
      - name: ml-pipeline
        steps:
        - - name: preprocess
            template: preprocess
        - - name: train
            template: train
        - - name: evaluate
            template: evaluate
      # Define step templates...
```

## Development

### Prerequisites

- Go 1.20+
- Operator SDK v1.25.0+
- Docker
- kubectl
- Access to a Kubernetes cluster (for testing)

### Building

```bash
# Clone repository
git clone https://github.com/yourusername/azure-blob-k8s-operator.git
cd azure-blob-k8s-operator

# Install dependencies
go mod download

# Run tests
make test

# Build operator
make build

# Build and push container image
make docker-build docker-push IMG=yourusername/azure-blob-k8s-operator:latest
```

### Running Locally

```bash
# Run against current kubectl context
make install run
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
