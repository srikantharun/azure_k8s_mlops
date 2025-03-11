FROM tensorflow/tensorflow:2.13.0

WORKDIR /app

# Install additional dependencies
RUN pip install --no-cache-dir \
    numpy \
    azure-storage-blob \
    tensorflow-model-optimization

# Copy deployment script
COPY deploy.py /app/

# Run the deployment script
ENTRYPOINT ["python", "/app/deploy.py"]
