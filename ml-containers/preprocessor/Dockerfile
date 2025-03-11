FROM python:3.10-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    azure-storage-blob \
    numpy \
    pandas \
    scikit-learn

# Copy preprocessing script
COPY preprocess.py /app/

# Run the preprocessing script
ENTRYPOINT ["python", "/app/preprocess.py"]
