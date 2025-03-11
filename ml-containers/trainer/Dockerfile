FROM tensorflow/tensorflow:2.13.0-gpu

WORKDIR /app

# Install additional dependencies
RUN pip install --no-cache-dir \
    numpy \
    pandas \
    scikit-learn \
    matplotlib \
    h5py \
    tensorflow-model-optimization \
    pyyaml

# Copy training script
COPY train.py /app/

# Run the training script
ENTRYPOINT ["python", "/app/train.py"]
