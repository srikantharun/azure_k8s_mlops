FROM tensorflow/tensorflow:2.13.0

WORKDIR /app

# Install additional dependencies
RUN pip install --no-cache-dir \
    numpy \
    pandas \
    scikit-learn \
    matplotlib \
    seaborn

# Copy evaluation script
COPY evaluate.py /app/

# Run the evaluation script
ENTRYPOINT ["python", "/app/evaluate.py"]
