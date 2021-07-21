# Change base Docker image if, for instance, a CUDA support is required.
FROM python:3.7.9
LABEL org.opencontainers.image.authors="Taner Topal <taner@adap.com>"

# Update you project dependencies in the ./requirements.txt file.
COPY requirements.txt /requirements.txt
RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir -r /requirements.txt

RUN python3 -c "import tensorflow as tf; tf.keras.datasets.cifar10.load_data()"

# A docker image must define an entry point. Rename `entry_point.py` to your main python file. Adjust if project
# contains multiple files.
COPY src /workspace/src
ENTRYPOINT ["python3", "/workspace/src/run.py"]
