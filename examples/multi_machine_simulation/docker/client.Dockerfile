FROM ubuntu:20.04

RUN apt-get update && apt-get install -y python3 python3-pip git curl

# Install flower and dependencies for machine learning
RUN python3 --version
RUN pip3 install numpy==1.19.5 tensorflow-cpu==2.4.1 flwr==0.15.0

# Cache the CIFAR-10 dataset which we will use later
RUN python3 -c "import tensorflow as tf; tf.keras.datasets.cifar10.load_data()"

WORKDIR /opt/prep

COPY dataset_generator.py .

RUN python3 dataset_generator.py

# Here we start the actual image and will copy the part of the dataset which we need
FROM ubuntu:20.04


# Dependencies
RUN apt-get update && apt-get install -y python3 python3-pip

# Install flower and dependencies for machine learning
RUN python3 --version
RUN pip3 install numpy==1.19.5 tensorflow-cpu==2.4.1 flwr==0.15.0

WORKDIR /opt/simulation
    
# Copy dataset
ARG index

COPY --from=0 /opt/prep/partitions/x_train_${index}.npy ./partitions/x_train.npy
COPY --from=0 /opt/prep/partitions/y_train_${index}.npy ./partitions/y_train.npy
COPY --from=0 /opt/prep/partitions/x_test_${index}.npy ./partitions/x_test.npy
COPY --from=0 /opt/prep/partitions/y_test_${index}.npy ./partitions/y_test.npy

# Copy the client code
COPY ./client.py ./client.py

# Start the client
ENTRYPOINT ["python3", "client.py"]

