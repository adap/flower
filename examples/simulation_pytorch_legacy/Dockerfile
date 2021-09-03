FROM ubuntu:20.04

# Dependencies
RUN apt-get update && apt-get install -y python3.8 python3-pip git curl

# Install flower and dependencies for machine learning
RUN python3 --version
RUN pip3 install flwr==0.15.0 numpy==1.19.5 torch==1.8.1 torchvision==0.9.1

# Copy code in final step so code changes don't invalidate the
# previous docker layers
WORKDIR /opt/simulation_pytorch
COPY . .

# Start the simulation
CMD python3 simulation.py
