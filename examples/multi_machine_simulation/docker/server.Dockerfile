FROM ubuntu:20.04

# Dependencies
RUN apt-get update && apt-get install -y python3 python3-pip

# Install flower and dependencies for machine learning
RUN python3 --version
RUN pip3 install flwr==0.15.0 numpy==1.19.5 tensorflow-cpu==2.4.1

WORKDIR /opt/simulation

# Copy the server code
COPY ./server.py ./server.py

# Start the server
CMD python3 server.py

