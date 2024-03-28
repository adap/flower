# Use an official Python runtime as a parent image
FROM python:3.10-slim-bookworm

# Set the working directory in the container to /app
WORKDIR /app

# Copy the requirements file into the container
COPY ./requirements.txt /app/requirements.txt

# Install gcc and other dependencies
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
    gcc=4:8.3.0-1 \
    python3-dev=3.7.3-1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt