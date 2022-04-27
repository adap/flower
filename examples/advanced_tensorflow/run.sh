#!/bin/bash

./certificates/generate.sh

echo "Starting server"

python server.py &
sleep 3  # Sleep for 3s to give the server enough time to start

# Ensure that the Keras dataset used in client.py is already cached.
python -c "import tensorflow as tf; tf.keras.datasets.cifar10.load_data()"

for i in `seq 0 9`; do
    echo "Starting client $i"
    python client.py --partition=${i} &
done

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
