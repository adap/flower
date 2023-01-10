#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/

# Define Dataset
export IMG_ROOT='/datasets/FedScale/openImg/'
export CID_CSV_ROOT='/datasets/FedScale/openImg/client_data_mapping/clean_ids'

# Define number of workers
export NUM_WORKERS=10

# Experiment timestamp
timestamp=$(date +%Y-%m-%d_%H%M%S)

# Start Server
echo "Starting server"
python server.py --num_workers=${NUM_WORKERS}&

# Collect GPU DATA
nvidia-smi --query-gpu=timestamp,name,index,pci.bus_id,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv --filename=${timestamp}_NVIDIA.csv --loop-ms=200 &

sleep 3  # Sleep for 3s to give the server enough time to start

export MAX_WORKER_IDX=$(($NUM_WORKERS-1))
for i in `seq 0 ${MAX_WORKER_IDX}`; do
    echo "Starting worker $i"
    python worker.py --path_imgs=${IMG_ROOT} --path_csv_map=${CID_CSV_ROOT} &
done


# Enable CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
