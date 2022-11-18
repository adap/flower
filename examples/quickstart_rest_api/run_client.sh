# Launch two Dummy Flower REST Clients 
for i in `seq 0 1`; do
    echo "Starting client $i"
    python client.py &
done