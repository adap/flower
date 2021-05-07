#!/bin/bash
set -e

rm -f *.dat *.jpg

python -m mprof run --include-children -o memory_usage_server.dat server.py &
sleep 2
python -m mprof run --include-children -o memory_usage_clients.dat clients.py

python -m mprof plot memory_usage_server.dat -o memory_usage_server.jpg
python -m mprof plot memory_usage_clients.dat -o memory_usage_clients.jpg
