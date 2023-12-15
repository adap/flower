#!/usr/bin/bash
rounds_per_run=20
total_rounds=1000
iterations=$total_rounds/$rounds_per_run

for (( i=1; i <= iterations; ++i ))
do
    echo "$i"
    python -m fedmlb.main num_rounds=$rounds_per_run
done
