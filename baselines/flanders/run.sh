#!/bin/bash

python -m flanders.main --multirun server.num_rounds=50 dataset=mnist,cifar strategy=flanders aggregate_fn=fedavg,trimmedmean,fedmedian,bulyan,dnc server.pool_size=100 server.num_malicious=2 server.attack_fn=gaussian,lie,fang,minmax server.warmup_rounds=2 &> flanders.log
