#!/bin/bash

python -m flanders.main --multirun server.num_rounds=50 dataset=mnist strategy=flanders aggregate_fn=fedavg,trimmedmean,fedmedian,krum,bulyan server.pool_size=100 server.num_malicious=0,20,60,80 server.attack_fn=gaussian,lie,fang,minmax server.warmup_rounds=2 client_resources.num_cpus=0.1 client_resources.num_gpus=0.1

python -m flanders.main --multirun server.num_rounds=50 dataset=mnist strategy=fedavg,trimmedmean,fedmedian,krum,bulyan server.pool_size=100 server.num_malicious=0,20,60,80 server.attack_fn=gaussian,lie,fang,minmax server.warmup_rounds=2 client_resources.num_cpus=0.1 client_resources.num_gpus=0.1

python -m flanders.main --multirun server.num_rounds=50 dataset=fmnist strategy=flanders aggregate_fn=fedavg,trimmedmean,fedmedian,krum,bulyan server.pool_size=100 server.num_malicious=0,20,60,80 server.attack_fn=gaussian,lie,fang,minmax server.warmup_rounds=2 client_resources.num_cpus=0.1 client_resources.num_gpus=0.1

python -m flanders.main --multirun server.num_rounds=50 dataset=fmnist strategy=fedavg,trimmedmean,fedmedian,krum,bulyan server.pool_size=100 server.num_malicious=0,20,60,80 server.attack_fn=gaussian,lie,fang,minmax server.warmup_rounds=2 client_resources.num_cpus=0.1 client_resources.num_gpus=0.1