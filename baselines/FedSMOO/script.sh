export HYDRA_CONFIG_PATH=conf/fedavg.yaml
HYDRA_FULL_ERROR=1 python -m FedSMOO.main --config-name=fedavg ++num_rounds=800
