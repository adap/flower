export TF_FORCE_GPU_ALLOW_GROWTH="1"
export TF_CPP_MIN_LOG_LEVEL="3"
flwr run . 
flwr run . --run-config conf/fedavg.toml 