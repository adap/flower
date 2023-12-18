#!/bin/bash

RUN_LOG_DIR=${RUN_LOG_DIR:-"exp_logs"}

pushd ../
mkdir -p $RUN_LOG_DIR
for seed in 123 124 125; do
    echo "Running seed $seed"

    echo "Running without KD ..."
    poetry run python -m fjord.main ++manual_seed=$seed |& tee $RUN_LOG_DIR/wout_kd_$seed.log

    echo "Running with KD ..."
    poetry run python -m fjord.main +train_mode=fjord_kd ++manual_seed=$seed  |& tee $RUN_LOG_DIR/w_kd_$seed.log
done
popd
