#!/bin/bash

python -m fedht.main --config-name base_mnist agg=fedavg num_keep=500 num_local_epochs=1 learning_rate=0.00001
python -m fedht.main --config-name base_mnist agg=fedht num_keep=500 num_local_epochs=1 learning_rate=0.00001
python -m fedht.main --config-name base_mnist agg=fedht iterht=True num_keep=500 num_local_epochs=1 learning_rate=0.00001

python -m fedht.main --config-name base_mnist agg=fedavg num_keep=500 num_local_epochs=5 learning_rate=0.00001
python -m fedht.main --config-name base_mnist agg=fedht num_keep=500 num_local_epochs=5 learning_rate=0.00001
python -m fedht.main --config-name base_mnist agg=fedht iterht=True num_keep=500 num_local_epochs=5 learning_rate=0.00001

python -m fedht.main --config-name base_mnist agg=fedavg num_keep=500 num_local_epochs=1 learning_rate=0.00001 dataset.seed=1
python -m fedht.main --config-name base_mnist agg=fedavg num_keep=500 num_local_epochs=1 learning_rate=0.00001 dataset.seed=2
python -m fedht.main --config-name base_mnist agg=fedavg num_keep=500 num_local_epochs=1 learning_rate=0.00001 dataset.seed=3
python -m fedht.main --config-name base_mnist agg=fedavg num_keep=500 num_local_epochs=1 learning_rate=0.00001 dataset.seed=4
python -m fedht.main --config-name base_mnist agg=fedavg num_keep=500 num_local_epochs=1 learning_rate=0.00001 dataset.seed=5

python -m fedht.main --config-name base_mnist agg=fedht num_keep=700 num_local_epochs=1 learning_rate=0.00001 dataset.seed=1
python -m fedht.main --config-name base_mnist agg=fedht num_keep=700 num_local_epochs=1 learning_rate=0.00001 dataset.seed=2
python -m fedht.main --config-name base_mnist agg=fedht num_keep=700 num_local_epochs=1 learning_rate=0.00001 dataset.seed=3
python -m fedht.main --config-name base_mnist agg=fedht num_keep=700 num_local_epochs=1 learning_rate=0.00001 dataset.seed=4
python -m fedht.main --config-name base_mnist agg=fedht num_keep=700 num_local_epochs=1 learning_rate=0.00001 dataset.seed=5

python -m fedht.main --config-name base_mnist agg=fedht num_keep=500 num_local_epochs=1 learning_rate=0.00001 dataset.seed=1
python -m fedht.main --config-name base_mnist agg=fedht num_keep=500 num_local_epochs=1 learning_rate=0.00001 dataset.seed=2
python -m fedht.main --config-name base_mnist agg=fedht num_keep=500 num_local_epochs=1 learning_rate=0.00001 dataset.seed=3
python -m fedht.main --config-name base_mnist agg=fedht num_keep=500 num_local_epochs=1 learning_rate=0.00001 dataset.seed=4
python -m fedht.main --config-name base_mnist agg=fedht num_keep=500 num_local_epochs=1 learning_rate=0.00001 dataset.seed=5

python -m fedht.main --config-name base_mnist agg=fedht num_keep=250 num_local_epochs=1 learning_rate=0.00001 dataset.seed=1
python -m fedht.main --config-name base_mnist agg=fedht num_keep=250 num_local_epochs=1 learning_rate=0.00001 dataset.seed=2
python -m fedht.main --config-name base_mnist agg=fedht num_keep=250 num_local_epochs=1 learning_rate=0.00001 dataset.seed=3
python -m fedht.main --config-name base_mnist agg=fedht num_keep=250 num_local_epochs=1 learning_rate=0.00001 dataset.seed=4
python -m fedht.main --config-name base_mnist agg=fedht num_keep=250 num_local_epochs=1 learning_rate=0.00001 dataset.seed=5

python -m fedht.main --config-name base_mnist agg=fedht num_keep=100 num_local_epochs=1 learning_rate=0.00001 dataset.seed=1
python -m fedht.main --config-name base_mnist agg=fedht num_keep=100 num_local_epochs=1 learning_rate=0.00001 dataset.seed=2
python -m fedht.main --config-name base_mnist agg=fedht num_keep=100 num_local_epochs=1 learning_rate=0.00001 dataset.seed=3
python -m fedht.main --config-name base_mnist agg=fedht num_keep=100 num_local_epochs=1 learning_rate=0.00001 dataset.seed=4
python -m fedht.main --config-name base_mnist agg=fedht num_keep=100 num_local_epochs=1 learning_rate=0.00001 dataset.seed=5

python -m fedht.main --config-name base_mnist agg=fedht num_keep=50 num_local_epochs=1 learning_rate=0.00001 dataset.seed=1
python -m fedht.main --config-name base_mnist agg=fedht num_keep=50 num_local_epochs=1 learning_rate=0.00001 dataset.seed=2
python -m fedht.main --config-name base_mnist agg=fedht num_keep=50 num_local_epochs=1 learning_rate=0.00001 dataset.seed=3
python -m fedht.main --config-name base_mnist agg=fedht num_keep=50 num_local_epochs=1 learning_rate=0.00001 dataset.seed=4
python -m fedht.main --config-name base_mnist agg=fedht num_keep=50 num_local_epochs=1 learning_rate=0.00001 dataset.seed=5

python -m fedht.main --config-name base_mnist agg=fedht num_keep=25 num_local_epochs=1 learning_rate=0.00001 dataset.seed=1
python -m fedht.main --config-name base_mnist agg=fedht num_keep=25 num_local_epochs=1 learning_rate=0.00001 dataset.seed=2
python -m fedht.main --config-name base_mnist agg=fedht num_keep=25 num_local_epochs=1 learning_rate=0.00001 dataset.seed=3
python -m fedht.main --config-name base_mnist agg=fedht num_keep=25 num_local_epochs=1 learning_rate=0.00001 dataset.seed=4
python -m fedht.main --config-name base_mnist agg=fedht num_keep=25 num_local_epochs=1 learning_rate=0.00001 dataset.seed=5

python -m fedht.main --config-name base_mnist agg=fedht num_keep=25 num_local_epochs=1 learning_rate=0.00001 dataset.seed=1
python -m fedht.main --config-name base_mnist agg=fedht num_keep=25 num_local_epochs=1 learning_rate=0.00001 dataset.seed=2
python -m fedht.main --config-name base_mnist agg=fedht num_keep=25 num_local_epochs=1 learning_rate=0.00001 dataset.seed=3
python -m fedht.main --config-name base_mnist agg=fedht num_keep=25 num_local_epochs=1 learning_rate=0.00001 dataset.seed=4
python -m fedht.main --config-name base_mnist agg=fedht num_keep=25 num_local_epochs=1 learning_rate=0.00001 dataset.seed=5









