#! /bin/bash

python -m depthfl.main --config-name="heterofl" 
python -m depthfl.main --config-name="heterofl" exclusive_learning=true model_size=1 model.scale=false
python -m depthfl.main --config-name="heterofl" exclusive_learning=true model_size=2 model.scale=false
python -m depthfl.main --config-name="heterofl" exclusive_learning=true model_size=3 model.scale=false
python -m depthfl.main --config-name="heterofl" exclusive_learning=true model_size=4 model.scale=false

python -m depthfl.main fit_config.feddyn=false fit_config.kd=false  
python -m depthfl.main fit_config.feddyn=false fit_config.kd=false  exclusive_learning=true model_size=1
python -m depthfl.main fit_config.feddyn=false fit_config.kd=false  exclusive_learning=true model_size=2
python -m depthfl.main fit_config.feddyn=false fit_config.kd=false  exclusive_learning=true model_size=3
python -m depthfl.main fit_config.feddyn=false fit_config.kd=false  exclusive_learning=true model_size=4

python -m depthfl.main 
python -m depthfl.main exclusive_learning=true model_size=1
python -m depthfl.main exclusive_learning=true model_size=2
python -m depthfl.main exclusive_learning=true model_size=3
python -m depthfl.main exclusive_learning=true model_size=4

python -m depthfl.main fit_config.feddyn=false fit_config.kd=false fit_config.extended=false

python -m depthfl.main fit_config.kd=false