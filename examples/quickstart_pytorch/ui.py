import streamlit as st
import pandas as pd
import numpy as np
import subprocess
st.title('UiS Federated Learning Platform')

comm_port = "8000"
vis_port = "6014"
st.write("federated learning port:{}, tensorboard port:{}".format(comm_port, vis_port))

server_cmd = "python server_board.py"
vis_cmd = "tensorboard --logdir /mnt/c/Users/janga/projects/flower/examples/quickstart_pytorch/flwr_logs --port ".format(vis_port)
hdl_run_server = subprocess.Popen(server_cmd.split(), stdout=subprocess.PIPE)
print("start serving:{}".format(server_cmd))
hdl_run_board = subprocess.Popen(vis_cmd.split(), stdout=subprocess.PIPE)
print("start visualization{}".format(vis_cmd))
