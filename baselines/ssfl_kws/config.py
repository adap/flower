import yaml

global cfg
if 'cfg' not in globals():
    # cfg = {}
    with open('config.yaml', 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

def update_cfg(c):
    global cfg
    cfg = c