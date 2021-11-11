import torch

import logging

def run():
    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)
#    torch.multiprocessing.freeze_support()
    import opacus_building_image_classifier

if __name__ == '__main__':
    run()
