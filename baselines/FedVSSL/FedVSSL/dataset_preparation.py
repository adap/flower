"""Handle the dataset partitioning and (optionally) complex downloads.

Please add here all the necessary logic to either download, uncompress, pre/post-process
your dataset (or all of the above). If the desired way of running your baseline is to
first download the dataset and partition it and then run the experiments, please
uncomment the lines below and tell us in the README.md (see the "Running the Experiment"
block) that this file should be executed first.
"""

# make sure you have installed unrar package. One can install it using apt install unrar

import subprocess
import CtP
# 
# first download the raw videos from the official websit
# subprocess.run(["mkdir -p data/ucf101/"], shell=True)
# subprocess.run(["wget https://www.crcv.ucf.edu/data/UCF101/UCF101.rar -O data/ucf101/UCF101.rar --no-check-certificate && \
# unrar e data/ucf101/UCF101.rar data/ucf101/UCF101_raw/"], shell=True)

# print("---unzipping the compressed file---")
# subprocess.run(["unrar e data/ucf101/UCF101.rar data/ucf101/UCF101_raw/"], shell=True)

# print("--Downloading the train/test split---")
# subprocess.run(["wget https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip -O \
# data/ucf101/UCF101TrainTestSplits-RecognitionTask.zip --no-check-certificate"], shell=True)

# subprocess.run(["unzip data/ucf101/UCF101TrainTestSplits-RecognitionTask.zip -d data/ucf101/."], shell=True)

print("--Preprocessing the dataset script---")
subprocess.run(["python CtP/scripts/process_ucf101.py --raw_dir data/ucf101/UCF101_raw/ \
--ann_dir data/ucf101/ucfTrainTestlist/ --out_dir data/ucf101/"], shell=True)

# print("---converting the annotation files into json format---")
# subprocess.run(["python CtP/scripts/cvt_txt_to_json.py"], shell=True)

# optional 
# rm data/ucf101/UCF101.rar
# rm -r data/ucf101/UCF101_raw/




# import hydra
# from hydra.core.hydra_config import HydraConfig
# from hydra.utils import call, instantiate
# from omegaconf import DictConfig, OmegaConf


# @hydra.main(config_path="conf", config_name="base", version_base=None)
# def download_and_preprocess(cfg: DictConfig) -> None:
#     """Does everything needed to get the dataset.

#     Parameters
#     ----------
#     cfg : DictConfig
#         An omegaconf object that stores the hydra config.
#     """

#     ## 1. print parsed config
#     print(OmegaConf.to_yaml(cfg))

#     # Please include here all the logic
#     # Please use the Hydra config style as much as possible specially
#     # for parts that can be customised (e.g. how data is partitioned)

# if __name__ == "__main__":

#     download_and_preprocess()
