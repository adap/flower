"""Handle the dataset partitioning and (optionally) complex downloads.

Please add here all the necessary logic to either download, uncompress, pre/post-process
your dataset (or all of the above). If the desired way of running your baseline is to
first download the dataset and partition it and then run the experiments, please
uncomment the lines below and tell us in the README.md (see the "Running the Experiment"
block) that this file should be executed first.
"""
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
from typing import Optional

import os
import urllib.request
import bz2
import shutil

def _download_data(
        dataset_name: Optional[str]="all"
) -> None:
    """
    Downloads (if necessary) and returns the dataset assigned by the dataset_name pparm.
    Parameters
    ----------
    dataset_name : String
        A string stating the name of the dataset that need to be dowenloaded.
    """
    
    ALL_DATASETS_PATH="./baselines/FL+XGBoost/FL+XGBoost/dataset"
    #CLASSIFICATION_PATH = os.path.join(DATASETS_PATH, "binary_classification")
    #REGRESSION_PATH = os.path.join(DATASETS_PATH, "regression")
    if dataset_name=="a9a" or dataset_name=="all" :
        DATASET_PATH=os.path.join(ALL_DATASETS_PATH, "a9a")
        if not os.path.exists(DATASET_PATH):
            os.makedirs(DATASET_PATH)
            urllib.request.urlretrieve(
                "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a9a",
                f"{os.path.join(DATASET_PATH, 'a9a')}",
            )
            urllib.request.urlretrieve(
                "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a9a.t",
                f"{os.path.join(DATASET_PATH, 'a9a.t')}",
            )
            
    if dataset_name=="cod-rna" or dataset_name=="all" :
        DATASET_PATH=os.path.join(ALL_DATASETS_PATH, "cod-rna")
        if not os.path.exists(DATASET_PATH):
            os.makedirs(DATASET_PATH)
            urllib.request.urlretrieve(
                "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/cod-rna",
                f"{os.path.join(DATASET_PATH, 'cod-rna')}",
            )
            urllib.request.urlretrieve(
                "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/cod-rna.t",
                f"{os.path.join(DATASET_PATH, 'cod-rna.t')}",
            )
            urllib.request.urlretrieve(
                "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/cod-rna.r",
                f"{os.path.join(DATASET_PATH, 'cod-rna.r')}",
            )

    if dataset_name=="ijcnn1" or dataset_name=="all" :
        DATASET_PATH=os.path.join(ALL_DATASETS_PATH, "ijcnn1")
        if not os.path.exists(DATASET_PATH):
            os.makedirs(DATASET_PATH)

            urllib.request.urlretrieve(
                "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/ijcnn1.bz2",
                f"{os.path.join(DATASET_PATH, 'ijcnn1.tr.bz2')}",
            )
            urllib.request.urlretrieve(
                "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/ijcnn1.t.bz2",
                f"{os.path.join(DATASET_PATH, 'ijcnn1.t.bz2')}",
            )
            urllib.request.urlretrieve(
                "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/ijcnn1.tr.bz2",
                f"{os.path.join(DATASET_PATH, 'ijcnn1.tr.bz2')}",
            )
            urllib.request.urlretrieve(
                "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/ijcnn1.val.bz2",
                f"{os.path.join(DATASET_PATH, 'ijcnn1.val.bz2')}",
            )

            for filepath in os.listdir(DATASET_PATH):
                abs_filepath = os.path.join(DATASET_PATH, filepath)
                with bz2.BZ2File(abs_filepath) as fr, open(abs_filepath[:-4], "wb") as fw:
                    shutil.copyfileobj(fr, fw)


    if dataset_name=="space_ga" or dataset_name=="all":
        DATASET_PATH=os.path.join(ALL_DATASETS_PATH, "space_ga")
        if not os.path.exists(DATASET_PATH):
            os.makedirs(DATASET_PATH)
            urllib.request.urlretrieve(
                "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/space_ga",
                f"{os.path.join(DATASET_PATH, 'space_ga')}",
            )
            urllib.request.urlretrieve(
                "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/space_ga_scale",
                f"{os.path.join(DATASET_PATH, 'space_ga_scale')}",
            )

    if dataset_name=="abalone" or dataset_name=="all":
        DATASET_PATH=os.path.join(ALL_DATASETS_PATH, "abalone")
        if not os.path.exists(DATASET_PATH): 
            os.makedirs(DATASET_PATH)       
            urllib.request.urlretrieve(
                "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/abalone",
                f"{os.path.join(DATASET_PATH, 'abalone')}",
            )
            urllib.request.urlretrieve(
                "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/abalone_scale",
                f"{os.path.join(DATASET_PATH, 'abalone_scale')}",
            )
    if dataset_name=="cpusmall" or dataset_name=="all":
        DATASET_PATH=os.path.join(ALL_DATASETS_PATH, "cpusmall")
        if not os.path.exists(DATASET_PATH):  
            os.makedirs(DATASET_PATH)      
            urllib.request.urlretrieve(
                "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/cpusmall",
                f"{os.path.join(DATASET_PATH, 'cpusmall')}",
            )
            urllib.request.urlretrieve(
                "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/cpusmall_scale",
                f"{os.path.join(DATASET_PATH, 'cpusmall_scale')}",
            )