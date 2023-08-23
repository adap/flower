"""Handle the dataset partitioning and (optionally) complex downloads.

Please add here all the necessary logic to either download, uncompress, pre/post-process
your dataset (or all of the above). If the desired way of running your baseline is to
first download the dataset and partition it and then run the experiments, please
uncomment the lines below and tell us in the README.md (see the "Running the Experiment"
block) that this file should be executed first.
"""
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import call, instantiate
from omegaconf import DictConfig, OmegaConf
from tensorflow import keras


@hydra.main(config_path="conf", config_name="base", version_base=None)
def download_and_preprocess(cfg: DictConfig) -> None:
  """Does everything needed to get the dataset.
    Parameters
    ----------
    cfg : DictConfig
      An omegaconf object that stores the hydra config.
  """

  # print(OmegaConf.to_yaml(cfg))

  # Please include here all the logic
  # Please use the Hydra config style as much as possible specially
  # for parts that can be customised (e.g. how data is partitioned)

  if FEMNIST == True:
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    input_shape = x_train.shape[1:]
    num_classes = len(np.unique(y_train))
  else:
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    input_shape = x_train.shape[1:]
    num_classes = len(np.unique(y_train))

  return x_train, y_train, x_test, y_test, input_shape, num_classes

if __name__ == "__main__":
  download_and_preprocess()
