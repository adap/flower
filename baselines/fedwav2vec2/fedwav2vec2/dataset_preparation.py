"""Handle the dataset partitioning and (optionally) complex downloads.

Please add here all the necessary logic to either download, uncompress, pre/post-process
your dataset (or all of the above). If the desired way of running your baseline is to
first download the dataset and partition it and then run the experiments, please
uncomment the lines below and tell us in the README.md (see the "Running the Experiment"
block) that this file should be executed first.
"""


import os
import ssl
import tarfile
import urllib.request
from shutil import rmtree

import hydra
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf


def _download_file(url, filename):
    """Download the file and show a progress bar."""
    print(f"Downloading {url}...")
    retries = 3
    while retries > 0:
        try:
            with urllib.request.urlopen(
                url,
                context=ssl._create_unverified_context(),  # pylint: disable=protected-access
            ) as response, open(filename, "wb") as out_file:
                total_size = int(response.getheader("Content-Length"))
                block_size = 1024 * 8
                count = 0
                while True:
                    data = response.read(block_size)
                    if not data:
                        break
                    count += 1
                    out_file.write(data)
                    percent = int(count * block_size * 100 / total_size)
                    print(
                        f"\rDownloading: {percent}% [{count * block_size}/{total_size}]",
                        end="",
                    )
                print("\nDownload complete.")
                break
        except Exception as error:  # pylint: disable=broad-except
            print(f"\nError occurred during download: {error}")
            retries -= 1
            if retries > 0:
                print(f"Retrying ({retries} retries left)...")
            else:
                print("Download failed.")
                raise error


def _extract_file(filename, extract_path):
    """Extract the contents and show a progress bar."""
    print(f"Extracting {filename}...")
    with tarfile.open(filename, "r:gz") as tar:
        members = tar.getmembers()
        total_files = len(members)
        current_file = 0
        for member in members:
            current_file += 1
            tar.extract(member, path=extract_path)
            percent = int(current_file * 100 / total_files)
            print(f"\rExtracting: {percent}% [{current_file}/{total_files}]", end="")
        print("\nExtraction complete.")


def _delete_file(filename):
    """Delete the downloaded file."""
    os.remove(filename)
    print(f"Deleted {filename}.")


def _csv_path_audio(data_path_base: str, extract_path: str):
    """Change the path corespond to your actual path."""
    # df_concat_train = None
    # df_concat_dev = None
    # df_concat_test = None
    # for i in range(1984):
    #     data_path = f"{data_path_base}/client_{i}/"
    #     if i == 1983:
    #         data_path = f"{data_path_base}/server/"

    #     df_train = pd.read_csv(f"{data_path}/ted_train.csv")
    #     df_dev = pd.read_csv(f"{data_path}/ted_dev.csv")
    #     df_test = pd.read_csv(f"{data_path}/ted_test.csv")
    
    #     df_train["wav"] = df_train["wav"].str.replace(
    #         "path", f"{extract_path}/TEDLIUM_release-3/legacy/train/sph/"
    #     )
    #     df_dev["wav"] = df_dev["wav"].str.replace(
    #         "path", f"{extract_path}/TEDLIUM_release-3/legacy/train/sph/"
    #     )
    #     df_test["wav"] = df_test["wav"].str.replace(
    #         "path", f"{extract_path}/TEDLIUM_release-3/legacy/train/sph/"
    #     )
    #     df_train.to_csv(f"{data_path}/ted_train.csv", index=False)
    #     df_dev.to_csv(f"{data_path}/ted_dev.csv", index=False)
    #     df_test.to_csv(f"{data_path}/ted_test.csv", index=False)



    #     if df_concat_train is None:
    #         df_concat_train = df_train
    #     else:
    #         df_concat_train = pd.concat([df_concat_train, df_train], ignore_index=True)
    #     if df_concat_dev is None:
    #         df_concat_dev = df_dev
    #     else:
    #         df_concat_dev = pd.concat([df_concat_dev, df_dev], ignore_index=True)
    #     if df_concat_test is None:
    #         df_concat_test = df_test
    #     else:
    #         df_concat_test = pd.concat([df_concat_test, df_test], ignore_index=True)

    # df_concat_train.to_csv(f"{data_path}/ted_train.csv", index=False)
    # df_concat_dev.to_csv(f"{data_path}/ted_dev.csv", index=False)
    # df_concat_test.to_csv(f"{data_path}/ted_test.csv", index=False)
    for subdir, dirs, files in os.walk("./data"):
        for file in files:
            if file.endswith('.csv'):
                if "train" in file:
                    path = os.path.join(extract_path, "legacy/train/sph")
                elif "dev" in file:
                    path = os.path.join(extract_path, "legacy/dev/sph")
                else:
                    path = os.path.join(extract_path, "legacy/test/sph")
                df = pd.read_csv(os.path.join(subdir,file))
                df["wav"] = df["wav"].str.replace("path", path)
                df.to_csv(os.path.join(subdir,file), index = False)


@hydra.main(config_path="./conf", config_name="base", version_base=None)
def download_and_extract(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    # URL = (
    #     "https://projets-lium.univ-lemans.fr"
    #     "/wp-content/uploads/corpus/TED-LIUM/TEDLIUM_release-3.tgz"
    # )
    URL = "https://www.openslr.org/resources/51/TEDLIUM_release-3.tgz"
    URL = "https://projets-lium.univ-lemans.fr/wp-content/uploads/corpus/TED-LIUM/TEDLIUM_release-3.tgz"
    FILENAME = f"{cfg.data_path}/{cfg.dataset.download_filename}"
    EXTRACT_PATH = f"{cfg.data_path}/{cfg.dataset.extract_subdirectory}"

    print(f"{EXTRACT_PATH = }")
    print(f"{FILENAME = }")

    # if not os.path.exists(EXTRACT_PATH):
    #     try:
    #         _download_file(URL, FILENAME)
    #         _extract_file(FILENAME, EXTRACT_PATH)
    #     finally:
    #         _delete_file(FILENAME)

    _csv_path_audio(cfg.data_path, EXTRACT_PATH)

    # remove output dir. No need to keep it around
    save_path = HydraConfig.get().runtime.output_dir
    rmtree(save_path)


if __name__ == "__main__":
    download_and_extract()
