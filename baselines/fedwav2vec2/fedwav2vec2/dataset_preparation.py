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
                # pylint: disable=protected-access
                context=ssl._create_unverified_context(),
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
                        f"\rDownload: {percent}% [{count * block_size}/{total_size}]",
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


def _csv_path_audio(extract_path: str):
    """Change the path corespond to your actual path."""
    for subdir, _dirs, files in os.walk("./data"):
        for file in files:
            if file.endswith(".csv"):
                if "client" in subdir:
                    path = path = os.path.join(extract_path, "legacy/train/sph")
                else:
                    if "train" in file:
                        path = os.path.join(extract_path, "legacy/train/sph")
                    elif "dev" in file:
                        path = os.path.join(extract_path, "legacy/dev/sph")
                    else:
                        path = os.path.join(extract_path, "legacy/test/sph")
                d_f = pd.read_csv(os.path.join(subdir, file))
                d_f["wav"] = d_f["wav"].str.replace("path", path)
                d_f.to_csv(os.path.join(subdir, file), index=False)


@hydra.main(config_path="./conf", config_name="base", version_base=None)
def download_and_extract(cfg: DictConfig) -> None:
    """Download and extract TEDIUM-3 dataset."""
    print(OmegaConf.to_yaml(cfg))
    url = (
        "https://projets-lium.univ-lemans.fr"
        "/wp-content/uploads/corpus/TED-LIUM/TEDLIUM_release-3.tgz"
    )
    # URL = "https://www.openslr.org/resources/51/TEDLIUM_release-3.tgz"
    filename = f"{cfg.data_path}/{cfg.dataset.download_filename}"
    extract_path = f"{cfg.data_path}/{cfg.dataset.extract_subdirectory}"

    print(f"{extract_path = }")
    print(f"{filename = }")

    if not os.path.exists(extract_path):
        try:
            _download_file(url, filename)
            _extract_file(filename, extract_path)
        finally:
            _delete_file(filename)

    _csv_path_audio(f"{extract_path}/TEDLIUM_release-3")

    # remove output dir. No need to keep it around
    save_path = HydraConfig.get().runtime.output_dir
    rmtree(save_path)


if __name__ == "__main__":
    download_and_extract()
