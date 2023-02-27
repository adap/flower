import os
import pathlib
import shutil
from typing import Union

import wget


class ZipDownloader:
    """Download and unzip a file."""

    def __init__(self, name: str, url: str, save_path: Union[str, pathlib.Path] = None) -> None:
        self._name = name
        self._url = url
        self._save_path = save_path if save_path is not None else pathlib.Path(f"./{name}" + ".zip")

    def download(self, unzip: bool = True) -> None:
        if self._save_path.with_suffix("").exists() and len(list(self._save_path.with_suffix("").glob("*"))) != 0:
            print("Files are already downloaded and extracted from the zip file")
            return None
        self._create_dir_structure()
        if self._save_path.exists():
            print("Zip file already downloaded. Skipping downloading.")
        else:
            wget.download(self._url, out=str(self._save_path))
        if unzip:
            self._unzip()

    def _create_dir_structure(self):
        self._save_path.parent.mkdir(parents=True, exist_ok=True)

    def _unzip(self):
        print(f"Unzipping of {self._save_path} started")
        shutil.unpack_archive(self._save_path, self._save_path.with_suffix(""))
        print(f"Unzipping of {self._save_path} done")
        print("Removing zip file started")
        os.remove(self._save_path)
        print("Removing zip file done")


if __name__ == "__main__":
    nist_by_class_url = "https://s3.amazonaws.com/nist-srd/SD19/by_class.zip"
    nist_by_writer_url = "https://s3.amazonaws.com/nist-srd/SD19/by_write.zip"
    nist_by_class_downloader = ZipDownloader("data/raw", nist_by_class_url)
    nist_by_writer_downloader = ZipDownloader("data/raw", nist_by_writer_url)
    nist_by_class_downloader.download()
    nist_by_writer_downloader.download()
