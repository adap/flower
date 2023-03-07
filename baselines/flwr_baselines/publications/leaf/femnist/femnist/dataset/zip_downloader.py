import os
import pathlib
import shutil
from logging import INFO
from typing import Optional

import wget
from flwr.common.logger import log


class ZipDownloader:
    """Zip downloader that enable also unzip and remove the downloaded file."""

    def __init__(
        self, name: str, url: str, save_path: Optional[pathlib.Path] = None
    ) -> None:
        self._name = name
        self._url = url
        self._save_path = (
            save_path if save_path is not None else pathlib.Path(f"./{name}" + ".zip")
        )

    def download(self, unzip: bool = True) -> None:
        """Download file from url only if it does not exist.

        Parameters
        ----------
        unzip - whether to unzip the downloaded filed
        """
        if (
            self._save_path.with_suffix("").exists()
            and len(list(self._save_path.with_suffix("").glob("*"))) != 0
        ):
            log(
                INFO,
                f"Files for {self._name} are already downloaded and extracted from the zip file",
            )
            return None
        self._create_dir_structure()
        if self._save_path.exists():
            log(
                INFO,
                f"Zip file for {self._name} are already downloaded. Skip downloading.",
            )
        else:
            wget.download(self._url, out=str(self._save_path))
        if unzip:
            self._unzip()

    def _create_dir_structure(self):
        self._save_path.parent.mkdir(parents=True, exist_ok=True)

    def _unzip(self):
        log(INFO, f"Unzipping of {self._save_path} to {self._save_path.parent} started")
        shutil.unpack_archive(self._save_path, self._save_path.parent)
        log(INFO, f"Unzipping of {self._save_path} done")
        log(INFO, f"Removing zip file started: {self._save_path}")
        os.remove(self._save_path)
        log(INFO, f"Removing zip file done: {self._save_path}")


if __name__ == "__main__":
    nist_by_class_url = "https://s3.amazonaws.com/nist-srd/SD19/by_class.zip"
    nist_by_writer_url = "https://s3.amazonaws.com/nist-srd/SD19/by_write.zip"
    nist_by_class_downloader = ZipDownloader("data/raw", nist_by_class_url)
    nist_by_writer_downloader = ZipDownloader("data/raw", nist_by_writer_url)
    nist_by_class_downloader.download()
    nist_by_writer_downloader.download()
