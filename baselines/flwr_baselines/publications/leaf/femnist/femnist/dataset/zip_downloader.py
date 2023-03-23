"""Module for downloading the ZIP files, extracting them and removing the
downloaded zip file."""
import os
import pathlib
import shutil
from logging import INFO
from typing import Optional

import wget
from flwr.common.logger import log


# pylint: disable=too-few-public-methods
class ZipDownloader:
    """Zip downloader that enable also unzip and remove the downloaded file."""

    def __init__(
        self, extract_path: str, url: str, save_path: Optional[pathlib.Path] = None
    ) -> None:
        self._extract_path = extract_path
        self._url = url
        self._save_path = (
            save_path
            if save_path is not None
            else pathlib.Path(f"{extract_path}" + ".zip")
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
                "Files from %s are already downloaded and extracted from the zip file "
                "into %s. Skipping downloading and extracting.",
                str(self._url),
                str(self._extract_path),
            )
            return
        self._create_dir_structure()
        if self._save_path.exists():
            log(
                INFO,
                "Zip file under %s already exists. Skipping downloading.",
                str(self._save_path),
            )
        else:
            wget.download(self._url, out=str(self._save_path))
        if unzip:
            self._unzip()

    def _create_dir_structure(self):
        self._save_path.parent.mkdir(parents=True, exist_ok=True)

    def _unzip(self):
        log(
            INFO,
            "Unzipping of %s to %s started",
            str(self._save_path),
            str(self._save_path.parent),
        )
        shutil.unpack_archive(self._save_path, self._save_path.parent)
        log(INFO, "Unzipping of %s done", str(self._save_path))
        log(INFO, "Removing zip file started: %s", str(self._save_path))
        os.remove(self._save_path)
        log(INFO, "Removing zip file done: %s", str(self._save_path))
