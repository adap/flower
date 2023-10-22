"""This code will download the required datasets from a Google Drive, save it
in the directory ./data/data.zip and extracts it."""


import os
import zipfile
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm  # type: ignore

# pylint: disable=invalid-name


def download_file_from_google_drive(file_id: Any, destination: str) -> None:
    """Download zip from Google drive."""
    url = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(url, params={"id": file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {"id": file_id, "confirm": token}
        print("ID", params["id"])
        response = session.get(url, params=params, stream=True)
        total_length = response.headers.get("content-length")
    print("Downloading...")
    save_response_content(response, destination, total_length)  # type:ignore
    print("Dowload done")


# pylint: enable=invalid-name


def get_confirm_token(response: Any) -> Any:
    """Conform Google cookies."""
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value
    return None


def save_response_content(response: Any, destination: str, total_length: float) -> None:
    """save data in the given data file."""
    chunk_size = 32768

    with open(destination, "wb") as download_file:
        total_length = int(total_length)
        for chunk in tqdm(
            response.iter_content(chunk_size), total=int(total_length / chunk_size)
        ):
            if chunk:  # filter out keep-alive new chunks
                download_file.write(chunk)


if __name__ == "__main__":
    Path("./data").mkdir(exist_ok=True)
    FILE_ID = "1P8g7uHyVxQJPcBKE8TAzfdKbimpRbj0I"
    DESTINATION = "data/data.zip"
    download_file_from_google_drive(FILE_ID, DESTINATION)
    print("Extracting...")
    with zipfile.ZipFile(DESTINATION, "r") as zip_ref:
        for file in tqdm(iterable=zip_ref.namelist(), total=len(zip_ref.namelist())):
            zip_ref.extract(member=file, path=os.path.dirname(DESTINATION))
