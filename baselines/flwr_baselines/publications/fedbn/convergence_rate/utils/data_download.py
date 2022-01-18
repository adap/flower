import os
import zipfile
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm  # type: ignore


def download_file_from_google_drive(id: Any, destination: str) -> None:
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={"id": id}, stream=True)
    token = get_confirm_token(response)
    print("This is token", token)

    if token:
        params = {"id": id, "confirm": token}
        print("ID", params["id"])
        response = session.get(URL, params=params, stream=True)
        total_length = response.headers.get("content-length") 
    print("Downloading...")
    save_response_content(response, destination, total_length) # type:ignore
    print("Dowload done")


def get_confirm_token(response: Any) -> Any:
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value
    return None # No error!


def save_response_content(response: Any, destination: str, total_length: float) -> None:
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        total_length = int(total_length)
        for chunk in tqdm(
            response.iter_content(CHUNK_SIZE), total=int(total_length / CHUNK_SIZE)
        ):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


if __name__ == "__main__":
    Path("./data").mkdir(exist_ok=True)
    file_id = "1P8g7uHyVxQJPcBKE8TAzfdKbimpRbj0I"
    destination = "data/data.zip"
    download_file_from_google_drive(file_id, destination)
    print("Extracting...")
    with zipfile.ZipFile(destination, "r") as zip_ref:
        for file in tqdm(iterable=zip_ref.namelist(), total=len(zip_ref.namelist())):
            zip_ref.extract(member=file, path=os.path.dirname(destination))
