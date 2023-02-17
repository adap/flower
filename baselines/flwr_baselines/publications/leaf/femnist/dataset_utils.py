import hashlib
import pathlib

import pandas as pd


def hex_decimal_to_char(char_hex: str) -> str:
    return chr(int("0x" + char_hex, 16))


def calculate_file_hash(path: pathlib.Path) -> str:
    with open(path, "rb") as file:
        file_read = file.read()
        hash_val = hashlib.md5(file_read).hexdigest()
    return hash_val


def calculate_series_hashes(series: pd.Series) -> pd.Series:
    series_hashes = series.map(calculate_file_hash)
    return series_hashes

