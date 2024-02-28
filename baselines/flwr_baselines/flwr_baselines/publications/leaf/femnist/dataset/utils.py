"""Utilities used for the FEMNIST dataset creation."""

import hashlib
import pathlib

import pandas as pd


def hex_decimal_to_char(char_hex: str) -> str:
    """Convert hexadecimal string to ASCII representation.

    Parameters
    ----------
    char_hex: str
        string (without "0x") representing hexadecimal value

    Returns
    -------
    char: str
        ASCII representation of the hexadecimal value
    """
    return chr(int("0x" + char_hex, 16))


def calculate_file_hash(path: pathlib.Path) -> str:
    """Calculate a hash of a file.

    Parameters
    ----------
    path : pathlib.Path
        path to a file

    Returns
    -------
    hash_value: str
        hash value of the object from the path
    """
    with open(path, "rb") as file:
        file_read = file.read()
        hash_val = hashlib.md5(file_read).hexdigest()
    return hash_val


def calculate_series_hashes(paths: pd.Series) -> pd.Series:
    """Calculate hashes from pd.Series of paths.

    Parameters
    ----------
    paths: pd.Series of pathlib.Path
        paths

    Returns
    -------
    series_hashes: pd.Series of strings
        hash values of the objects from paths
    """
    series_hashes = paths.map(calculate_file_hash)
    return series_hashes


def _create_samples_division_list(n_samples, n_groups, keep_remainder=False):
    """Create ids for clients such that it enables indexing."""
    group_size = n_samples // n_groups
    n_samples_in_full_groups = n_groups * group_size
    samples_division_list = []
    for i in range(n_groups):
        samples_division_list.extend([i] * group_size)
    if keep_remainder:
        if n_samples_in_full_groups != n_samples:
            # add remainder only if it is needed == remainder is not equal zero
            remainder = n_samples - n_samples_in_full_groups
            samples_division_list.extend([n_groups] * remainder)
    return samples_division_list
