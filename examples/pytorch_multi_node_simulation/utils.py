from pathlib import Path
from argparse import ArgumentTypeError


def valid_folder(path_str: str) -> bool:
    """Tests if a path is a valid FL partition folder

    Args:
                path_str (str): Path to directory containing train and test folder.

    Returns:
                bool: result of checks
    """
    tmp_path = Path(path_str)
    test = True
    for sub_folder in ["train", "test"]:
        test = test and (tmp_path / sub_folder).exists()
    if not test:
        raise ArgumentTypeError
    return test
