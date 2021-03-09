import toml
from os import PathLike
from typing import Any, Dict

def read_config(filename:Union[str, bytes, PathLike], upper_config:Dict[Any]={}):
    """Hierarchically reads toml files to generate conficurations.
       This performs a hierarchical depth-first inclusion of multiple files.
       
       Example: If fileA contains "import =['fileB', 'fileC']" and fileB contains 
       "import = ['fileD', 'fileE']", then import order will be:
       first fileD followed by fileE, fileB, fileC, and finally fileA.
       Files imported last override earlier ones. 

    Args:
        filename (Union[str, bytes, PathLike]): Full path to toml configuration file.

    Returns:
        Dict[Any]: Dictionary 
    """
    this_config:Dict[Any] = toml.load(filename)
    higher_level:Dict[Any]  = {}
    if "import" in this_config:
        for import_file in this_config["import"]:
            # For each element in the import list
            temp_dict = read_config(import_file)
            higher_level = {**higher_level, **temp_dict}

    # where there is a conflict, this_config replaces higher_dict
    config = {**higher_level, **this_config}

    return config
