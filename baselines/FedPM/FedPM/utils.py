"""Define any utility function.

They are not directly relevant to  the other (more FL specific) python modules. For
example, you may define here things like: loading a model from a checkpoint, saving
results, plotting.
"""

from compressors.qsgd import QSGDCompressor
from compressors.compressor import Compressor

compressor_dict = {'qsgd': QSGDCompressor}


def get_compressor(
        compressor_type: str,
        **kwargs
) -> Compressor:
    return compressor_dict.get(compressor_type.lower())(**kwargs)
