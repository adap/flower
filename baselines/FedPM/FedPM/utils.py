"""Define any utility function.

They are not directly relevant to  the other (more FL specific) python modules. For
example, you may define here things like: loading a model from a checkpoint, saving
results, plotting.
"""

from compressors.qsgd import QSGDCompressor
from compressors.sign_sgd import SignSGDCompressor

compressor_dict = {'qsgd': QSGDCompressor,
                   'sign_sgd': SignSGDCompressor}


def get_compressor(
        compressor_type: str,
        **kwargs
):
    return compressor_dict.get(compressor_type.lower())(**kwargs)
