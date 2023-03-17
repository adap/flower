import pathlib
import random
import tempfile
import unittest
from pathlib import Path

from femnist.dataset.nist_preprocessor import NISTPreprocessor
from hamcrest import assert_that, equal_to
from PIL import Image

from ..utils import recreate_nist


class NistPreprocessorTest(unittest.TestCase):
    def test_preprocessing(self):
        """Test if the number of created images examples match expected."""
        temp_dir = recreate_nist()
        print(temp_dir.name)
        print(list(Path(temp_dir.name).glob("*/*/*")))
        nist_preprocessor = NISTPreprocessor(temp_dir.name)
        nist_preprocessor.preprocess()
        print("writer df")
        print(nist_preprocessor._writer_df)
        print("class df")
        print(nist_preprocessor._class_df)
        print("df")
        print(nist_preprocessor._df)
        print("preprocessed df")
        print(nist_preprocessor._preprocessed_df)
        preprocessed_df = nist_preprocessor._preprocessed_df
        assert_that(preprocessed_df.shape, equal_to((13, 3)))
        temp_dir.cleanup()


if __name__ == "__main__":
    unittest.main()
