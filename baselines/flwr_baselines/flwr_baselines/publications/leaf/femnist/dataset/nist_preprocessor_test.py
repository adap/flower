"""NIST preprocessor test."""

# pylint: disable=no-self-use
import unittest

from hamcrest import assert_that, equal_to

from flwr_baselines.publications.leaf.femnist.dataset.dataset_test import recreate_nist
from flwr_baselines.publications.leaf.femnist.dataset.nist_preprocessor import (
    NISTPreprocessor,
)


class NistPreprocessorTest(unittest.TestCase):
    """Test nist processor."""

    def test_preprocessing(self):
        """Test if the number of created images examples match expected."""
        temp_dir = recreate_nist()
        print(temp_dir.name)
        nist_preprocessor = NISTPreprocessor(temp_dir.name)
        nist_preprocessor.preprocess()
        # pylint: disable=protected-access
        preprocessed_df = nist_preprocessor._preprocessed_df
        assert_that(preprocessed_df.shape, equal_to((13, 3)))
        temp_dir.cleanup()


if __name__ == "__main__":
    unittest.main()
