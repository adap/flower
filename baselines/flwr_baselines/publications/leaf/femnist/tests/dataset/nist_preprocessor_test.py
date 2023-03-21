import unittest

from hamcrest import assert_that, equal_to

from femnist.dataset.nist_preprocessor import NISTPreprocessor
from tests.utils import recreate_nist


class NistPreprocessorTest(unittest.TestCase):
    def test_preprocessing(self):
        """Test if the number of created images examples match expected."""
        temp_dir = recreate_nist()
        print(temp_dir.name)
        nist_preprocessor = NISTPreprocessor(temp_dir.name)
        nist_preprocessor.preprocess()
        preprocessed_df = nist_preprocessor._preprocessed_df
        assert_that(preprocessed_df.shape, equal_to((13, 3)))
        temp_dir.cleanup()


if __name__ == "__main__":
    unittest.main()
