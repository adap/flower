"""NIST sampler test."""

# pylint: disable=no-self-use, protected-access
import unittest

from hamcrest import assert_that, contains_string, equal_to, is_

from flwr_baselines.publications.leaf.femnist.dataset.dataset_test import recreate_nist
from flwr_baselines.publications.leaf.femnist.dataset.nist_preprocessor import (
    NISTPreprocessor,
)
from flwr_baselines.publications.leaf.femnist.dataset.nist_sampler import NistSampler


class TestNistSampler(unittest.TestCase):
    """Test NIST sampler."""

    def test_niid_sampling_information(self):
        """Checks if the sampled dataframe has a string denoting the
        location."""
        temp_dir = recreate_nist()
        nist_preprocessor = NISTPreprocessor(temp_dir.name)
        nist_preprocessor.preprocess()
        preprocessed_df = nist_preprocessor._preprocessed_df
        nist_sampler = NistSampler(preprocessed_df)
        sampled_df = nist_sampler.sample(sampling_type="niid", frac=1.0, random_seed=42)
        assert_that(str(sampled_df.path.iloc[0]), contains_string("processed_FeMNIST"))
        temp_dir.cleanup()

    def test_full_niid_size(self):
        """Tests if the fully sampled data is the same as processed data."""
        temp_dir = recreate_nist()
        nist_preprocessor = NISTPreprocessor(temp_dir.name)
        nist_preprocessor.preprocess()
        preprocessed_df = nist_preprocessor._preprocessed_df
        nist_sampler = NistSampler(preprocessed_df)
        sampled_df = nist_sampler.sample(sampling_type="niid", frac=1.0, random_seed=42)
        assert_that(sampled_df.shape[0], is_(equal_to(preprocessed_df.shape[0])))
        temp_dir.cleanup()

    def test_fraction_niid_sampling_size(self):
        """Tests if the fraction sampled data is correct."""
        temp_dir = recreate_nist()
        nist_preprocessor = NISTPreprocessor(temp_dir.name)
        nist_preprocessor.preprocess()
        preprocessed_df = nist_preprocessor._preprocessed_df
        nist_sampler = NistSampler(preprocessed_df)
        # total sample size is 13
        # 0.9 * 13 is 11.7 and ceil(11.7) is 12  (sampling works with ceil)
        sampled_df = nist_sampler.sample(sampling_type="niid", frac=0.9, random_seed=42)
        assert_that(sampled_df.shape[0] + 1, is_(equal_to(preprocessed_df.shape[0])))
        temp_dir.cleanup()


if __name__ == "__main__":
    unittest.main()
