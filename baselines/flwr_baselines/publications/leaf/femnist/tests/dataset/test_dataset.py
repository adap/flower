import unittest

from femnist.dataset.dataset import (
    create_dataset,
    create_partition_list,
    partition_dataset,
    train_valid_test_partition,
)
from femnist.dataset.nist_preprocessor import NISTPreprocessor
from femnist.dataset.nist_sampler import NistSampler
from hamcrest import assert_that, contains_string, equal_to, is_
from sklearn import preprocessing
from tests.utils import recreate_nist


class TestDataset(unittest.TestCase):
    def test_partitioning(self):
        temp_dir = recreate_nist()
        nist_preprocessor = NISTPreprocessor(temp_dir.name)
        nist_preprocessor.preprocess()
        preprocessed_df = nist_preprocessor._preprocessed_df
        nist_sampler = NistSampler(preprocessed_df)
        sampled_df = nist_sampler.sample(sampling_type="niid", frac=1.0, random_seed=42)
        label_encoder = preprocessing.LabelEncoder()
        labels = label_encoder.fit_transform(sampled_df["character"])
        full_dataset = create_dataset(sampled_df, labels)
        division_list = create_partition_list(sampled_df)
        partitioned_datasets = partition_dataset(full_dataset, division_list)
        assert_that(
            len(partitioned_datasets),
            is_(equal_to(sampled_df.writer_id.unique().shape[0])),
        )
        temp_dir.cleanup()

    def test_train_valid_test_div(self):
        """Test division of the already partitioned dataset into train test
        valid."""
        temp_dir = recreate_nist()
        nist_preprocessor = NISTPreprocessor(temp_dir.name)
        nist_preprocessor.preprocess()
        preprocessed_df = nist_preprocessor._preprocessed_df
        nist_sampler = NistSampler(preprocessed_df)
        sampled_df = nist_sampler.sample(sampling_type="niid", frac=1.0, random_seed=42)
        label_encoder = preprocessing.LabelEncoder()
        labels = label_encoder.fit_transform(sampled_df["character"])
        full_dataset = create_dataset(sampled_df, labels)
        division_list = create_partition_list(sampled_df)
        partitioned_datasets = partition_dataset(full_dataset, division_list)
        train_fraction = 0.5
        validation_fraction = 0.0
        test_fraction = 0.5
        (
            partitioned_train,
            partitioned_validation,
            partitioned_test,
        ) = train_valid_test_partition(
            partitioned_datasets,
            random_seed=42,
            train_split=train_fraction,
            validation_split=validation_fraction,
            test_split=test_fraction,
        )
        assert_that(
            len(partitioned_train[0]),
            is_(equal_to(int(len(division_list[0]) * train_fraction))),
        )
        temp_dir.cleanup()


if __name__ == "__main__":
    unittest.main()
