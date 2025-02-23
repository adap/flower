import boto3
import pytest
import numpy as np
from moto import mock_s3
from flwr_serverless.shared_folder.s3_folder import S3FolderWithPickle


@mock_s3
def test_simple_s3_get():
    conn = boto3.resource("s3", region_name="us-east-1")
    # We need to create the bucket since this is all in Moto's 'virtual' AWS account
    conn.create_bucket(Bucket="mybucket")
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.put_object(Bucket="mybucket", Key="test_object", Body=b"some content")
    body = s3.get_object(Bucket="mybucket", Key="test_object")["Body"].read()
    assert body == b"some content"


@mock_s3
def test_s3_storage_backend():
    conn = boto3.resource("s3", region_name="us-east-1")
    # We need to create the bucket since this is all in Moto's 'virtual' AWS account
    conn.create_bucket(Bucket="mybucket")
    storage = S3FolderWithPickle(directory="mybucket/experiment1")
    with pytest.raises(ValueError):
        storage["test"] = None
    storage["model_1"] = [0, 1, 2]
    assert storage["model_1"] == [0, 1, 2]

    storage["model_1"] = [0, 1, 2, 3]
    assert storage["model_1"] == [0, 1, 2, 3]

    storage["model_2"] = np.array([0, 1, 5])
    assert np.array_equal(storage["model_2"], np.array([0, 1, 5]))

    keys = []
    for key, _ in storage.items():
        keys.append(key)
    assert keys == ["model_1", "model_2"]
