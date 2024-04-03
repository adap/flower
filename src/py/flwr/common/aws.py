
from mypy_boto3_s3.service_resource import Bucket 
from dataclasses import dataclass


@dataclass
class BucketManager:
    bucket: Bucket
