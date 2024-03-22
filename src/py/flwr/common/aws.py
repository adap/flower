
from mypy_boto3_s3.service_resource import Bucket 
from .typing import Parameters

import uuid
import json

class BucketManager:

    def __init__(self, bucket: Bucket) -> None:
        self.bucket = bucket
    
    def put_parameters(self, parameters: Parameters) -> uuid.UUID:
        id = uuid.uuid4()
        body = b''.join(parameters.tensors)
        self.bucket.put_object(
            Key=str(id),
            Metadata=dict(
                tensor_type=parameters.tensor_type,
                dimensions=json.dumps(parameters.dimensions)
            ),
            Body=body
        )
        return id
        

    def pull_parameters(self, id: str | uuid.UUID):
        key = str(id)
        
        result = self.bucket.Object(key).get()
        tensor_type = result["Metadata"]["tensor_type"]
        dimensions = json.loads(result["Metadata"]["dimensions"])
        tensor_bytes = result["Body"].read()
        
        return Parameters.parse_bytes(
            tensor_type,
            tensor_bytes,
            dimensions
        )


