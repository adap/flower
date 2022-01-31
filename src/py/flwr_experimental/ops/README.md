# Flower Ops
## Compute
### EC2 Adapter
For permission management an IAM instance profile named `FlowerInstanceProfile` is expected.
The instances will use that profile for all nessecary permissions. In case of logfile upload
the profile must include the permission to upload the logfile from the machine to the respective
S3 bucket.

An example policy attached to the profile for the logfiles might look like:
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "FlowerInstanceProfileS3Policy",
            "Effect": "Allow",
            "Action": [
                "s3:PutObject",
                "s3:PutObjectRetention",
                "s3:PutObjectVersionAcl",
                "s3:PutObjectAcl"
            ],
            "Resource": "arn:aws:s3:::mylogfilebucket/*"
        }
    ]
}
```
