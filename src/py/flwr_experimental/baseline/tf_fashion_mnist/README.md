# Fashion-MNIST Baselines

## Prepare

To execute the `run.py` script you need to create a `.flower_ops` file in the
git root of this project. The file needs to contain the following fields:

```
[paths]
wheel_dir = ~/development/adap/flower/dist/
wheel_filename = flwr-0.0.1-py3-none-any.whl

[aws]
image_id = ami-0370b0294d7241341
key_name = AWS_KEY_NAME
subnet_id = YOUR_AWS_SUBNET_ID
security_group_ids = YOUR_AWS_SECURITY_GROUP_ID
logserver_s3_bucket = YOUR_S3_BUCKET

[ssh]
private_key = PATH_TO_YOU_PRIVATE_KEY_TO_SSH_INTO_THE_MACHINES
```

### Remarks

#### Wheel directory

Adjust the wheel directory according to the localation of the repo on your
machine.

#### Security Group

The security group needs to have port 8080 open so that the clients can connect
to the server.

#### Subnet Id

We are starting all instances in the same subnet to be more cost efficent
(traffic between EC2 instances in the same subnet over their private IP does
not incure any cost).

#### AMI

The provided AMI is a bare Ubuntu 18.04 image which was modified using the
`dev/aws-ami-bootstrap.sh` script.

## Build Docker Container

```bash
./docker/build.sh
```

## Build Python Wheel

To execute the latest version of your baselines during development, please
ensure that the `.whl` build in `dist/` reflects your changes. Re-build
if necessary:

```bash
./dev/build.sh
```

## Execute

To execute a baseline setting locally using docker:

```bash
python -m flwr_experimental.baseline.tf_fashion_mnist.run --adapter="docker" --setting="minimal"
```

To execute a baseline setting remotely on AWS:

```bash
python -m flwr_experimental.baseline.tf_fashion_mnist.run --adapter="ec2" --setting="minimal"
```

Or alternatively, customize the wrapper script `run.sh` and run it using your AWS profile:

```bash
AWS_PROFILE=your-aws-profile src/py/flwr_experimental/baseline/run.sh
```

## Get Results

See all current and past results on the S3 website of your S3 bucket:

```
http://[your-flower-log-s3-bucket].s3-website.eu-central-1.amazonaws.com/
```

Download and filter invididual logs using `cURL` and `jq`:

```bash
curl http://[your-flower-log-s3-bucket].s3-eu-central-1.amazonaws.com/[your-experiment].log | jq '.identifier + " => " + .message'
```
