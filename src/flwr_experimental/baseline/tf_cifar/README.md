# CIFAR-10/100

## Ops
To execute the `run_aws.py` script you will have to create a `.flower_ops` file in the
git root of this project. The file needs to contain the following fields

```
[paths]
wheel_dir = ~/development/adap/flower/dist/
wheel_filename = WHEEL_FILENAME

[aws]
image_id = ami-0370b0294d7241341
key_name = AWS_KEY_NAME
subnet_id = YOUR_AWS_SUBNET_ID
security_group_ids = YOUR_AWS_SECURITY_GROUP_ID

[ssh]
private_key = PATH_TO_YOU_PRIVATE_KEY_TO_SSH_INTO_THE_MACHINES
```

### Remarks

#### Wheel directory
Adjust the wheel directory according to the localation of the repo on your machine.

#### Security Group
The security group needs to have port 8080 open so that the clients can connect to the server.

#### Subnet Id
We are starting all instances in the same subnet to be more cost efficent (traffic between EC2
instances in the same subnet over their private IP does not incure any cost).

#### AMI
The provided AMI is a bare Ubuntu 18.04 image which was modified with the
`dev/aws-ami-bootstrap.sh` script.

### Execution
To execute the script simply do:
```bash
python -m flwr_experimental.baseline.tf_cifar.run_aws
```
