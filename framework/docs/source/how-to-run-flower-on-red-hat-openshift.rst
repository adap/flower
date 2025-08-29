:og:description: A step-by-step guide to learn how to create, deploy and run a Flower app on Red Hat OpenShift using the Red Hat OpenShift Service on AWS.
.. meta::
    :description: A step-by-step guide to learn how to create, deploy and run a Flower app on Red Hat OpenShift using the Red Hat OpenShift Service on AWS.

Run Flower on Red Hat OpenShift
===============================

In this guide, you will learn how to create, deploy, and run a Flower app on the [Red
Hat OpenShift (RHOS)](https://www.redhat.com/en/technologies/cloud-computing/openshift)
application platform. The platform will be hosted in AWS and we will follow the steps to
install the cluster on installer-provisioned Infrastructure using the [Red Hat OpenShift
Service on AWS](https://aws.amazon.com/rosa/).

AWS Prerequisites
-----------------

Here, we outline the pre-requisites for AWS to create and manage a Red Hat OpenShift
cluster. The instructions are based on the [RHOS getting started guide accessible from
your AWS console](https://console.aws.amazon.com/rosa/home#/get-started).

1. Enable RHOS service on AWS (ROSA) in your AWS account.
2. Ensure that you have service quotas for ROSA.
3. Create a service-linked role for Elastic Load Balancing. This should be automatically
   creatd for you if not present.
4. Link your AWS and Red Hat account.
5. Create AWS Identity and Access Management (IAM) roles. You will need to create an IAM
   user with these required permissions:

   - ``AmazonEC2FullAccess``
   - ``AWSCloudFormationFullAccess``
   - ``IAMFullAccess``
   - ``ServiceQuotasReadOnlyAccess``

For convenience, install the ``aws`` CLI tool for your system. You can alternatively run
it with Docker using the command:

.. code-block:: shell

    docker run --rm -it --volume ~/.aws:/root/.aws public.ecr.aws/aws-cli/aws-cli

The ``--volume ~/.aws:/root/.aws`` option mounts your AWS credentials to the Docker
container. Next, run the following to configure the AWS CLI tool:

.. code-block:: shell

    ➜ aws configure
    AWS Access Key ID [None]: [...]
    AWS Secret Access Key [None]: [...]
    Default region name [None]: [...]  # your region
    Default output format [None]: table  # the recommended output format

Create a Red Hat OpenShift Cluster on AWS
-----------------------------------------

WIP

Deploy Flower SuperLink and SuperNodes on OpenShift
---------------------------------------------------

WIP

Deploy Red Hat OpenShift AI
---------------------------

WIP

Build a custom OpenShift AI Image with Flower
---------------------------------------------

WIP

Run the Custom OpenShift AI Workbench with Flower
-------------------------------------------------

WIP

Run the Flower App in OpenShift AI
----------------------------------

WIP
