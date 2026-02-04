Generate Demo Data for Deployment
=================================

You can generate demo datasets for deployment using the
``flwr-datasets create`` CLI command. This command splits a dataset into
multiple partitions, which can then be used for testing and demonstration
purposes when running FL deployments.

For example, to generate demo data from the MNIST dataset with five
partitions and store the result in the ``./demo_data`` directory, run the
following command in your terminal:

.. code-block:: bash

   flwr-datasets create ylecun/mnist --num-partitions 5 --out-dir ./demo_data

This command downloads the MNIST dataset, partitions it into five subsets,
and saves each partition to the specified output directory.
