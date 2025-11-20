:og:description: Guide to manage Flower federations using the Deployment Engine.
.. meta::
    :description: Guide to manage Flower federations using the Deployment Engine.


.. note::

    Flower Federation management is a new feature introduced in Flower 1.24.0. It gain new functionality in subsequen releases. Changes to the functionality described in this guide is also expected as the features provided via the ``flwr federation`` commands matuer.


###########################
 Manage Flower Federations
###########################

A Flower federation is comprised by a set of users and some or all the SuperNodes they
own that are registered with the same SuperLink. Members of a Flower federation can
execute runs (e.g. to federate the training of an AI model) across all SuperNodes that
are part of it.


In this how-to guide, you will:

- Learn how to see the federations you are part of.
- Learn how to display information about a specific federation.


***************
 Listing Federations
***************

With the Flower CLI, you can easily inspect the federations your Flower account is part of:


.. code-block:: shell

    $ flwr federation list