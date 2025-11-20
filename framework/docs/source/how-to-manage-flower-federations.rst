:og:description: Guide to manage Flower federations using the Deployment Engine.
.. meta::
    :description: Guide to manage Flower federations using the Deployment Engine.

.. |flower_cli_federation_link| replace:: ``Flower CLI``

.. _flower_cli_federation_link: ref-api-cli.html#flwr-federation

.. note::

    Flower Federation management is a new feature introduced in Flower 1.24.0. It gain
    new functionality in subsequen releases. Changes to the functionality described in
    this guide is also expected as the features provided via the ``flwr federation``
    commands mature.

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

*********************
 Listing Federations
*********************

With the |flower_cli_federation_link|_, you can easily inspect the federations your
Flower account is part of:

.. code-block:: shell

    $ flwr federation list

The above command will display a table with a row for each federation you are part of.
In this case there is only one federation named ``default``:

.. code-block:: shell

    Loading project configuration...
    Success
    📄 Listing federations...
    ┏━━━━━━━━━━━━┓
    ┃ Federation ┃
    ┡━━━━━━━━━━━━┩
    │ default    │
    └────────────┘

*************************
 Inspecting a Federation
*************************

You can inspect a specific federation by using the ``flwr federation show`` command,
another command provided by the |flower_cli_federation_link|_. With this command, you
will be able to see the following information about a federation:

- The members of the federation.
- The SuperNodes registered with the federation and their status.
- The runs executed via the federation.

The ``flwr federation show`` command requires the name of the federation to inspect as
an argument. this can be specified as part of your `pyproject.toml` configuration. For
example:

.. code-block:: toml
    :emphasize-lines: 4
    :caption: pyproject.toml

    [tool.flwr.federations.local-deployment]
    address = "127.0.0.1:9093"
    insecure = true
    federation = "default"

In this example, the federation named ``default`` is specified. You can now inspect it
by running:

.. code-block:: shell

    $ flwr federation show local-deployment

Assuming the ``default`` federation has ...
