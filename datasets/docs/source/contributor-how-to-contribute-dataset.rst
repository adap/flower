How to contribute a dataset
===========================

To make a dataset available in Flower Dataset (`flwr-datasets`), you need to add the dataset to `HuggingFace Hub <https://huggingface.co/datasets>`_ .

This guide will explain the best practices we found when adding datasets ourselves and point to the HFs guides. To see the datasets added by Flower, visit https://huggingface.co/flwrlabs.

Dataset contribution process
----------------------------
The contribution contains three steps: first, on your development machine transform your dataset into a ``datasets.Dataset`` object, the preferred format for datasets in HF Hub; second, upload the dataset to HuggingFace Hub and detail it its readme how can be used with Flower Dataset; third, share your dataset with us and we will add it to the `recommended FL dataset list <https://flower.ai/docs/datasets/recommended-fl-datasets.html>`_ 

Creating a dataset locally
^^^^^^^^^^^^^^^^^^^^^^^^^^
You can create a local dataset directly using the `datasets` library or load it in any custom way and transform it to the `datasets.Dataset` from other Python objects.
To complete this step, we recommend reading our :doc:`how-to-use-with-local-data` guide or/and the `Create a dataset <https://huggingface.co/docs/datasets/create_dataset>`_ guide from HF.

.. tip::
    We recommend that you do not upload custom scripts to HuggingFace Hub; instead, create the dataset locally and upload the data, which will speed up the processing time each time the data set is downloaded.

Contribution to the HuggingFace Hub
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Each dataset in the HF Hub is a Git repository with a specific structure and readme file, and HuggingFace provides an API to push the dataset and, alternatively, a user interface directly in the website to populate the information in the readme file.

Contributions to the HuggingFace Hub come down to:

1. creating an HF repository for the dataset.
2. uploading the dataset.
3. filling in the information in the readme file.

To complete this step, follow this HF's guide `Share dataset to the Hub <https://huggingface.co/docs/datasets/upload_dataset>`_.

Note that the push of the dataset is straightforward, and here's what it could look like:

.. code-block:: python

    from datasets import Dataset

    # Example dataset
    data = {
        'column1': [1, 2, 3],
        'column2': ['a', 'b', 'c']
    }

    # Create a Dataset object
    dataset = Dataset.from_dict(data)

    # Push the dataset to the HuggingFace Hub
    dataset.push_to_hub("you-hf-username/your-ds-name")

To make the dataset easily accessible in FL we recommend adding the "Use in FL" section. Here's an example of how it is done in `one of our repos  <https://huggingface.co/datasets/flwrlabs/cinic10#use-in-fl>`_ for the cinic10 dataset.

Increasing visibility of the dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
If you want the dataset listed in our `recommended FL dataset list <https://flower.ai/docs/datasets/recommended-fl-datasets.html>`_  , please send a PR or ping us in `Slack <https://flower.ai/join-slack/>`_ #contributions channel.

That's it! You have successfully contributed a dataset to the HuggingFace Hub and made it available for FL community. Thank you for your contribution!