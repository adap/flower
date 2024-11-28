How to contribute a dataset
===========================

To make a dataset available in Flower Dataset (`flwr-datasets`), you need to add the dataset to `HuggingFace Hub <https://huggingface.co/>`_ .

This guide will explain the best practices we found when adding datasets ourselves and point to the HFs guides. To see the datasets added by Flower, visit https://huggingface.co/flwrlabs.

Dataset contribution process
----------------------------
The contribution contains two steps:

1. Create the dataset locally.

We recommend that you do not upload custom scripts to HuggingFace Hub; instead, create the dataset locally and upload the data, which will speed up the processing time each time the data set is downloaded.

2. Contribute to HuggingFace Hub.

Each dataset in the HF Hub is a Git repository with a specific structure and readme file, and HuggingFace provides an API to push the dataset and, alternatively, a user interface directly in the website to populate the information in the readme file.



Creating a dataset locally
^^^^^^^^^^^^^^^^^^^^^^^^^^
You can create a local dataset directly using the `datasets` library or load it in any custom way and transform it to the `datasets.Dataset` from other Python objects.
To complete this step, we recommend reading our guide available here: :doc:`how-to-use-with-local-data` or/and reading the guide from HF `Create a dataset <https://huggingface.co/docs/datasets/create_dataset>`_.

Contribution to the HuggingFace Hub
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
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

To make the dataset easily accessible in FL we recommend adding the "Use in FL" section. Here's an example of how it is done in `one of our reps  <https://huggingface.co/datasets/flwrlabs/cinic10#use-in-fl>`_ for the cinic10 dataset.

That's it! You have successfully contributed a dataset to the HuggingFace Hub. If you want the dataset listed in our `recommended FL dataset list <https://flower.ai/docs/datasets/recommended-fl-datasets.html>`_  , please send a PR or ping us in `Slack <https://flower.ai/join-slack/>`_ #contributions channel.