Flower Datasets
===============

Flower Datasets (``flwr-datasets``) is a library that enables the quick and easy creation of datasets for federated learning/analytics/evaluation. It enables heterogeneity (non-iidness) simulation and division of datasets with the preexisting notion of IDs. The library was created by the ``Flower Labs`` team that also created `Flower <https://flower.ai>`_ : A Friendly Federated AI Framework.

Try out an interactive demo to generate code and visualize heterogeneous divisions at the :ref:`bottom of this page<demo>`.

Flower Datasets Framework
-------------------------

Install
~~~~~~~

.. code-block:: bash

  python -m pip install "flwr-datasets[vision]"

Check out all the details on how to install Flower Datasets in :doc:`how-to-install-flwr-datasets`.

Tutorials
~~~~~~~~~

A learning-oriented series of tutorials is the best place to start.

.. toctree::
   :maxdepth: 1
   :caption: Tutorial

   tutorial-quickstart
   tutorial-use-partitioners
   tutorial-visualize-label-distribution

How-to guides
~~~~~~~~~~~~~

Problem-oriented how-to guides show step-by-step how to achieve a specific goal.

.. toctree::
   :maxdepth: 1
   :caption: How-to guides

   how-to-install-flwr-datasets
   how-to-use-with-pytorch
   how-to-use-with-tensorflow
   how-to-use-with-numpy
   how-to-use-with-local-data
   how-to-disable-enable-progress-bar

References
~~~~~~~~~~

Information-oriented API reference and other reference material.

.. autosummary::
   :toctree: ref-api
   :template: autosummary/module.rst
   :caption: API reference
   :recursive:

      flwr_datasets

.. toctree::
   :maxdepth: 1
   :caption: Reference docs

   recommended-fl-datasets
   ref-telemetry

Main features
-------------
Flower Datasets library supports:

- **Downloading datasets** - choose the dataset from Hugging Face's ``dataset`` (`link <https://huggingface.co/datasets>`_)(*)
- **Partitioning datasets** - choose one of the implemented partitioning scheme or create your own.
- **Creating centralized datasets** - leave parts of the dataset unpartitioned (e.g. for centralized evaluation)
- **Visualization of the partitioned datasets** - visualize the label distribution of the partitioned dataset (and compare the results on different parameters of the same partitioning schemes, different datasets, different partitioning schemes, or any mix of them)

.. note::

  (*) Once the dataset is available on HuggingFace Hub it can be **immediately** used in ``Flower Datasets`` (no approval from the Flower team needed, no custom code needed).


.. image:: ./_static/readme/comparison_of_partitioning_schemes.png
  :align: center
  :alt: Comparison of Partitioning Schemes on CIFAR10

Thanks to using Hugging Face's ``datasets`` used under the hood, Flower Datasets integrates with the following popular formats/frameworks:

- Hugging Face
- PyTorch
- TensorFlow
- Numpy
- Pandas
- Jax
- Arrow

Here are a few of the ``Partitioner`` s that are available: (for a full list see `link <ref-api/flwr_datasets.partitioner.html#module-flwr_datasets.partitioner>`_ )

* Partitioner (the abstract base class) ``Partitioner``
* IID partitioning ``IidPartitioner(num_partitions)``
* Dirichlet partitioning ``DirichletPartitioner(num_partitions, partition_by, alpha)``
* Distribution partitioning ``DistributionPartitioner(distribution_array, num_partitions, num_unique_labels_per_partition, partition_by, preassigned_num_samples_per_label, rescale)``
* InnerDirichlet partitioning ``InnerDirichletPartitioner(partition_sizes, partition_by, alpha)``
* PathologicalPartitioner ``PathologicalPartitioner(num_partitions, partition_by, num_classes_per_partition, class_assignment_mode)``
* Natural ID partitioner ``NaturalIdPartitioner(partition_by)``
* Size partitioner (the abstract base class for the partitioners dictating the division based the number of samples) ``SizePartitioner``
* Linear partitioner ``LinearPartitioner(num_partitions)``
* Square partitioner ``SquarePartitioner(num_partitions)``
* Exponential partitioner ``ExponentialPartitioner(num_partitions)``
* more to come in the future releases (contributions are welcome).


How To Use the library
----------------------
Learn how to use the ``flwr-datasets`` library from the :doc:`tutorial-quickstart` examples .

Distinguishing Features
-----------------------
What makes Flower Datasets stand out from other libraries?

* Access to the largest online repository of datasets:

  * The library functionality is independent of the dataset, so you can use any dataset available on `🤗Hugging Face Datasets <https://huggingface.co/datasets>`_, which means that others can immediately benefit from the dataset you added.

  * Out-of-the-box reproducibility across different projects.

  * Access to naturally dividable datasets (with some notion of id) and datasets typically used in centralized ML that need partitioning.

* Customizable levels of dataset heterogeneity:

  * Each ``Partitioner`` takes arguments that allow you to customize the partitioning scheme to your needs.

  * Partitioning can also be applied to the dataset with naturally available division.

* Flexible and open for extensions API.

  * New custom partitioning schemes (``Partitioner`` subclasses) integrated with the whole ecosystem.

Join the Flower Community
-------------------------

The Flower Community is growing quickly - we're a friendly group of researchers, engineers, students, professionals, academics, and other enthusiasts.

.. button-link:: https://flower.ai/join-slack
    :color: primary
    :shadow:

    Join us on Slack

Recommended FL Datasets
-----------------------

Below we present a list of recommended datasets for federated learning research, which can be
used with Flower Datasets ``flwr-datasets``.

.. note::

    All datasets from `HuggingFace Hub <https://huggingface.co/datasets>`_ can be used with our library. This page presents just a set of datasets we collected that you might find useful.

For more information about any dataset, visit its page by clicking the dataset name. For more information how to use the

Image Datasets
^^^^^^^^^^^^^^

.. list-table:: Image Datasets
   :widths: 40 40 20
   :header-rows: 1

   * - Name
     - Size
     - Image Shape
   * - `ylecun/mnist <https://huggingface.co/datasets/ylecun/mnist>`_
     - train 60k;  
       test 10k
     - 28x28
   * - `uoft-cs/cifar10 <https://huggingface.co/datasets/uoft-cs/cifar10>`_
     - train 50k;  
       test 10k
     - 32x32x3
   * - `uoft-cs/cifar100 <https://huggingface.co/datasets/uoft-cs/cifar100>`_
     - train 50k;  
       test 10k
     - 32x32x3
   * - `zalando-datasets/fashion_mnist <https://huggingface.co/datasets/zalando-datasets/fashion_mnist>`_
     - train 60k;  
       test 10k
     - 28x28
   * - `flwrlabs/femnist <https://huggingface.co/datasets/flwrlabs/femnist>`_
     - train 814k
     - 28x28
   * - `zh-plus/tiny-imagenet <https://huggingface.co/datasets/zh-plus/tiny-imagenet>`_
     - train 100k;  
       valid 10k
     - 64x64x3
   * - `flwrlabs/usps <https://huggingface.co/datasets/flwrlabs/usps>`_
     - train 7.3k;  
       test 2k
     - 16x16
   * - `flwrlabs/pacs <https://huggingface.co/datasets/flwrlabs/pacs>`_
     - train 10k
     - 227x227
   * - `flwrlabs/cinic10 <https://huggingface.co/datasets/flwrlabs/cinic10>`_
     - train 90k;  
       valid 90k;  
       test 90k
     - 32x32x3
   * - `flwrlabs/caltech101 <https://huggingface.co/datasets/flwrlabs/caltech101>`_
     - train 8.7k
     - varies
   * - `flwrlabs/office-home <https://huggingface.co/datasets/flwrlabs/office-home>`_
     - train 15.6k
     - varies
   * - `flwrlabs/fed-isic2019 <https://huggingface.co/datasets/flwrlabs/fed-isic2019>`_
     - train 18.6k;  
       test 4.7k
     - varies
   * - `ufldl-stanford/svhn <https://huggingface.co/datasets/ufldl-stanford/svhn>`_
     - train 73.3k;  
       test 26k;  
       extra 531k
     - 32x32x3
   * - `sasha/dog-food <https://huggingface.co/datasets/sasha/dog-food>`_
     - train 2.1k;  
       test 0.9k
     - varies
   * - `Mike0307/MNIST-M <https://huggingface.co/datasets/Mike0307/MNIST-M>`_
     - train 59k;  
       test 9k
     - 32x32

Audio Datasets
^^^^^^^^^^^^^^

.. list-table:: Audio Datasets
   :widths: 35 30 15
   :header-rows: 1

   * - Name
     - Size
     - Subset
   * - `google/speech_commands <https://huggingface.co/datasets/google/speech_commands>`_
     - train 64.7k
     - v0.01
   * - `google/speech_commands <https://huggingface.co/datasets/google/speech_commands>`_
     - train 105.8k
     - v0.02
   * - `flwrlabs/ambient-acoustic-context <https://huggingface.co/datasets/flwrlabs/ambient-acoustic-context>`_
     - train 70.3k
     - 
   * - `fixie-ai/common_voice_17_0 <https://huggingface.co/datasets/fixie-ai/common_voice_17_0>`_
     - varies
     - 14 versions
   * - `fixie-ai/librispeech_asr <https://huggingface.co/datasets/fixie-ai/librispeech_asr>`_
     - varies
     - clean/other

Tabular Datasets
^^^^^^^^^^^^^^^^

.. list-table:: Tabular Datasets
   :widths: 35 30
   :header-rows: 1

   * - Name
     - Size
   * - `scikit-learn/adult-census-income <https://huggingface.co/datasets/scikit-learn/adult-census-income>`_
     - train 32.6k
   * - `jlh/uci-mushrooms <https://huggingface.co/datasets/jlh/uci-mushrooms>`_
     - train 8.1k
   * - `scikit-learn/iris <https://huggingface.co/datasets/scikit-learn/iris>`_
     - train 150

Text Datasets
^^^^^^^^^^^^^

.. list-table:: Text Datasets
   :widths: 40 30 30
   :header-rows: 1

   * - Name
     - Size
     - Category
   * - `sentiment140 <https://huggingface.co/datasets/sentiment140>`_
     - train 1.6M;  
       test 0.5k
     - Sentiment
   * - `google-research-datasets/mbpp <https://huggingface.co/datasets/google-research-datasets/mbpp>`_
     - full 974; sanitized 427
     - General
   * - `openai/openai_humaneval <https://huggingface.co/datasets/openai/openai_humaneval>`_
     - test 164
     - General
   * - `lukaemon/mmlu <https://huggingface.co/datasets/lukaemon/mmlu>`_
     - varies
     - General
   * - `takala/financial_phrasebank <https://huggingface.co/datasets/takala/financial_phrasebank>`_
     - train 4.8k
     - Financial
   * - `pauri32/fiqa-2018 <https://huggingface.co/datasets/pauri32/fiqa-2018>`_
     - train 0.9k; validation 0.1k; test 0.2k
     - Financial
   * - `zeroshot/twitter-financial-news-sentiment <https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment>`_
     - train 9.5k; validation 2.4k
     - Financial
   * - `bigbio/pubmed_qa <https://huggingface.co/datasets/bigbio/pubmed_qa>`_
     - train 2M; validation 11k
     - Medical
   * - `openlifescienceai/medmcqa <https://huggingface.co/datasets/openlifescienceai/medmcqa>`_
     - train 183k; validation 4.3k; test 6.2k
     - Medical
   * - `bigbio/med_qa <https://huggingface.co/datasets/bigbio/med_qa>`_
     - train 10.1k; test 1.3k; validation 1.3k
     - Medical

.. _demo:
Demo
----

.. raw:: html

  <script
    type="module"
    src="https://gradio.s3-us-west-2.amazonaws.com/4.44.0/gradio.js"
  ></script>

  <gradio-app src="https://flwrlabs-federated-learning-datasets-by-flwr-datasets.hf.space"></gradio-app>
