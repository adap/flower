Changelog
=========

Unreleased
----------

Incompatible changes
~~~~~~~~~~~~~~~~~~~~

* **Configurable** ``get_parameters`` (`#1242 <https://github.com/adap/flower/pull/1242>`_)

  The ``get_parameters`` method now accepts a configuration dictionary, just like ``get_properties``, ``fit``, and ``evaluate``.

Minor updates
~~~~~~~~~~~~~

* Add secure gRPC connection to the ``advanced_tensorflow`` code example (`#847 <https://github.com/adap/flower/pull/847>`_)


v0.19.0 (2022-05-18)
--------------------

* **Flower Baselines (preview): FedOpt, FedBN, FedAvgM** (`#919 <https://github.com/adap/flower/pull/919>`_, `#1127 <https://github.com/adap/flower/pull/1127>`_, `#914 <https://github.com/adap/flower/pull/914>`_)

  The first preview release of Flower Baselines has arrived! We're kickstarting Flower Baselines with implementations of FedOpt (FedYogi, FedAdam, FedAdagrad), FedBN, and FedAvgM. Check the documentation on how to use `Flower Baselines <https://flower.dev/docs/using-baselines.html>`_. With this first preview release we're also inviting the community to `contribute their own baselines <https://flower.dev/docs/contributing-baselines.html>`_.

* **C++ client SDK (preview) and code example** (`#1111 <https://github.com/adap/flower/pull/1111>`_)

  Preview support for Flower clients written in C++. The C++ preview includes a Flower client SDK and a quickstart code example that demonstrates a simple C++ client using the SDK.

* **Add experimental support for Python 3.10 and Python 3.11** (`#1135 <https://github.com/adap/flower/pull/1135>`_)

  Python 3.10 is the latest stable release of Python and Python 3.11 is due to be released in October. This Flower release adds experimental support for both Python versions.

* **Aggregate custom metrics through user-provided functions** (`#1144 <https://github.com/adap/flower/pull/1144>`_)

  Custom metrics (e.g., :code:`accuracy`) can now be aggregated without having to customize the strategy. Built-in strategies support two new arguments, :code:`fit_metrics_aggregation_fn` and :code:`evaluate_metrics_aggregation_fn`, that allow passing custom metric aggregation functions.

* **User-configurable round timeout** (`#1162 <https://github.com/adap/flower/pull/1162>`_)

  A new configuration value allows the round timeout to be set for :code:`start_server` and :code:`start_simulation`. If the :code:`config` dictionary contains a :code:`round_timeout` key (with a :code:`float` value in seconds), the server will wait *at least* :code:`round_timeout` seconds before it closes the connection.

* **Enable both federated evaluation and centralized evaluation to be used at the same time in all built-in strategies** (`#1091 <https://github.com/adap/flower/pull/1091>`_)

  Built-in strategies can now perform both federated evaluation (i.e., client-side) and centralized evaluation (i.e., server-side) in the same round. Federated evaluation can be disabled by setting :code:`fraction_eval` to :code:`0.0`.

* **Two new Jupyter Notebook tutorials** (`#1141 <https://github.com/adap/flower/pull/1141>`_)

  Two Jupyter Notebook tutorials (compatible with Google Colab) explain basic and intermediate Flower features:

  *An Introduction to Federated Learning*: `Open in Colab <https://colab.research.google.com/github/adap/flower/blob/main/tutorials/Flower-1-Intro-to-FL-PyTorch.ipynb>`_

  *Using Strategies in Federated Learning*: `Open in Colab <https://colab.research.google.com/github/adap/flower/blob/main/tutorials/Flower-2-Strategies-in-FL-PyTorch.ipynb>`_

* **New FedAvgM strategy (Federated Averaging with Server Momentum)** (`#1076 <https://github.com/adap/flower/pull/1076>`_)

  The new :code:`FedAvgM` strategy implements Federated Averaging with Server Momentum [Hsu et al., 2019].

* **New advanced PyTorch code example** (`#1007 <https://github.com/adap/flower/pull/1007>`_)

  A new code example (:code:`advanced_pytorch`) demonstrates advanced Flower concepts with PyTorch.

* **New JAX code example** (`#906 <https://github.com/adap/flower/pull/906>`_, `#1143 <https://github.com/adap/flower/pull/1143>`_)

  A new code example (:code:`jax_from_centralized_to_federated`) shows federated learning with JAX and Flower.

* **Minor updates**
    * New option to keep Ray running if Ray was already initialized in :code:`start_simulation` (`#1177 <https://github.com/adap/flower/pull/1177>`_)
    * Add support for custom :code:`ClientManager` as a :code:`start_simulation` parameter (`#1171 <https://github.com/adap/flower/pull/1171>`_)
    * New documentation for `implementing strategies <https://flower.dev/docs/implementing-strategies.html>`_ (`#1097 <https://github.com/adap/flower/pull/1097>`_, `#1175 <https://github.com/adap/flower/pull/1175>`_)
    * New mobile-friendly documentation theme (`#1174 <https://github.com/adap/flower/pull/1174>`_)
    * Limit version range for (optional) :code:`ray` dependency to include only compatible releases (:code:`>=1.9.2,<1.12.0`) (`#1205 <https://github.com/adap/flower/pull/1205>`_)

Incompatible changes:
~~~~~~~~~~~~~~~~~~~~~

* **Remove deprecated support for Python 3.6** (`#871 <https://github.com/adap/flower/pull/871>`_)
* **Remove deprecated KerasClient** (`#857 <https://github.com/adap/flower/pull/857>`_)
* **Remove deprecated no-op extra installs** (`#973 <https://github.com/adap/flower/pull/973>`_)
* **Remove deprecated proto fields from** :code:`FitRes` **and** :code:`EvaluateRes` (`#869 <https://github.com/adap/flower/pull/869>`_)
* **Remove deprecated QffedAvg strategy (replaced by QFedAvg)** (`#1107 <https://github.com/adap/flower/pull/1107>`_)
* **Remove deprecated DefaultStrategy strategy** (`#1142 <https://github.com/adap/flower/pull/1142>`_)
* **Remove deprecated support for eval_fn accuracy return value** (`#1142 <https://github.com/adap/flower/pull/1142>`_)
* **Remove deprecated support for passing initial parameters as NumPy ndarrays** (`#1142 <https://github.com/adap/flower/pull/1142>`_)


v0.18.0 (2022-02-28)
--------------------

What's new?
~~~~~~~~~~~

* **Improved Virtual Client Engine compatibility with Jupyter Notebook / Google Colab** (`#866 <https://github.com/adap/flower/pull/866>`_, `#872 <https://github.com/adap/flower/pull/872>`_, `#833 <https://github.com/adap/flower/pull/833>`_, `#1036 <https://github.com/adap/flower/pull/1036>`_)

  Simulations (using the Virtual Client Engine through :code:`start_simulation`) now work more smoothly on Jupyter Notebooks (incl. Google Colab) after installing Flower with the :code:`simulation` extra (:code:`pip install flwr[simulation]`).

* **New Jupyter Notebook code example** (`#833 <https://github.com/adap/flower/pull/833>`_)

  A new code example (:code:`quickstart_simulation`) demonstrates Flower simulations using the Virtual Client Engine through Jupyter Notebook (incl. Google Colab).

* **Client properties (feature preview)** (`#795 <https://github.com/adap/flower/pull/795>`_)

  Clients can implement a new method :code:`get_properties` to enable server-side strategies to query client properties.

* **Experimental Android support with TFLite** (`#865 <https://github.com/adap/flower/pull/865>`_)

  Android support has finally arrived in :code:`main`! Flower is both client-agnostic and framework-agnostic by design. One can integrate arbitrary client platforms and with this release, using Flower on Android has become a lot easier.

  The example uses TFLite on the client side, along with a new :code:`FedAvgAndroid` strategy. The Android client and :code:`FedAvgAndroid` are still experimental, but they are a first step towards a fully-fledged Android SDK and a unified :code:`FedAvg` implementation that integrated the new functionality from :code:`FedAvgAndroid`.

* **Make gRPC keepalive time user-configurable and decrease default keepalive time** (`#1069 <https://github.com/adap/flower/pull/1069>`_)

  The default gRPC keepalive time has been reduced to increase the compatibility of Flower with more cloud environments (for example, Microsoft Azure). Users can configure the keepalive time to customize the gRPC stack based on specific requirements.

* **New differential privacy example using Opacus and PyTorch** (`#805 <https://github.com/adap/flower/pull/805>`_)

  A new code example (:code:`opacus`) demonstrates differentially-private federated learning with Opacus, PyTorch, and Flower.

* **New Hugging Face Transformers code example** (`#863 <https://github.com/adap/flower/pull/863>`_)

  A new code example (:code:`quickstart_huggingface`) demonstrates usage of Hugging Face Transformers with Flower.

* **New MLCube code example** (`#779 <https://github.com/adap/flower/pull/779>`_, `#1034 <https://github.com/adap/flower/pull/1034>`_, `#1065 <https://github.com/adap/flower/pull/1065>`_, `#1090 <https://github.com/adap/flower/pull/1090>`_)

  A new code example (:code:`quickstart_mlcube`) demonstrates usage of MLCube with Flower.

* **SSL-enabled server and client** (`#842 <https://github.com/adap/flower/pull/842>`_,  `#844 <https://github.com/adap/flower/pull/844>`_,  `#845 <https://github.com/adap/flower/pull/845>`_, `#847 <https://github.com/adap/flower/pull/847>`_, `#993 <https://github.com/adap/flower/pull/993>`_, `#994 <https://github.com/adap/flower/pull/994>`_)

  SSL enables secure encrypted connections between clients and servers. This release open-sources the Flower secure gRPC implementation to make encrypted communication channels accessible to all Flower users.

* **Updated** :code:`FedAdam` **and** :code:`FedYogi` **strategies** (`#885 <https://github.com/adap/flower/pull/885>`_, `#895 <https://github.com/adap/flower/pull/895>`_)

  :code:`FedAdam` and :code:`FedAdam` match the latest version of the Adaptive Federated Optimization paper.

* **Initialize** :code:`start_simulation` **with a list of client IDs** (`#860 <https://github.com/adap/flower/pull/860>`_)

  :code:`start_simulation` can now be called with a list of client IDs (:code:`clients_ids`, type: :code:`List[str]`). Those IDs will be passed to the :code:`client_fn` whenever a client needs to be initialized, which can make it easier to load data partitions that are not accessible through :code:`int` identifiers.

* **Minor updates**
    * Update :code:`num_examples` calculation in PyTorch code examples in (`#909 <https://github.com/adap/flower/pull/909>`_)
    * Expose Flower version through :code:`flwr.__version__` (`#952 <https://github.com/adap/flower/pull/952>`_)
    * :code:`start_server` in :code:`app.py` now returns a :code:`History` object containing metrics from training (`#974 <https://github.com/adap/flower/pull/974>`_)
    * Make :code:`max_workers` (used by :code:`ThreadPoolExecutor`) configurable (`#978 <https://github.com/adap/flower/pull/978>`_)
    * Increase sleep time after server start to three seconds in all code examples (`#1086 <https://github.com/adap/flower/pull/1086>`_)
    * Added a new FAQ section to the documentation (`#948 <https://github.com/adap/flower/pull/948>`_)
    * And many more under-the-hood changes, library updates, documentation changes, and tooling improvements!

Incompatible changes:
~~~~~~~~~~~~~~~~~~~~~

* **Removed** :code:`flwr_example` **and** :code:`flwr_experimental` **from release build** (`#869 <https://github.com/adap/flower/pull/869>`_)
  
  The packages :code:`flwr_example` and :code:`flwr_experimental` have been deprecated since Flower 0.12.0 and they are not longer included in Flower release builds. The associated extras (:code:`baseline`, :code:`examples-pytorch`, :code:`examples-tensorflow`, :code:`http-logger`, :code:`ops`) are now no-op and will be removed in an upcoming release.


v0.17.0 (2021-09-24)
--------------------

What's new?
~~~~~~~~~~~

* **Experimental virtual client engine** (`#781 <https://github.com/adap/flower/pull/781>`_ `#790 <https://github.com/adap/flower/pull/790>`_ `#791 <https://github.com/adap/flower/pull/791>`_)

  One of Flower's goals is to enable research at scale. This release enables a first (experimental) peek at a major new feature, codenamed the virtual client engine. Virtual clients enable simulations that scale to a (very) large number of clients on a single machine or compute cluster. The easiest way to test the new functionality is to look at the two new code examples called :code:`quickstart_simulation` and :code:`simulation_pytorch`.

  The feature is still experimental, so there's no stability guarantee for the API. It's also not quite ready for prime time and comes with a few known caveats. However, those who are curious are encouraged to try it out and share their thoughts.

* **New built-in strategies** (`#828 <https://github.com/adap/flower/pull/828>`_ `#822 <https://github.com/adap/flower/pull/822>`_)
    * FedYogi - Federated learning strategy using Yogi on server-side. Implementation based on https://arxiv.org/abs/2003.00295
    * FedAdam - Federated learning strategy using Adam on server-side. Implementation based on https://arxiv.org/abs/2003.00295

* **New PyTorch Lightning code example** (`#617 <https://github.com/adap/flower/pull/617>`_)

* **New Variational Auto-Encoder code example** (`#752 <https://github.com/adap/flower/pull/752>`_)

* **New scikit-learn code example** (`#748 <https://github.com/adap/flower/pull/748>`_)

* **New experimental TensorBoard strategy** (`#789 <https://github.com/adap/flower/pull/789>`_)

* **Minor updates**
    * Improved advanced TensorFlow code example (`#769 <https://github.com/adap/flower/pull/769>`_)
    * Warning when :code:`min_available_clients` is misconfigured (`#830 <https://github.com/adap/flower/pull/830>`_)
    * Improved gRPC server docs (`#841 <https://github.com/adap/flower/pull/841>`_)
    * Improved error message in :code:`NumPyClient` (`#851 <https://github.com/adap/flower/pull/851>`_)
    * Improved PyTorch quickstart code example (`#852 <https://github.com/adap/flower/pull/852>`_)

Incompatible changes:
~~~~~~~~~~~~~~~~~~~~~

* **Disabled final distributed evaluation** (`#800 <https://github.com/adap/flower/pull/800>`_)

  Prior behaviour was to perform a final round of distributed evaluation on all connected clients, which is often not required (e.g., when using server-side evaluation). The prior behaviour can be enabled by passing :code:`force_final_distributed_eval=True` to :code:`start_server`.

* **Renamed q-FedAvg strategy** (`#802 <https://github.com/adap/flower/pull/802>`_)

  The strategy named :code:`QffedAvg` was renamed to `QFedAvg` to better reflect the notation given in the original paper (q-FFL is the optimization objective, q-FedAvg is the proposed solver). Note the the original (now deprecated) :code:`QffedAvg` class is still available for compatibility reasons (it will be removed in a future release).

* **Deprecated and renamed code example** :code:`simulation_pytorch` **to** :code:`simulation_pytorch_legacy` (`#791 <https://github.com/adap/flower/pull/791>`_)

  This example has been replaced by a new example. The new example is based on the experimental virtual client engine, which will become the new default way of doing most types of large-scale simulations in Flower. The existing example was kept for reference purposes, but it might be removed in the future.


v0.16.0 (2021-05-11)
--------------------

What's new?

* **New built-in strategies** (`#549 <https://github.com/adap/flower/pull/549>`_)
    * (abstract) FedOpt
    * FedAdagrad

* **Custom metrics for server and strategies** (`#717 <https://github.com/adap/flower/pull/717>`_)

  The Flower server is now fully task-agnostic, all remaining instances of task-specific metrics (such as :code:`accuracy`) have been replaced by custom metrics dictionaries. Flower 0.15 introduced the capability to pass a dictionary containing custom metrics from client to server. As of this release, custom metrics replace task-specific metrics on the server.

  Custom metric dictionaries are now used in two user-facing APIs: they are returned from Strategy methods :code:`aggregate_fit`/:code:`aggregate_evaluate` and they enable evaluation functions passed to build-in strategies (via :code:`eval_fn`) to return more than two evaluation metrics. Strategies can even return *aggregated* metrics dictionaries for the server to keep track of.

  Stratey implementations should migrate their :code:`aggregate_fit` and :code:`aggregate_evaluate` methods to the new return type (e.g., by simply returning an empty :code:`{}`), server-side evaluation functions should migrate from :code:`return loss, accuracy` to :code:`return loss, {"accuracy": accuracy}`.

  Flower 0.15-style return types are deprecated (but still supported), compatibility will be removed in a future release.

* **Migration warnings for deprecated functionality** (`#690 <https://github.com/adap/flower/pull/690>`_)

  Earlier versions of Flower were often migrated to new APIs, while maintaining compatibility with legacy APIs. This release introduces detailed warning messages if usage of deprecated APIs is detected. The new warning messages often provide details on how to migrate to more recent APIs, thus easing the transition from one release to another.

* Improved docs and docstrings (`#691 <https://github.com/adap/flower/pull/691>`_ `#692 <https://github.com/adap/flower/pull/692>`_ `#713 <https://github.com/adap/flower/pull/713>`_)

* MXNet example and documentation

* FedBN implementation in example PyTorch: From Centralized To Federated (`#696 <https://github.com/adap/flower/pull/696>`_ `#702 <https://github.com/adap/flower/pull/702>`_ `#705 <https://github.com/adap/flower/pull/705>`_)

Incompatible changes:

* **Serialization-agnostic server** (`#721 <https://github.com/adap/flower/pull/721>`_)

  The Flower server is now fully serialization-agnostic. Prior usage of class :code:`Weights` (which represents parameters as deserialized NumPy ndarrays) was replaced by class :code:`Parameters` (e.g., in :code:`Strategy`). :code:`Parameters` objects are fully serialization-agnostic and represents parameters as byte arrays, the :code:`tensor_type` attributes indicates how these byte arrays should be interpreted (e.g., for serialization/deserialization).

  Built-in strategies implement this approach by handling serialization and deserialization to/from :code:`Weights` internally. Custom/3rd-party Strategy implementations should update to the slighly changed Strategy method definitions. Strategy authors can consult PR `#721 <https://github.com/adap/flower/pull/721>`_ to see how strategies can easily migrate to the new format.

* Deprecated :code:`flwr.server.Server.evaluate`, use :code:`flwr.server.Server.evaluate_round` instead (`#717 <https://github.com/adap/flower/pull/717>`_)


v0.15.0 (2021-03-12)
--------------------

What's new?

* **Server-side parameter initialization** (`#658 <https://github.com/adap/flower/pull/658>`_)

  Model parameters can now be initialized on the server-side. Server-side parameter initialization works via a new :code:`Strategy` method called :code:`initialize_parameters`.

  Built-in strategies support a new constructor argument called :code:`initial_parameters` to set the initial parameters. Built-in strategies will provide these initial parameters to the server on startup and then delete them to free the memory afterwards.

  .. code-block:: python

    # Create model
    model = tf.keras.applications.EfficientNetB0(
        input_shape=(32, 32, 3), weights=None, classes=10
    )
    model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

    # Create strategy and initilize parameters on the server-side
    strategy = fl.server.strategy.FedAvg(
        # ... (other constructor arguments)
        initial_parameters=model.get_weights(),
    )

    # Start Flower server with the strategy
    fl.server.start_server("[::]:8080", config={"num_rounds": 3}, strategy=strategy)

  If no initial parameters are provided to the strategy, the server will continue to use the current behaviour (namely, it will ask one of the connected clients for its parameters and use these as the initial global parameters).

Deprecations

* Deprecate :code:`flwr.server.strategy.DefaultStrategy` (migrate to :code:`flwr.server.strategy.FedAvg`, which is equivalent)


v0.14.0 (2021-02-18)
--------------------

What's new?

* **Generalized** :code:`Client.fit` **and** :code:`Client.evaluate` **return values** (`#610 <https://github.com/adap/flower/pull/610>`_ `#572 <https://github.com/adap/flower/pull/572>`_ `#633 <https://github.com/adap/flower/pull/633>`_)

  Clients can now return an additional dictionary mapping :code:`str` keys to values of the following types: :code:`bool`, :code:`bytes`, :code:`float`, :code:`int`, :code:`str`. This means one can return almost arbitrary values from :code:`fit`/:code:`evaluate` and make use of them on the server side!
  
  This improvement also allowed for more consistent return types between :code:`fit` and :code:`evaluate`: :code:`evaluate` should now return a tuple :code:`(float, int, dict)` representing the loss, number of examples, and a dictionary holding arbitrary problem-specific values like accuracy. 
  
  In case you wondered: this feature is compatible with existing projects, the additional dictionary return value is optional. New code should however migrate to the new return types to be compatible with upcoming Flower releases (:code:`fit`: :code:`List[np.ndarray], int, Dict[str, Scalar]`, :code:`evaluate`: :code:`float, int, Dict[str, Scalar]`). See the example below for details.

  *Code example:* note the additional dictionary return values in both :code:`FlwrClient.fit` and :code:`FlwrClient.evaluate`: 

  .. code-block:: python

    class FlwrClient(fl.client.NumPyClient):
        def fit(self, parameters, config):
            net.set_parameters(parameters)
            train_loss = train(net, trainloader)
            return net.get_weights(), len(trainloader), {"train_loss": train_loss}

        def evaluate(self, parameters, config):
            net.set_parameters(parameters)
            loss, accuracy, custom_metric = test(net, testloader)
            return loss, len(testloader), {"accuracy": accuracy, "custom_metric": custom_metric}

* **Generalized** :code:`config` **argument in** :code:`Client.fit` **and** :code:`Client.evaluate` (`#595 <https://github.com/adap/flower/pull/595>`_)

  The :code:`config` argument used to be of type :code:`Dict[str, str]`, which means that dictionary values were expected to be strings. The new release generalizes this to enable values of the following types: :code:`bool`, :code:`bytes`, :code:`float`, :code:`int`, :code:`str`.
  
  This means one can now pass almost arbitrary values to :code:`fit`/:code:`evaluate` using the :code:`config` dictionary. Yay, no more :code:`str(epochs)` on the server-side and :code:`int(config["epochs"])` on the client side!

  *Code example:* note that the :code:`config` dictionary now contains non-:code:`str` values in both :code:`Client.fit` and :code:`Client.evaluate`: 

  .. code-block:: python
  
    class FlwrClient(fl.client.NumPyClient):
        def fit(self, parameters, config):
            net.set_parameters(parameters)
            epochs: int = config["epochs"]
            train_loss = train(net, trainloader, epochs)
            return net.get_weights(), len(trainloader), {"train_loss": train_loss}

        def evaluate(self, parameters, config):
            net.set_parameters(parameters)
            batch_size: int = config["batch_size"]
            loss, accuracy = test(net, testloader, batch_size)
            return loss, len(testloader), {"accuracy": accuracy}


v0.13.0 (2021-01-08)
--------------------

What's new?

* New example: PyTorch From Centralized To Federated (`#549 <https://github.com/adap/flower/pull/549>`_)
* Improved documentation
    * New documentation theme (`#551 <https://github.com/adap/flower/pull/551>`_)
    * New API reference (`#554 <https://github.com/adap/flower/pull/554>`_)
    * Updated examples documentation (`#549 <https://github.com/adap/flower/pull/549>`_)
    * Removed obsolete documentation (`#548 <https://github.com/adap/flower/pull/548>`_)

Bugfix:

* :code:`Server.fit` does not disconnect clients when finished, disconnecting the clients is now handled in :code:`flwr.server.start_server` (`#553 <https://github.com/adap/flower/pull/553>`_ `#540 <https://github.com/adap/flower/issues/540>`_).


v0.12.0 (2020-12-07)
--------------------

Important changes:

* Added an example for embedded devices (`#507 <https://github.com/adap/flower/pull/507>`_)
* Added a new NumPyClient (in addition to the existing KerasClient) (`#504 <https://github.com/adap/flower/pull/504>`_ `#508 <https://github.com/adap/flower/pull/508>`_)
* Deprecated `flwr_example` package and started to migrate examples into the top-level `examples` directory (`#494 <https://github.com/adap/flower/pull/494>`_ `#512 <https://github.com/adap/flower/pull/512>`_)


v0.11.0 (2020-11-30)
--------------------

Incompatible changes:

* Renamed strategy methods (`#486 <https://github.com/adap/flower/pull/486>`_) to unify the naming of Flower's public APIs. Other public methods/functions (e.g., every method in :code:`Client`, but also :code:`Strategy.evaluate`) do not use the :code:`on_` prefix, which is why we're removing it from the four methods in Strategy. To migrate rename the following :code:`Strategy` methods accordingly:
    * :code:`on_configure_evaluate` => :code:`configure_evaluate`
    * :code:`on_aggregate_evaluate` => :code:`aggregate_evaluate`
    * :code:`on_configure_fit` => :code:`configure_fit`
    * :code:`on_aggregate_fit` => :code:`aggregate_fit`

Important changes:

* Deprecated :code:`DefaultStrategy` (`#479 <https://github.com/adap/flower/pull/479>`_). To migrate use :code:`FedAvg` instead.
* Simplified examples and baselines (`#484 <https://github.com/adap/flower/pull/484>`_).
* Removed presently unused :code:`on_conclude_round` from strategy interface (`#483 <https://github.com/adap/flower/pull/483>`_).
* Set minimal Python version to 3.6.1 instead of 3.6.9 (`#471 <https://github.com/adap/flower/pull/471>`_).
* Improved :code:`Strategy` docstrings (`#470 <https://github.com/adap/flower/pull/470>`_).
