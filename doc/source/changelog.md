# Changelog

## Unreleased

- **Log** `Client` **exceptions in Virtual Client Engine**

  All `Client` exceptions happening in the VCE will now be logged by default and not just exposed through the strategies.

- **Support optional** `Client`**/**`NumPyClient` **methods in Virtual Client Engine**

- **Provide type information to packages using** `flwr` ([#1377](https://github.com/adap/flower/pull/1377))

  The package `flwr` is now bundled with a `py.typed` file indicating that the package is typed. This enables typing support for projects or packages that use `flwr` by enabling them to type-check their code using tool like `mypy`.

## v1.0.0 (2022-07-28)

### Highlights

- Stable **Virtual Client Engine** (accessible via `start_simulation`)
- All `Client`/`NumPyClient` methods are now optional
- Configurable `get_parameters`
- Tons of small API cleanups resulting in a more coherent developer experience

### Thanks to our contributors

We would like to give our **special thanks** to all the contributors who made Flower 1.0 possible (in reverse [GitHub Contributors](https://github.com/adap/flower/graphs/contributors) order):

[@rtaiello](https://github.com/rtaiello), [@g-pichler](https://github.com/g-pichler), [@rob-luke](https://github.com/rob-luke), [@andreea-zaharia](https://github.com/andreea-zaharia), [@kinshukdua](https://github.com/kinshukdua), [@nfnt](https://github.com/nfnt), [@tatiana-s](https://github.com/tatiana-s), [@TParcollet](https://github.com/TParcollet), [@vballoli](https://github.com/vballoli), [@negedng](https://github.com/negedng), [@RISHIKESHAVAN](https://github.com/RISHIKESHAVAN), [@hei411](https://github.com/hei411), [@SebastianSpeitel](https://github.com/SebastianSpeitel), [@AmitChaulwar](https://github.com/AmitChaulwar), [@Rubiel1](https://github.com/Rubiel1), [@FANTOME-PAN](https://github.com/FANTOME-PAN), [@Rono-BC](https://github.com/Rono-BC), [@lbhm](https://github.com/lbhm), [@sishtiaq](https://github.com/sishtiaq), [@remde](https://github.com/remde), [@Jueun-Park](https://github.com/Jueun-Park), [@architjen](https://github.com/architjen), [@PratikGarai](https://github.com/PratikGarai), [@mrinaald](https://github.com/mrinaald), [@zliel](https://github.com/zliel), [@MeiruiJiang](https://github.com/MeiruiJiang), [@sandracl72](https://github.com/sandracl72), [@gubertoli](https://github.com/gubertoli), [@Vingt100](https://github.com/Vingt100), [@MakGulati](https://github.com/MakGulati), [@cozek](https://github.com/cozek), [@jafermarq](https://github.com/jafermarq), [@sisco0](https://github.com/sisco0), [@akhilmathurs](https://github.com/akhilmathurs), [@CanTuerk](https://github.com/CanTuerk), [@mariaboerner1987](https://github.com/mariaboerner1987), [@pedropgusmao](https://github.com/pedropgusmao), [@tanertopal](https://github.com/tanertopal), [@danieljanes](https://github.com/danieljanes).

### Incompatible changes

- **All arguments must be passed as keyword arguments** ([#1338](https://github.com/adap/flower/pull/1338))

  Pass all arguments as keyword arguments, positional arguments are not longer supported. Code that uses positional arguments (e.g., ``start_client("127.0.0.1:8080", FlowerClient())``) must add the keyword for each positional argument (e.g., ``start_client(server_address="127.0.0.1:8080", client=FlowerClient())``).

- **Introduce configuration object** `ServerConfig` **in** `start_server` **and** `start_simulation` ([#1317](https://github.com/adap/flower/pull/1317))

  Instead of a config dictionary `{"num_rounds": 3, "round_timeout": 600.0}`, `start_server` and `start_simulation` now expect a configuration object of type `flwr.server.ServerConfig`. `ServerConfig` takes the same arguments that as the previous config dict, but it makes writing type-safe code easier and the default parameters values more transparent.

- **Rename built-in strategy parameters for clarity** ([#1334](https://github.com/adap/flower/pull/1334))

  The following built-in strategy parameters were renamed to improve readability and consistency with other API's:
  - `fraction_eval` --> `fraction_evaluate`
  - `min_eval_clients` --> `min_evaluate_clients`
  - `eval_fn` --> `evaluate_fn`

- **Update default arguments of built-in strategies** ([#1278](https://github.com/adap/flower/pull/1278))

  All built-in strategies now use `fraction_fit=1.0` and `fraction_evaluate=1.0`, which means they select *all* currently available clients for training and evaluation. Projects that relied on the previous default values can get the previous behaviour by initializing the strategy in the following way:

  `strategy = FedAvg(fraction_fit=0.1, fraction_evaluate=0.1)`

- **Add** `server_round` **to** `Strategy.evaluate` ([#1334](https://github.com/adap/flower/pull/1334))

  The `Strategy` method `evaluate` now receives the current round of federated learning/evaluation as the first parameter.

- **Add** `server_round` **and** `config` **parameters to** `evaluate_fn` ([#1334](https://github.com/adap/flower/pull/1334))

  The `evaluate_fn` passed to built-in strategies like `FedAvg` now takes three parameters: (1) The current round of federated learning/evaluation (`server_round`), (2) the model parameters to evaluate (`parameters`), and (3) a config dictionary (`config`).

- **Rename** `rnd` **to** `server_round` ([#1321](https://github.com/adap/flower/pull/1321))

  Several Flower methods and functions (`evaluate_fn`, `configure_fit`, `aggregate_fit`, `configure_evaluate`, `aggregate_evaluate`) receive the current round of federated learning/evaluation as their first parameter. To improve reaability and avoid confusion with *random*, this parameter has been renamed from `rnd` to `server_round`.

- **Move** `flwr.dataset` **to** `flwr_baselines` ([#1273](https://github.com/adap/flower/pull/1273))

  The experimental package `flwr.dataset` was migrated to Flower Baselines.

- **Remove experimental strategies** ([#1280](https://github.com/adap/flower/pull/1280))

  Remove unmaintained experimental strategies (`FastAndSlow`, `FedFSv0`, `FedFSv1`).

- **Rename** `Weights` **to** `NDArrays` ([#1258](https://github.com/adap/flower/pull/1258), [#1259](https://github.com/adap/flower/pull/1259))

  `flwr.common.Weights` was renamed to `flwr.common.NDArrays` to better capture what this type is all about.

- **Remove antiquated** `force_final_distributed_eval` **from** `start_server` ([#1258](https://github.com/adap/flower/pull/1258), [#1259](https://github.com/adap/flower/pull/1259))

  The `start_server` parameter `force_final_distributed_eval` has long been a historic artefact, in this release it is finally gone for good.

- **Make** `get_parameters` **configurable** ([#1242](https://github.com/adap/flower/pull/1242))

  The `get_parameters` method now accepts a configuration dictionary, just like `get_properties`, `fit`, and `evaluate`.

- **Replace** `num_rounds` **in** `start_simulation` **with new** `config` **parameter** ([#1281](https://github.com/adap/flower/pull/1281))
  
  The `start_simulation` function now accepts a configuration dictionary `config` instead of the `num_rounds` integer. This improves the consistency between `start_simulation` and `start_server` and makes transitioning between the two easier.

### New features

- **Support Python 3.10** ([#1320](https://github.com/adap/flower/pull/1320))

  The previous Flower release introduced experimental support for Python 3.10, this release declares Python 3.10 support as stable.

- **Make all** `Client` **and** `NumPyClient` **methods optional** ([#1260](https://github.com/adap/flower/pull/1260), [#1277](https://github.com/adap/flower/pull/1277))

  The `Client`/`NumPyClient` methods `get_properties`, `get_parameters`, `fit`, and `evaluate` are all optional. This enables writing clients that implement, for example, only `fit`, but no other method. No need to implement `evaluate` when using centralized evaluation!

- **Enable passing a** `Server` **instance to** `start_simulation` ([#1281](https://github.com/adap/flower/pull/1281))

  Similar to `start_server`, `start_simulation` now accepts a full `Server` instance. This enables users to heavily customize the execution of eperiments and opens the door to running, for example, async FL using the Virtual Client Engine.

- **Update code examples** ([#1291](https://github.com/adap/flower/pull/1291), [#1286](https://github.com/adap/flower/pull/1286), [#1282](https://github.com/adap/flower/pull/1282))

  Many code examples received small or even large maintenance updates, among them are
  - `scikit-learn`
  - `simulation_pytorch`
  - `quickstart_pytorch`
  - `quickstart_simulation`
  - `quickstart_tensorflow`
  - `advanced_tensorflow`

- **Remove the obsolete simulation example** ([#1328](https://github.com/adap/flower/pull/1328))

  Removes the obsolete `simulation` example and renames `quickstart_simulation` to `simulation_tensorflow` so it fits withs the naming of `simulation_pytorch`

- **Update documentation** ([#1223](https://github.com/adap/flower/pull/1223), [#1209](https://github.com/adap/flower/pull/1209), [#1251](https://github.com/adap/flower/pull/1251), [#1257](https://github.com/adap/flower/pull/1257), [#1267](https://github.com/adap/flower/pull/1267), [#1268](https://github.com/adap/flower/pull/1268), [#1300](https://github.com/adap/flower/pull/1300), [#1304](https://github.com/adap/flower/pull/1304), [#1305](https://github.com/adap/flower/pull/1305), [#1307](https://github.com/adap/flower/pull/1307))

  One substantial documentation update fixes multiple smaller rendering issues, makes titles more succinct to improve navigation, removes a deprecated library, updates documentation dependencies, includes the `flwr.common` module in the API reference, includes support for markdown-based documentation, migrates the changelog from `.rst` to `.md`, and fixes a number of smaller details!

- **Minor updates**
  - Add round number to fit and evaluate log messages ([#1266](https://github.com/adap/flower/pull/1266))
  - Add secure gRPC connection to the `advanced_tensorflow` code example ([#847](https://github.com/adap/flower/pull/847))
  - Update developer tooling ([#1231](https://github.com/adap/flower/pull/1231), [#1276](https://github.com/adap/flower/pull/1276), [#1301](https://github.com/adap/flower/pull/1301), [#1310](https://github.com/adap/flower/pull/1310))
  - Rename ProtoBuf messages to improve consistency ([#1214](https://github.com/adap/flower/pull/1214), [#1258](https://github.com/adap/flower/pull/1258), [#1259](https://github.com/adap/flower/pull/1259))

## v0.19.0 (2022-05-18)

- **Flower Baselines (preview): FedOpt, FedBN, FedAvgM** ([#919](https://github.com/adap/flower/pull/919), [#1127](https://github.com/adap/flower/pull/1127), [#914](https://github.com/adap/flower/pull/914))

  The first preview release of Flower Baselines has arrived! We're kickstarting Flower Baselines with implementations of FedOpt (FedYogi, FedAdam, FedAdagrad), FedBN, and FedAvgM. Check the documentation on how to use [Flower Baselines](https://flower.dev/docs/using-baselines.html). With this first preview release we're also inviting the community to [contribute their own baselines](https://flower.dev/docs/contributing-baselines.html).

- **C++ client SDK (preview) and code example** ([#1111](https://github.com/adap/flower/pull/1111))

  Preview support for Flower clients written in C++. The C++ preview includes a Flower client SDK and a quickstart code example that demonstrates a simple C++ client using the SDK.

- **Add experimental support for Python 3.10 and Python 3.11** ([#1135](https://github.com/adap/flower/pull/1135))

  Python 3.10 is the latest stable release of Python and Python 3.11 is due to be released in October. This Flower release adds experimental support for both Python versions.

- **Aggregate custom metrics through user-provided functions** ([#1144](https://github.com/adap/flower/pull/1144))

  Custom metrics (e.g., `accuracy`) can now be aggregated without having to customize the strategy. Built-in strategies support two new arguments, `fit_metrics_aggregation_fn` and `evaluate_metrics_aggregation_fn`, that allow passing custom metric aggregation functions.

- **User-configurable round timeout** ([#1162](https://github.com/adap/flower/pull/1162))

  A new configuration value allows the round timeout to be set for `start_server` and `start_simulation`. If the `config` dictionary contains a `round_timeout` key (with a `float` value in seconds), the server will wait *at least* `round_timeout` seconds before it closes the connection.

- **Enable both federated evaluation and centralized evaluation to be used at the same time in all built-in strategies** ([#1091](https://github.com/adap/flower/pull/1091))

  Built-in strategies can now perform both federated evaluation (i.e., client-side) and centralized evaluation (i.e., server-side) in the same round. Federated evaluation can be disabled by setting `fraction_eval` to `0.0`.

- **Two new Jupyter Notebook tutorials** ([#1141](https://github.com/adap/flower/pull/1141))

  Two Jupyter Notebook tutorials (compatible with Google Colab) explain basic and intermediate Flower features:

  *An Introduction to Federated Learning*: [Open in Colab](https://colab.research.google.com/github/adap/flower/blob/main/tutorials/Flower-1-Intro-to-FL-PyTorch.ipynb)

  *Using Strategies in Federated Learning*: [Open in Colab](https://colab.research.google.com/github/adap/flower/blob/main/tutorials/Flower-2-Strategies-in-FL-PyTorch.ipynb)

- **New FedAvgM strategy (Federated Averaging with Server Momentum)** ([#1076](https://github.com/adap/flower/pull/1076))

  The new `FedAvgM` strategy implements Federated Averaging with Server Momentum [Hsu et al., 2019].

- **New advanced PyTorch code example** ([#1007](https://github.com/adap/flower/pull/1007))

  A new code example (`advanced_pytorch`) demonstrates advanced Flower concepts with PyTorch.

- **New JAX code example** ([#906](https://github.com/adap/flower/pull/906), [#1143](https://github.com/adap/flower/pull/1143))

  A new code example (`jax_from_centralized_to_federated`) shows federated learning with JAX and Flower.

- **Minor updates**
    - New option to keep Ray running if Ray was already initialized in `start_simulation` ([#1177](https://github.com/adap/flower/pull/1177))
    - Add support for custom `ClientManager` as a `start_simulation` parameter ([#1171](https://github.com/adap/flower/pull/1171))
    - New documentation for [implementing strategies](https://flower.dev/docs/implementing-strategies.html) ([#1097](https://github.com/adap/flower/pull/1097), [#1175](https://github.com/adap/flower/pull/1175))
    - New mobile-friendly documentation theme ([#1174](https://github.com/adap/flower/pull/1174))
    - Limit version range for (optional) `ray` dependency to include only compatible releases (`>=1.9.2,<1.12.0`) ([#1205](https://github.com/adap/flower/pull/1205))

### Incompatible changes

- **Remove deprecated support for Python 3.6** ([#871](https://github.com/adap/flower/pull/871))
- **Remove deprecated KerasClient** ([#857](https://github.com/adap/flower/pull/857))
- **Remove deprecated no-op extra installs** ([#973](https://github.com/adap/flower/pull/973))
- **Remove deprecated proto fields from** `FitRes` **and** `EvaluateRes` ([#869](https://github.com/adap/flower/pull/869))
- **Remove deprecated QffedAvg strategy (replaced by QFedAvg)** ([#1107](https://github.com/adap/flower/pull/1107))
- **Remove deprecated DefaultStrategy strategy** ([#1142](https://github.com/adap/flower/pull/1142))
- **Remove deprecated support for eval_fn accuracy return value** ([#1142](https://github.com/adap/flower/pull/1142))
- **Remove deprecated support for passing initial parameters as NumPy ndarrays** ([#1142](https://github.com/adap/flower/pull/1142))

## v0.18.0 (2022-02-28)

### What's new?

- **Improved Virtual Client Engine compatibility with Jupyter Notebook / Google Colab** ([#866](https://github.com/adap/flower/pull/866), [#872](https://github.com/adap/flower/pull/872), [#833](https://github.com/adap/flower/pull/833), [#1036](https://github.com/adap/flower/pull/1036))

  Simulations (using the Virtual Client Engine through `start_simulation`) now work more smoothly on Jupyter Notebooks (incl. Google Colab) after installing Flower with the `simulation` extra (`pip install flwr[simulation]`).

- **New Jupyter Notebook code example** ([#833](https://github.com/adap/flower/pull/833))

  A new code example (`quickstart_simulation`) demonstrates Flower simulations using the Virtual Client Engine through Jupyter Notebook (incl. Google Colab).

- **Client properties (feature preview)** ([#795](https://github.com/adap/flower/pull/795))

  Clients can implement a new method `get_properties` to enable server-side strategies to query client properties.

- **Experimental Android support with TFLite** ([#865](https://github.com/adap/flower/pull/865))

  Android support has finally arrived in `main`! Flower is both client-agnostic and framework-agnostic by design. One can integrate arbitrary client platforms and with this release, using Flower on Android has become a lot easier.

  The example uses TFLite on the client side, along with a new `FedAvgAndroid` strategy. The Android client and `FedAvgAndroid` are still experimental, but they are a first step towards a fully-fledged Android SDK and a unified `FedAvg` implementation that integrated the new functionality from `FedAvgAndroid`.

- **Make gRPC keepalive time user-configurable and decrease default keepalive time** ([#1069](https://github.com/adap/flower/pull/1069))

  The default gRPC keepalive time has been reduced to increase the compatibility of Flower with more cloud environments (for example, Microsoft Azure). Users can configure the keepalive time to customize the gRPC stack based on specific requirements.

- **New differential privacy example using Opacus and PyTorch** ([#805](https://github.com/adap/flower/pull/805))

  A new code example (`opacus`) demonstrates differentially-private federated learning with Opacus, PyTorch, and Flower.

- **New Hugging Face Transformers code example** ([#863](https://github.com/adap/flower/pull/863))

  A new code example (`quickstart_huggingface`) demonstrates usage of Hugging Face Transformers with Flower.

- **New MLCube code example** ([#779](https://github.com/adap/flower/pull/779), [#1034](https://github.com/adap/flower/pull/1034), [#1065](https://github.com/adap/flower/pull/1065), [#1090](https://github.com/adap/flower/pull/1090))

  A new code example (`quickstart_mlcube`) demonstrates usage of MLCube with Flower.

- **SSL-enabled server and client** ([#842](https://github.com/adap/flower/pull/842),  [#844](https://github.com/adap/flower/pull/844),  [#845](https://github.com/adap/flower/pull/845), [#847](https://github.com/adap/flower/pull/847), [#993](https://github.com/adap/flower/pull/993), [#994](https://github.com/adap/flower/pull/994))

  SSL enables secure encrypted connections between clients and servers. This release open-sources the Flower secure gRPC implementation to make encrypted communication channels accessible to all Flower users.

- **Updated** `FedAdam` **and** `FedYogi` **strategies** ([#885](https://github.com/adap/flower/pull/885), [#895](https://github.com/adap/flower/pull/895))

  `FedAdam` and `FedAdam` match the latest version of the Adaptive Federated Optimization paper.

- **Initialize** `start_simulation` **with a list of client IDs** ([#860](https://github.com/adap/flower/pull/860))

  `start_simulation` can now be called with a list of client IDs (`clients_ids`, type: `List[str]`). Those IDs will be passed to the `client_fn` whenever a client needs to be initialized, which can make it easier to load data partitions that are not accessible through `int` identifiers.

- **Minor updates**
    - Update `num_examples` calculation in PyTorch code examples in ([#909](https://github.com/adap/flower/pull/909))
    - Expose Flower version through `flwr.__version__` ([#952](https://github.com/adap/flower/pull/952))
    - `start_server` in `app.py` now returns a `History` object containing metrics from training ([#974](https://github.com/adap/flower/pull/974))
    - Make `max_workers` (used by `ThreadPoolExecutor`) configurable ([#978](https://github.com/adap/flower/pull/978))
    - Increase sleep time after server start to three seconds in all code examples ([#1086](https://github.com/adap/flower/pull/1086))
    - Added a new FAQ section to the documentation ([#948](https://github.com/adap/flower/pull/948))
    - And many more under-the-hood changes, library updates, documentation changes, and tooling improvements!

### Incompatible changes:

- **Removed** `flwr_example` **and** `flwr_experimental` **from release build** ([#869](https://github.com/adap/flower/pull/869))
  
  The packages `flwr_example` and `flwr_experimental` have been deprecated since Flower 0.12.0 and they are not longer included in Flower release builds. The associated extras (`baseline`, `examples-pytorch`, `examples-tensorflow`, `http-logger`, `ops`) are now no-op and will be removed in an upcoming release.

## v0.17.0 (2021-09-24)

### What's new?

- **Experimental virtual client engine** ([#781](https://github.com/adap/flower/pull/781) [#790](https://github.com/adap/flower/pull/790) [#791](https://github.com/adap/flower/pull/791))

  One of Flower's goals is to enable research at scale. This release enables a first (experimental) peek at a major new feature, codenamed the virtual client engine. Virtual clients enable simulations that scale to a (very) large number of clients on a single machine or compute cluster. The easiest way to test the new functionality is to look at the two new code examples called `quickstart_simulation` and `simulation_pytorch`.

  The feature is still experimental, so there's no stability guarantee for the API. It's also not quite ready for prime time and comes with a few known caveats. However, those who are curious are encouraged to try it out and share their thoughts.

- **New built-in strategies** ([#828](https://github.com/adap/flower/pull/828) [#822](https://github.com/adap/flower/pull/822))
    - FedYogi - Federated learning strategy using Yogi on server-side. Implementation based on https://arxiv.org/abs/2003.00295
    - FedAdam - Federated learning strategy using Adam on server-side. Implementation based on https://arxiv.org/abs/2003.00295

- **New PyTorch Lightning code example** ([#617](https://github.com/adap/flower/pull/617))

- **New Variational Auto-Encoder code example** ([#752](https://github.com/adap/flower/pull/752))

- **New scikit-learn code example** ([#748](https://github.com/adap/flower/pull/748))

- **New experimental TensorBoard strategy** ([#789](https://github.com/adap/flower/pull/789))

- **Minor updates**
    - Improved advanced TensorFlow code example ([#769](https://github.com/adap/flower/pull/769))
    - Warning when `min_available_clients` is misconfigured ([#830](https://github.com/adap/flower/pull/830))
    - Improved gRPC server docs ([#841](https://github.com/adap/flower/pull/841))
    - Improved error message in `NumPyClient` ([#851](https://github.com/adap/flower/pull/851))
    - Improved PyTorch quickstart code example ([#852](https://github.com/adap/flower/pull/852))

### Incompatible changes:

- **Disabled final distributed evaluation** ([#800](https://github.com/adap/flower/pull/800))

  Prior behaviour was to perform a final round of distributed evaluation on all connected clients, which is often not required (e.g., when using server-side evaluation). The prior behaviour can be enabled by passing `force_final_distributed_eval=True` to `start_server`.

- **Renamed q-FedAvg strategy** ([#802](https://github.com/adap/flower/pull/802))

  The strategy named `QffedAvg` was renamed to `QFedAvg` to better reflect the notation given in the original paper (q-FFL is the optimization objective, q-FedAvg is the proposed solver). Note the the original (now deprecated) `QffedAvg` class is still available for compatibility reasons (it will be removed in a future release).

- **Deprecated and renamed code example** `simulation_pytorch` **to** `simulation_pytorch_legacy` ([#791](https://github.com/adap/flower/pull/791))

  This example has been replaced by a new example. The new example is based on the experimental virtual client engine, which will become the new default way of doing most types of large-scale simulations in Flower. The existing example was kept for reference purposes, but it might be removed in the future.

## v0.16.0 (2021-05-11)

What's new?

- **New built-in strategies** ([#549](https://github.com/adap/flower/pull/549))
    - (abstract) FedOpt
    - FedAdagrad

- **Custom metrics for server and strategies** ([#717](https://github.com/adap/flower/pull/717))

  The Flower server is now fully task-agnostic, all remaining instances of task-specific metrics (such as `accuracy`) have been replaced by custom metrics dictionaries. Flower 0.15 introduced the capability to pass a dictionary containing custom metrics from client to server. As of this release, custom metrics replace task-specific metrics on the server.

  Custom metric dictionaries are now used in two user-facing APIs: they are returned from Strategy methods `aggregate_fit`/`aggregate_evaluate` and they enable evaluation functions passed to build-in strategies (via `eval_fn`) to return more than two evaluation metrics. Strategies can even return *aggregated* metrics dictionaries for the server to keep track of.

  Stratey implementations should migrate their `aggregate_fit` and `aggregate_evaluate` methods to the new return type (e.g., by simply returning an empty `{}`), server-side evaluation functions should migrate from `return loss, accuracy` to `return loss, {"accuracy": accuracy}`.

  Flower 0.15-style return types are deprecated (but still supported), compatibility will be removed in a future release.

- **Migration warnings for deprecated functionality** ([#690](https://github.com/adap/flower/pull/690))

  Earlier versions of Flower were often migrated to new APIs, while maintaining compatibility with legacy APIs. This release introduces detailed warning messages if usage of deprecated APIs is detected. The new warning messages often provide details on how to migrate to more recent APIs, thus easing the transition from one release to another.

- Improved docs and docstrings ([#691](https://github.com/adap/flower/pull/691) [#692](https://github.com/adap/flower/pull/692) [#713](https://github.com/adap/flower/pull/713))

- MXNet example and documentation

- FedBN implementation in example PyTorch: From Centralized To Federated ([#696](https://github.com/adap/flower/pull/696) [#702](https://github.com/adap/flower/pull/702) [#705](https://github.com/adap/flower/pull/705))

Incompatible changes:

- **Serialization-agnostic server** ([#721](https://github.com/adap/flower/pull/721))

  The Flower server is now fully serialization-agnostic. Prior usage of class `Weights` (which represents parameters as deserialized NumPy ndarrays) was replaced by class `Parameters` (e.g., in `Strategy`). `Parameters` objects are fully serialization-agnostic and represents parameters as byte arrays, the `tensor_type` attributes indicates how these byte arrays should be interpreted (e.g., for serialization/deserialization).

  Built-in strategies implement this approach by handling serialization and deserialization to/from `Weights` internally. Custom/3rd-party Strategy implementations should update to the slighly changed Strategy method definitions. Strategy authors can consult PR [#721](https://github.com/adap/flower/pull/721) to see how strategies can easily migrate to the new format.

- Deprecated `flwr.server.Server.evaluate`, use `flwr.server.Server.evaluate_round` instead ([#717](https://github.com/adap/flower/pull/717))

## v0.15.0 (2021-03-12)

What's new?

- **Server-side parameter initialization** ([#658](https://github.com/adap/flower/pull/658))

  Model parameters can now be initialized on the server-side. Server-side parameter initialization works via a new `Strategy` method called `initialize_parameters`.

  Built-in strategies support a new constructor argument called `initial_parameters` to set the initial parameters. Built-in strategies will provide these initial parameters to the server on startup and then delete them to free the memory afterwards.

  ```python
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
  ```

  If no initial parameters are provided to the strategy, the server will continue to use the current behaviour (namely, it will ask one of the connected clients for its parameters and use these as the initial global parameters).

Deprecations

- Deprecate `flwr.server.strategy.DefaultStrategy` (migrate to `flwr.server.strategy.FedAvg`, which is equivalent)

## v0.14.0 (2021-02-18)

What's new?

- **Generalized** `Client.fit` **and** `Client.evaluate` **return values** ([#610](https://github.com/adap/flower/pull/610) [#572](https://github.com/adap/flower/pull/572) [#633](https://github.com/adap/flower/pull/633))

  Clients can now return an additional dictionary mapping `str` keys to values of the following types: `bool`, `bytes`, `float`, `int`, `str`. This means one can return almost arbitrary values from `fit`/`evaluate` and make use of them on the server side!
  
  This improvement also allowed for more consistent return types between `fit` and `evaluate`: `evaluate` should now return a tuple `(float, int, dict)` representing the loss, number of examples, and a dictionary holding arbitrary problem-specific values like accuracy. 
  
  In case you wondered: this feature is compatible with existing projects, the additional dictionary return value is optional. New code should however migrate to the new return types to be compatible with upcoming Flower releases (`fit`: `List[np.ndarray], int, Dict[str, Scalar]`, `evaluate`: `float, int, Dict[str, Scalar]`). See the example below for details.

  *Code example:* note the additional dictionary return values in both `FlwrClient.fit` and `FlwrClient.evaluate`: 

  ```python
  class FlwrClient(fl.client.NumPyClient):
      def fit(self, parameters, config):
          net.set_parameters(parameters)
          train_loss = train(net, trainloader)
          return net.get_weights(), len(trainloader), {"train_loss": train_loss}

      def evaluate(self, parameters, config):
          net.set_parameters(parameters)
          loss, accuracy, custom_metric = test(net, testloader)
          return loss, len(testloader), {"accuracy": accuracy, "custom_metric": custom_metric}
  ```

- **Generalized** `config` **argument in** `Client.fit` **and** `Client.evaluate` ([#595](https://github.com/adap/flower/pull/595))

  The `config` argument used to be of type `Dict[str, str]`, which means that dictionary values were expected to be strings. The new release generalizes this to enable values of the following types: `bool`, `bytes`, `float`, `int`, `str`.
  
  This means one can now pass almost arbitrary values to `fit`/`evaluate` using the `config` dictionary. Yay, no more `str(epochs)` on the server-side and `int(config["epochs"])` on the client side!

  *Code example:* note that the `config` dictionary now contains non-`str` values in both `Client.fit` and `Client.evaluate`: 

  ```python
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
  ```

## v0.13.0 (2021-01-08)

What's new?

- New example: PyTorch From Centralized To Federated ([#549](https://github.com/adap/flower/pull/549))
- Improved documentation
    - New documentation theme ([#551](https://github.com/adap/flower/pull/551))
    - New API reference ([#554](https://github.com/adap/flower/pull/554))
    - Updated examples documentation ([#549](https://github.com/adap/flower/pull/549))
    - Removed obsolete documentation ([#548](https://github.com/adap/flower/pull/548))

Bugfix:

- `Server.fit` does not disconnect clients when finished, disconnecting the clients is now handled in `flwr.server.start_server` ([#553](https://github.com/adap/flower/pull/553) [#540](https://github.com/adap/flower/issues/540)).

## v0.12.0 (2020-12-07)

Important changes:

- Added an example for embedded devices ([#507](https://github.com/adap/flower/pull/507))
- Added a new NumPyClient (in addition to the existing KerasClient) ([#504](https://github.com/adap/flower/pull/504) [#508](https://github.com/adap/flower/pull/508))
- Deprecated `flwr_example` package and started to migrate examples into the top-level `examples` directory ([#494](https://github.com/adap/flower/pull/494) [#512](https://github.com/adap/flower/pull/512))

## v0.11.0 (2020-11-30)

Incompatible changes:

- Renamed strategy methods ([#486](https://github.com/adap/flower/pull/486)) to unify the naming of Flower's public APIs. Other public methods/functions (e.g., every method in `Client`, but also `Strategy.evaluate`) do not use the `on_` prefix, which is why we're removing it from the four methods in Strategy. To migrate rename the following `Strategy` methods accordingly:
    - `on_configure_evaluate` => `configure_evaluate`
    - `on_aggregate_evaluate` => `aggregate_evaluate`
    - `on_configure_fit` => `configure_fit`
    - `on_aggregate_fit` => `aggregate_fit`

Important changes:

- Deprecated `DefaultStrategy` ([#479](https://github.com/adap/flower/pull/479)). To migrate use `FedAvg` instead.
- Simplified examples and baselines ([#484](https://github.com/adap/flower/pull/484)).
- Removed presently unused `on_conclude_round` from strategy interface ([#483](https://github.com/adap/flower/pull/483)).
- Set minimal Python version to 3.6.1 instead of 3.6.9 ([#471](https://github.com/adap/flower/pull/471)).
- Improved `Strategy` docstrings ([#470](https://github.com/adap/flower/pull/470)).
