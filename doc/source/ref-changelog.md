# Changelog

## v1.13.1 (2024-11-26)

### Thanks to our contributors

We would like to give our special thanks to all the contributors who made the new version of Flower possible (in `git shortlog` order):

`Adam Narozniak`, `Charles Beauville`, `Heng Pan`, `Javier`, `Robert Steiner` <!---TOKEN_v1.13.1-->

### What's new?

- **Fix `SimulationEngine` Executor for SuperLink** ([#4563](https://github.com/adap/flower/pull/4563), [#4568](https://github.com/adap/flower/pull/4568), [#4570](https://github.com/adap/flower/pull/4570))

  Resolved an issue that prevented SuperLink from functioning correctly when using the `SimulationEngine` executor.

- **Improve FAB build and install** ([#4571](https://github.com/adap/flower/pull/4571))

  An updated FAB build and install process produces smaller FAB files and doesn't rely on `pip install` any more. It also resolves an issue where all files were unnecessarily included in the FAB file. The `flwr` CLI commands now correctly pack only the necessary files, such as `.md`, `.toml` and `.py`, ensuring more efficient and accurate packaging.

- **Update** `embedded-devices` **example** ([#4381](https://github.com/adap/flower/pull/4381))

  The example now uses the `flwr run` command and the Deployment Engine.

- **Update Documentation** ([#4566](https://github.com/adap/flower/pull/4566), [#4569](https://github.com/adap/flower/pull/4569), [#4560](https://github.com/adap/flower/pull/4560), [#4556](https://github.com/adap/flower/pull/4556), [#4581](https://github.com/adap/flower/pull/4581), [#4537](https://github.com/adap/flower/pull/4537), [#4562](https://github.com/adap/flower/pull/4562), [#4582](https://github.com/adap/flower/pull/4582))

  Enhanced documentation across various aspects, including updates to translation workflows, Docker-related READMEs, and recommended datasets. Improvements also include formatting fixes for dataset partitioning docs and better references to resources in the datasets documentation index.

- **Update Infrastructure and CI/CD** ([#4577](https://github.com/adap/flower/pull/4577), [#4578](https://github.com/adap/flower/pull/4578), [#4558](https://github.com/adap/flower/pull/4558), [#4551](https://github.com/adap/flower/pull/4551), [#3356](https://github.com/adap/flower/pull/3356), [#4559](https://github.com/adap/flower/pull/4559), [#4575](https://github.com/adap/flower/pull/4575))

- **General improvements** ([#4557](https://github.com/adap/flower/pull/4557), [#4564](https://github.com/adap/flower/pull/4564), [#4573](https://github.com/adap/flower/pull/4573), [#4561](https://github.com/adap/flower/pull/4561), [#4579](https://github.com/adap/flower/pull/4579), [#4572](https://github.com/adap/flower/pull/4572))

  As always, many parts of the Flower framework and quality infrastructure were improved and updated.

## v1.13.0 (2024-11-20)

### Thanks to our contributors

We would like to give our special thanks to all the contributors who made the new version of Flower possible (in `git shortlog` order):

`Adam Narozniak`, `Charles Beauville`, `Chong Shen Ng`, `Daniel J. Beutel`, `Daniel Nata Nugraha`, `Dimitris Stripelis`, `Heng Pan`, `Javier`, `Mohammad Naseri`, `Robert Steiner`, `Waris Gill`, `William Lindskog`, `Yan Gao`, `Yao Xu`, `wwjang` <!---TOKEN_v1.13.0-->

### What's new?

- **Introduce `flwr ls` command** ([#4460](https://github.com/adap/flower/pull/4460), [#4459](https://github.com/adap/flower/pull/4459), [#4477](https://github.com/adap/flower/pull/4477))

  The `flwr ls` command is now available to display details about all runs (or one specific run). It supports the following usage options:

  - `flwr ls --runs [<app>] [<federation>]`: Lists all runs.
  - `flwr ls --run-id <run-id> [<app>] [<federation>]`: Displays details for a specific run.

  This command provides information including the run ID, FAB ID and version, run status, elapsed time, and timestamps for when the run was created, started running, and finished.

- **Fuse SuperLink and SuperExec** ([#4358](https://github.com/adap/flower/pull/4358), [#4403](https://github.com/adap/flower/pull/4403), [#4406](https://github.com/adap/flower/pull/4406), [#4357](https://github.com/adap/flower/pull/4357), [#4359](https://github.com/adap/flower/pull/4359), [#4354](https://github.com/adap/flower/pull/4354), [#4229](https://github.com/adap/flower/pull/4229), [#4283](https://github.com/adap/flower/pull/4283), [#4352](https://github.com/adap/flower/pull/4352))

  SuperExec has been integrated into SuperLink, enabling SuperLink to directly manage ServerApp processes (`flwr-serverapp`). The `flwr` CLI now targets SuperLink's Exec API. Additionally, SuperLink introduces two isolation modes for running ServerApps: `subprocess` (default) and `process`, which can be specified using the `--isolation {subprocess,process}` flag.

- **Introduce `flwr-serverapp` command** ([#4394](https://github.com/adap/flower/pull/4394), [#4370](https://github.com/adap/flower/pull/4370), [#4367](https://github.com/adap/flower/pull/4367), [#4350](https://github.com/adap/flower/pull/4350), [#4364](https://github.com/adap/flower/pull/4364), [#4400](https://github.com/adap/flower/pull/4400), [#4363](https://github.com/adap/flower/pull/4363), [#4401](https://github.com/adap/flower/pull/4401), [#4388](https://github.com/adap/flower/pull/4388), [#4402](https://github.com/adap/flower/pull/4402))

  The `flwr-serverapp` command has been introduced as a CLI entry point that runs a `ServerApp` process. This process communicates with SuperLink to load and execute the `ServerApp` object, enabling isolated execution and more flexible deployment.

- **Improve simulation engine and introduce `flwr-simulation` command** ([#4433](https://github.com/adap/flower/pull/4433), [#4486](https://github.com/adap/flower/pull/4486), [#4448](https://github.com/adap/flower/pull/4448), [#4427](https://github.com/adap/flower/pull/4427), [#4438](https://github.com/adap/flower/pull/4438), [#4421](https://github.com/adap/flower/pull/4421), [#4430](https://github.com/adap/flower/pull/4430), [#4462](https://github.com/adap/flower/pull/4462))

  The simulation engine has been significantly improved, resulting in dramatically faster simulations. Additionally, the `flwr-simulation` command has been introduced to enhance maintainability and provide a dedicated entry point for running simulations.

- **Improve SuperLink message management** ([#4378](https://github.com/adap/flower/pull/4378), [#4369](https://github.com/adap/flower/pull/4369))

  SuperLink now validates the destination node ID of instruction messages and checks the TTL (time-to-live) for reply messages. When pulling reply messages, an error reply will be generated and returned if the corresponding instruction message does not exist, has expired, or if the reply message exists but has expired.

- **Introduce FedDebug baseline** ([#3783](https://github.com/adap/flower/pull/3783))

  FedDebug is a framework that enhances debugging in Federated Learning by enabling interactive inspection of the training process and automatically identifying clients responsible for degrading the global model's performance—all without requiring testing data or labels. Learn more in the [FedDebug baseline documentation](https://flower.ai/docs/baselines/feddebug.html).

- **Update documentation** ([#4511](https://github.com/adap/flower/pull/4511), [#4010](https://github.com/adap/flower/pull/4010), [#4396](https://github.com/adap/flower/pull/4396), [#4499](https://github.com/adap/flower/pull/4499), [#4269](https://github.com/adap/flower/pull/4269), [#3340](https://github.com/adap/flower/pull/3340), [#4482](https://github.com/adap/flower/pull/4482), [#4387](https://github.com/adap/flower/pull/4387), [#4342](https://github.com/adap/flower/pull/4342), [#4492](https://github.com/adap/flower/pull/4492), [#4474](https://github.com/adap/flower/pull/4474), [#4500](https://github.com/adap/flower/pull/4500), [#4514](https://github.com/adap/flower/pull/4514), [#4236](https://github.com/adap/flower/pull/4236), [#4112](https://github.com/adap/flower/pull/4112), [#3367](https://github.com/adap/flower/pull/3367), [#4501](https://github.com/adap/flower/pull/4501), [#4373](https://github.com/adap/flower/pull/4373), [#4409](https://github.com/adap/flower/pull/4409), [#4356](https://github.com/adap/flower/pull/4356), [#4520](https://github.com/adap/flower/pull/4520), [#4524](https://github.com/adap/flower/pull/4524), [#4525](https://github.com/adap/flower/pull/4525), [#4526](https://github.com/adap/flower/pull/4526), [#4527](https://github.com/adap/flower/pull/4527), [#4528](https://github.com/adap/flower/pull/4528), [#4545](https://github.com/adap/flower/pull/4545), [#4522](https://github.com/adap/flower/pull/4522), [#4534](https://github.com/adap/flower/pull/4534), [#4513](https://github.com/adap/flower/pull/4513), [#4529](https://github.com/adap/flower/pull/4529), [#4441](https://github.com/adap/flower/pull/4441), [#4530](https://github.com/adap/flower/pull/4530), [#4470](https://github.com/adap/flower/pull/4470), [#4553](https://github.com/adap/flower/pull/4553), [#4531](https://github.com/adap/flower/pull/4531), [#4554](https://github.com/adap/flower/pull/4554), [#4555](https://github.com/adap/flower/pull/4555), [#4552](https://github.com/adap/flower/pull/4552), [#4533](https://github.com/adap/flower/pull/4533))

  Many documentation pages and tutorials have been updated to improve clarity, fix typos, incorporate user feedback, and stay aligned with the latest features in the framework. Key updates include adding a guide for designing stateful `ClientApp` objects, updating the comprehensive guide for setting up and running Flower's `Simulation Engine`, updating the XGBoost, scikit-learn, and JAX quickstart tutorials to use `flwr run`, updating DP guide, removing outdated pages, updating Docker docs, and marking legacy functions as deprecated. The [Secure Aggregation Protocols](https://flower.ai/docs/framework/contributor-ref-secure-aggregation-protocols.html) page has also been updated.

- **Update examples and templates** ([#4510](https://github.com/adap/flower/pull/4510), [#4368](https://github.com/adap/flower/pull/4368), [#4121](https://github.com/adap/flower/pull/4121), [#4329](https://github.com/adap/flower/pull/4329), [#4382](https://github.com/adap/flower/pull/4382), [#4248](https://github.com/adap/flower/pull/4248), [#4395](https://github.com/adap/flower/pull/4395), [#4386](https://github.com/adap/flower/pull/4386), [#4408](https://github.com/adap/flower/pull/4408))

  Multiple examples and templates have been updated to enhance usability and correctness. The updates include the `30-minute-tutorial`, `quickstart-jax`, `quickstart-pytorch`, `advanced-tensorflow` examples, and the FlowerTune template.

- **Improve Docker support** ([#4506](https://github.com/adap/flower/pull/4506), [#4424](https://github.com/adap/flower/pull/4424), [#4224](https://github.com/adap/flower/pull/4224), [#4413](https://github.com/adap/flower/pull/4413), [#4414](https://github.com/adap/flower/pull/4414), [#4336](https://github.com/adap/flower/pull/4336), [#4420](https://github.com/adap/flower/pull/4420), [#4407](https://github.com/adap/flower/pull/4407), [#4422](https://github.com/adap/flower/pull/4422), [#4532](https://github.com/adap/flower/pull/4532), [#4540](https://github.com/adap/flower/pull/4540))

  Docker images and configurations have been updated, including updating Docker Compose files to version 1.13.0, refactoring the Docker build matrix for better maintainability, updating `docker/build-push-action` to 6.9.0, and improving Docker documentation.

- **Allow app installation without internet access** ([#4479](https://github.com/adap/flower/pull/4479), [#4475](https://github.com/adap/flower/pull/4475))

  The `flwr build` command now includes a wheel file in the FAB, enabling Flower app installation in environments without internet access via `flwr install`.

- **Improve `flwr log` command** ([#4391](https://github.com/adap/flower/pull/4391), [#4411](https://github.com/adap/flower/pull/4411), [#4390](https://github.com/adap/flower/pull/4390), [#4397](https://github.com/adap/flower/pull/4397))

- **Refactor SuperNode for better maintainability and efficiency** ([#4439](https://github.com/adap/flower/pull/4439), [#4348](https://github.com/adap/flower/pull/4348), [#4512](https://github.com/adap/flower/pull/4512), [#4485](https://github.com/adap/flower/pull/4485))

- **Support NumPy `2.0`** ([#4440](https://github.com/adap/flower/pull/4440))

- **Update infrastructure and CI/CD** ([#4466](https://github.com/adap/flower/pull/4466), [#4419](https://github.com/adap/flower/pull/4419), [#4338](https://github.com/adap/flower/pull/4338), [#4334](https://github.com/adap/flower/pull/4334), [#4456](https://github.com/adap/flower/pull/4456), [#4446](https://github.com/adap/flower/pull/4446), [#4415](https://github.com/adap/flower/pull/4415))

- **Bugfixes** ([#4404](https://github.com/adap/flower/pull/4404), [#4518](https://github.com/adap/flower/pull/4518), [#4452](https://github.com/adap/flower/pull/4452), [#4376](https://github.com/adap/flower/pull/4376), [#4493](https://github.com/adap/flower/pull/4493), [#4436](https://github.com/adap/flower/pull/4436), [#4410](https://github.com/adap/flower/pull/4410), [#4442](https://github.com/adap/flower/pull/4442), [#4375](https://github.com/adap/flower/pull/4375), [#4515](https://github.com/adap/flower/pull/4515))

- **General improvements** ([#4454](https://github.com/adap/flower/pull/4454), [#4365](https://github.com/adap/flower/pull/4365), [#4423](https://github.com/adap/flower/pull/4423), [#4516](https://github.com/adap/flower/pull/4516), [#4509](https://github.com/adap/flower/pull/4509), [#4498](https://github.com/adap/flower/pull/4498), [#4371](https://github.com/adap/flower/pull/4371), [#4449](https://github.com/adap/flower/pull/4449), [#4488](https://github.com/adap/flower/pull/4488), [#4478](https://github.com/adap/flower/pull/4478), [#4392](https://github.com/adap/flower/pull/4392), [#4483](https://github.com/adap/flower/pull/4483), [#4517](https://github.com/adap/flower/pull/4517), [#4330](https://github.com/adap/flower/pull/4330), [#4458](https://github.com/adap/flower/pull/4458), [#4347](https://github.com/adap/flower/pull/4347), [#4429](https://github.com/adap/flower/pull/4429), [#4463](https://github.com/adap/flower/pull/4463), [#4496](https://github.com/adap/flower/pull/4496), [#4508](https://github.com/adap/flower/pull/4508), [#4444](https://github.com/adap/flower/pull/4444), [#4417](https://github.com/adap/flower/pull/4417), [#4504](https://github.com/adap/flower/pull/4504), [#4418](https://github.com/adap/flower/pull/4418), [#4480](https://github.com/adap/flower/pull/4480), [#4455](https://github.com/adap/flower/pull/4455), [#4468](https://github.com/adap/flower/pull/4468), [#4385](https://github.com/adap/flower/pull/4385), [#4487](https://github.com/adap/flower/pull/4487), [#4393](https://github.com/adap/flower/pull/4393), [#4489](https://github.com/adap/flower/pull/4489), [#4389](https://github.com/adap/flower/pull/4389), [#4507](https://github.com/adap/flower/pull/4507), [#4469](https://github.com/adap/flower/pull/4469), [#4340](https://github.com/adap/flower/pull/4340), [#4353](https://github.com/adap/flower/pull/4353), [#4494](https://github.com/adap/flower/pull/4494), [#4461](https://github.com/adap/flower/pull/4461), [#4362](https://github.com/adap/flower/pull/4362), [#4473](https://github.com/adap/flower/pull/4473), [#4405](https://github.com/adap/flower/pull/4405), [#4416](https://github.com/adap/flower/pull/4416), [#4453](https://github.com/adap/flower/pull/4453), [#4491](https://github.com/adap/flower/pull/4491), [#4539](https://github.com/adap/flower/pull/4539), [#4542](https://github.com/adap/flower/pull/4542), [#4538](https://github.com/adap/flower/pull/4538), [#4543](https://github.com/adap/flower/pull/4543), [#4541](https://github.com/adap/flower/pull/4541), [#4550](https://github.com/adap/flower/pull/4550), [#4481](https://github.com/adap/flower/pull/4481))

  As always, many parts of the Flower framework and quality infrastructure were improved and updated.

### Deprecations

- **Deprecate Python 3.9**

  Flower is deprecating support for Python 3.9 as several of its dependencies are phasing out compatibility with this version. While no immediate changes have been made, users are encouraged to plan for upgrading to a supported Python version.

### Incompatible changes

- **Remove `flower-superexec` command** ([#4351](https://github.com/adap/flower/pull/4351))

  The `flower-superexec` command, previously used to launch SuperExec, is no longer functional as SuperExec has been merged into SuperLink. Starting an additional SuperExec is no longer necessary when SuperLink is initiated.

- **Remove `flower-server-app` command** ([#4490](https://github.com/adap/flower/pull/4490))

  The `flower-server-app` command has been removed. To start a Flower app, please use the `flwr run` command instead.

- **Remove `app` argument from `flower-supernode` command** ([#4497](https://github.com/adap/flower/pull/4497))

  The usage of `flower-supernode <app-dir>` has been removed. SuperNode will now load the FAB delivered by SuperLink, and it is no longer possible to directly specify an app directory.

- **Remove support for non-app simulations** ([#4431](https://github.com/adap/flower/pull/4431))

  The simulation engine (via `flower-simulation`) now exclusively supports passing an app.

- **Rename CLI arguments for `flower-superlink` command** ([#4412](https://github.com/adap/flower/pull/4412))

  The `--driver-api-address` argument has been renamed to `--serverappio-api-address` in the `flower-superlink` command to reflect the renaming of the `Driver` service to the `ServerAppIo` service.

- **Rename CLI arguments for `flwr-serverapp` and `flwr-clientapp` commands** ([#4495](https://github.com/adap/flower/pull/4495))

  The CLI arguments have been renamed for clarity and consistency. Specifically, `--superlink` for `flwr-serverapp` is now `--serverappio-api-address`, and `--supernode` for `flwr-clientapp` is now `--clientappio-api-address`.

## v1.12.0 (2024-10-14)

### Thanks to our contributors

We would like to give our special thanks to all the contributors who made the new version of Flower possible (in `git shortlog` order):

`Adam Narozniak`, `Audris`, `Charles Beauville`, `Chong Shen Ng`, `Daniel J. Beutel`, `Daniel Nata Nugraha`, `Heng Pan`, `Javier`, `Jiahao Tan`, `Julian Rußmeyer`, `Mohammad Naseri`, `Ray Sun`, `Robert Steiner`, `Yan Gao`, `xiliguguagua` <!---TOKEN_v1.12.0-->

### What's new?

- **Introduce SuperExec log streaming** ([#3577](https://github.com/adap/flower/pull/3577), [#3584](https://github.com/adap/flower/pull/3584), [#4242](https://github.com/adap/flower/pull/4242), [#3611](https://github.com/adap/flower/pull/3611), [#3613](https://github.com/adap/flower/pull/3613))

  Flower now supports log streaming from a remote SuperExec using the `flwr log` command. This new feature allows you to monitor logs from SuperExec in real time via `flwr log <run-id>` (or `flwr log <run-id> <app-dir> <federation>`).

- **Improve `flwr new` templates** ([#4291](https://github.com/adap/flower/pull/4291), [#4292](https://github.com/adap/flower/pull/4292), [#4293](https://github.com/adap/flower/pull/4293), [#4294](https://github.com/adap/flower/pull/4294), [#4295](https://github.com/adap/flower/pull/4295))

  The `flwr new` command templates for MLX, NumPy, sklearn, JAX, and PyTorch have been updated to improve usability and consistency across frameworks.

- **Migrate ID handling to use unsigned 64-bit integers** ([#4170](https://github.com/adap/flower/pull/4170), [#4237](https://github.com/adap/flower/pull/4237), [#4243](https://github.com/adap/flower/pull/4243))

  Node IDs, run IDs, and related fields have been migrated from signed 64-bit integers (`sint64`) to unsigned 64-bit integers (`uint64`). To support this change, the `uint64` type is fully supported in all communications. You may now use `uint64` values in config and metric dictionaries. For Python users, that means using `int` values larger than the maximum value of `sint64` but less than the maximum value of `uint64`.

- **Add Flower architecture explanation** ([#3270](https://github.com/adap/flower/pull/3270))

  A new [Flower architecture explainer](https://flower.ai/docs/framework/explanation-flower-architecture.html) page introduces Flower components step-by-step. Check out the `EXPLANATIONS` section of the Flower documentation if you're interested.

- **Introduce FedRep baseline** ([#3790](https://github.com/adap/flower/pull/3790))

  FedRep is a federated learning algorithm that learns shared data representations across clients while allowing each to maintain personalized local models, balancing collaboration and individual adaptation. Read all the details in the paper: "Exploiting Shared Representations for Personalized Federated Learning" ([arxiv](https://arxiv.org/abs/2102.07078))

- **Improve FlowerTune template and LLM evaluation pipelines** ([#4286](https://github.com/adap/flower/pull/4286), [#3769](https://github.com/adap/flower/pull/3769), [#4272](https://github.com/adap/flower/pull/4272), [#4257](https://github.com/adap/flower/pull/4257), [#4220](https://github.com/adap/flower/pull/4220), [#4282](https://github.com/adap/flower/pull/4282), [#4171](https://github.com/adap/flower/pull/4171), [#4228](https://github.com/adap/flower/pull/4228), [#4258](https://github.com/adap/flower/pull/4258), [#4296](https://github.com/adap/flower/pull/4296), [#4287](https://github.com/adap/flower/pull/4287), [#4217](https://github.com/adap/flower/pull/4217), [#4249](https://github.com/adap/flower/pull/4249), [#4324](https://github.com/adap/flower/pull/4324), [#4219](https://github.com/adap/flower/pull/4219), [#4327](https://github.com/adap/flower/pull/4327))

  Refined evaluation pipelines, metrics, and documentation for the upcoming FlowerTune LLM Leaderboard across multiple domains including Finance, Medical, and general NLP. Stay tuned for the official launch—we welcome all federated learning and LLM enthusiasts to participate in this exciting challenge!

- **Enhance Docker Support and Documentation** ([#4191](https://github.com/adap/flower/pull/4191), [#4251](https://github.com/adap/flower/pull/4251), [#4190](https://github.com/adap/flower/pull/4190), [#3928](https://github.com/adap/flower/pull/3928), [#4298](https://github.com/adap/flower/pull/4298), [#4192](https://github.com/adap/flower/pull/4192), [#4136](https://github.com/adap/flower/pull/4136), [#4187](https://github.com/adap/flower/pull/4187), [#4261](https://github.com/adap/flower/pull/4261), [#4177](https://github.com/adap/flower/pull/4177), [#4176](https://github.com/adap/flower/pull/4176), [#4189](https://github.com/adap/flower/pull/4189), [#4297](https://github.com/adap/flower/pull/4297), [#4226](https://github.com/adap/flower/pull/4226))

  Upgraded Ubuntu base image to 24.04, added SBOM and gcc to Docker images, and comprehensively updated [Docker documentation](https://flower.ai/docs/framework/docker/index.html) including quickstart guides and distributed Docker Compose instructions.

- **Introduce Flower glossary** ([#4165](https://github.com/adap/flower/pull/4165), [#4235](https://github.com/adap/flower/pull/4235))

  Added the [Federated Learning glossary](https://flower.ai/glossary/) to the Flower repository, located under the `flower/glossary/` directory. This resource aims to provide clear definitions and explanations of key FL concepts. Community contributions are highly welcomed to help expand and refine this knowledge base — this is probably the easiest way to become a Flower contributor!

- **Implement Message Time-to-Live (TTL)** ([#3620](https://github.com/adap/flower/pull/3620), [#3596](https://github.com/adap/flower/pull/3596), [#3615](https://github.com/adap/flower/pull/3615), [#3609](https://github.com/adap/flower/pull/3609), [#3635](https://github.com/adap/flower/pull/3635))

  Added comprehensive TTL support for messages in Flower's SuperLink. Messages are now automatically expired and cleaned up based on configurable TTL values, available through the low-level API (and used by default in the high-level API).

- **Improve FAB handling** ([#4303](https://github.com/adap/flower/pull/4303), [#4264](https://github.com/adap/flower/pull/4264), [#4305](https://github.com/adap/flower/pull/4305), [#4304](https://github.com/adap/flower/pull/4304))

  An 8-character hash is now appended to the FAB file name. The `flwr install` command installs FABs with a more flattened folder structure, reducing it from 3 levels to 1.

- **Update documentation** ([#3341](https://github.com/adap/flower/pull/3341), [#3338](https://github.com/adap/flower/pull/3338), [#3927](https://github.com/adap/flower/pull/3927), [#4152](https://github.com/adap/flower/pull/4152), [#4151](https://github.com/adap/flower/pull/4151), [#3993](https://github.com/adap/flower/pull/3993))

  Updated quickstart tutorials (PyTorch Lightning, TensorFlow, Hugging Face, Fastai) to use the new `flwr run` command and removed default title from documentation base template. A new blockchain example has been added to FAQ.

- **Update example projects** ([#3716](https://github.com/adap/flower/pull/3716), [#4007](https://github.com/adap/flower/pull/4007), [#4130](https://github.com/adap/flower/pull/4130), [#4234](https://github.com/adap/flower/pull/4234), [#4206](https://github.com/adap/flower/pull/4206), [#4188](https://github.com/adap/flower/pull/4188), [#4247](https://github.com/adap/flower/pull/4247), [#4331](https://github.com/adap/flower/pull/4331))

  Refreshed multiple example projects including vertical FL, PyTorch (advanced), Pandas, Secure Aggregation, and XGBoost examples. Optimized Hugging Face quickstart with a smaller language model and removed legacy simulation examples.

- **Update translations** ([#4070](https://github.com/adap/flower/pull/4070), [#4316](https://github.com/adap/flower/pull/4316), [#4252](https://github.com/adap/flower/pull/4252), [#4256](https://github.com/adap/flower/pull/4256), [#4210](https://github.com/adap/flower/pull/4210), [#4263](https://github.com/adap/flower/pull/4263), [#4259](https://github.com/adap/flower/pull/4259))

- **General improvements** ([#4239](https://github.com/adap/flower/pull/4239), [4276](https://github.com/adap/flower/pull/4276), [4204](https://github.com/adap/flower/pull/4204), [4184](https://github.com/adap/flower/pull/4184), [4227](https://github.com/adap/flower/pull/4227), [4183](https://github.com/adap/flower/pull/4183), [4202](https://github.com/adap/flower/pull/4202), [4250](https://github.com/adap/flower/pull/4250), [4267](https://github.com/adap/flower/pull/4267), [4246](https://github.com/adap/flower/pull/4246), [4240](https://github.com/adap/flower/pull/4240), [4265](https://github.com/adap/flower/pull/4265), [4238](https://github.com/adap/flower/pull/4238), [4275](https://github.com/adap/flower/pull/4275), [4318](https://github.com/adap/flower/pull/4318), [#4178](https://github.com/adap/flower/pull/4178), [#4315](https://github.com/adap/flower/pull/4315), [#4241](https://github.com/adap/flower/pull/4241), [#4289](https://github.com/adap/flower/pull/4289), [#4290](https://github.com/adap/flower/pull/4290), [#4181](https://github.com/adap/flower/pull/4181), [#4208](https://github.com/adap/flower/pull/4208), [#4225](https://github.com/adap/flower/pull/4225), [#4314](https://github.com/adap/flower/pull/4314), [#4174](https://github.com/adap/flower/pull/4174), [#4203](https://github.com/adap/flower/pull/4203), [#4274](https://github.com/adap/flower/pull/4274), [#3154](https://github.com/adap/flower/pull/3154), [#4201](https://github.com/adap/flower/pull/4201), [#4268](https://github.com/adap/flower/pull/4268), [#4254](https://github.com/adap/flower/pull/4254), [#3990](https://github.com/adap/flower/pull/3990), [#4212](https://github.com/adap/flower/pull/4212), [#2938](https://github.com/adap/flower/pull/2938), [#4205](https://github.com/adap/flower/pull/4205), [#4222](https://github.com/adap/flower/pull/4222), [#4313](https://github.com/adap/flower/pull/4313), [#3936](https://github.com/adap/flower/pull/3936), [#4278](https://github.com/adap/flower/pull/4278), [#4319](https://github.com/adap/flower/pull/4319), [#4332](https://github.com/adap/flower/pull/4332), [#4333](https://github.com/adap/flower/pull/4333))

  As always, many parts of the Flower framework and quality infrastructure were improved and updated.

### Incompatible changes

- **Drop Python 3.8 support and update minimum version to 3.9** ([#4180](https://github.com/adap/flower/pull/4180), [#4213](https://github.com/adap/flower/pull/4213), [#4193](https://github.com/adap/flower/pull/4193), [#4199](https://github.com/adap/flower/pull/4199), [#4196](https://github.com/adap/flower/pull/4196), [#4195](https://github.com/adap/flower/pull/4195), [#4198](https://github.com/adap/flower/pull/4198), [#4194](https://github.com/adap/flower/pull/4194))

  Python 3.8 support was deprecated in Flower 1.9, and this release removes support. Flower now requires Python 3.9 or later (Python 3.11 is recommended). CI and documentation were updated to use Python 3.9 as the minimum supported version. Flower now supports Python 3.9 to 3.12.

## v1.11.1 (2024-09-11)

### Thanks to our contributors

We would like to give our special thanks to all the contributors who made the new version of Flower possible (in `git shortlog` order):

`Charles Beauville`, `Chong Shen Ng`, `Daniel J. Beutel`, `Heng Pan`, `Javier`, `Robert Steiner`, `Yan Gao` <!---TOKEN_v1.11.1-->

### Improvements

- **Implement** `keys/values/items` **methods for** `TypedDict` ([#4146](https://github.com/adap/flower/pull/4146))

- **Fix parsing of** `--executor-config` **if present** ([#4125](https://github.com/adap/flower/pull/4125))

- **Adjust framework name in templates docstrings** ([#4127](https://github.com/adap/flower/pull/4127))

- **Update** `flwr new` **Hugging Face template** ([#4169](https://github.com/adap/flower/pull/4169))

- **Fix** `flwr new` **FlowerTune template** ([#4123](https://github.com/adap/flower/pull/4123))

- **Add buffer time after** `ServerApp` **thread initialization** ([#4119](https://github.com/adap/flower/pull/4119))

- **Handle unsuitable resources for simulation** ([#4143](https://github.com/adap/flower/pull/4143))

- **Update example READMEs** ([#4117](https://github.com/adap/flower/pull/4117))

- **Update SuperNode authentication docs** ([#4160](https://github.com/adap/flower/pull/4160))

### Incompatible changes

None

## v1.11.0 (2024-08-30)

### Thanks to our contributors

We would like to give our special thanks to all the contributors who made the new version of Flower possible (in `git shortlog` order):

`Adam Narozniak`, `Charles Beauville`, `Chong Shen Ng`, `Daniel J. Beutel`, `Daniel Nata Nugraha`, `Danny`, `Edoardo Gabrielli`, `Heng Pan`, `Javier`, `Meng Yan`, `Michal Danilowski`, `Mohammad Naseri`, `Robert Steiner`, `Steve Laskaridis`, `Taner Topal`, `Yan Gao` <!---TOKEN_v1.11.0-->

### What's new?

- **Deliver Flower App Bundle (FAB) to SuperLink and SuperNodes** ([#4006](https://github.com/adap/flower/pull/4006), [#3945](https://github.com/adap/flower/pull/3945), [#3999](https://github.com/adap/flower/pull/3999), [#4027](https://github.com/adap/flower/pull/4027), [#3851](https://github.com/adap/flower/pull/3851), [#3946](https://github.com/adap/flower/pull/3946), [#4003](https://github.com/adap/flower/pull/4003), [#4029](https://github.com/adap/flower/pull/4029), [#3942](https://github.com/adap/flower/pull/3942), [#3957](https://github.com/adap/flower/pull/3957), [#4020](https://github.com/adap/flower/pull/4020), [#4044](https://github.com/adap/flower/pull/4044), [#3852](https://github.com/adap/flower/pull/3852), [#4019](https://github.com/adap/flower/pull/4019), [#4031](https://github.com/adap/flower/pull/4031), [#4036](https://github.com/adap/flower/pull/4036), [#4049](https://github.com/adap/flower/pull/4049), [#4017](https://github.com/adap/flower/pull/4017), [#3943](https://github.com/adap/flower/pull/3943), [#3944](https://github.com/adap/flower/pull/3944), [#4011](https://github.com/adap/flower/pull/4011), [#3619](https://github.com/adap/flower/pull/3619))

  Dynamic code updates are here! `flwr run` can now ship and install the latest version of your `ServerApp` and `ClientApp` to an already-running federation (SuperLink and SuperNodes).

  How does it work? `flwr run` bundles your Flower app into a single FAB (Flower App Bundle) file. It then ships this FAB file, via the SuperExec, to both the SuperLink and those SuperNodes that need it. This allows you to keep SuperExec, SuperLink and SuperNodes running as permanent infrastructure, and then ship code updates (including completely new projects!) dynamically.

  `flwr run` is all you need.

- **Introduce isolated** `ClientApp` **execution** ([#3970](https://github.com/adap/flower/pull/3970), [#3976](https://github.com/adap/flower/pull/3976), [#4002](https://github.com/adap/flower/pull/4002), [#4001](https://github.com/adap/flower/pull/4001), [#4034](https://github.com/adap/flower/pull/4034), [#4037](https://github.com/adap/flower/pull/4037), [#3977](https://github.com/adap/flower/pull/3977), [#4042](https://github.com/adap/flower/pull/4042), [#3978](https://github.com/adap/flower/pull/3978), [#4039](https://github.com/adap/flower/pull/4039), [#4033](https://github.com/adap/flower/pull/4033), [#3971](https://github.com/adap/flower/pull/3971), [#4035](https://github.com/adap/flower/pull/4035), [#3973](https://github.com/adap/flower/pull/3973), [#4032](https://github.com/adap/flower/pull/4032))

  The SuperNode can now run your `ClientApp` in a fully isolated way. In an enterprise deployment, this allows you to set strict limits on what the `ClientApp` can and cannot do.

  `flower-supernode` supports three `--isolation` modes:

  - Unset: The SuperNode runs the `ClientApp` in the same process (as in previous versions of Flower). This is the default mode.
  - `--isolation=subprocess`: The SuperNode starts a subprocess to run the `ClientApp`.
  - `--isolation=process`: The SuperNode expects an externally-managed process to run the `ClientApp`. This external process is not managed by the SuperNode, so it has to be started beforehand and terminated manually. The common way to use this isolation mode is via the new `flwr/clientapp` Docker image.

- **Improve Docker support for enterprise deployments** ([#4050](https://github.com/adap/flower/pull/4050), [#4090](https://github.com/adap/flower/pull/4090), [#3784](https://github.com/adap/flower/pull/3784), [#3998](https://github.com/adap/flower/pull/3998), [#4094](https://github.com/adap/flower/pull/4094), [#3722](https://github.com/adap/flower/pull/3722))

  Flower 1.11 ships many Docker improvements that are especially useful for enterprise deployments:

  - `flwr/supernode` comes with a new Alpine Docker image.
  - `flwr/clientapp` is a new image to be used with the `--isolation=process` option. In this mode, SuperNode and `ClientApp` run in two different Docker containers. `flwr/supernode` (preferably the Alpine version) runs the long-running SuperNode with `--isolation=process`. `flwr/clientapp` runs the `ClientApp`. This is the recommended way to deploy Flower in enterprise settings.
  - New all-in-one Docker Compose enables you to easily start a full Flower Deployment Engine on a single machine.
  - Completely new Docker documentation: https://flower.ai/docs/framework/docker/index.html

- **Improve SuperNode authentication** ([#4043](https://github.com/adap/flower/pull/4043), [#4047](https://github.com/adap/flower/pull/4047), [#4074](https://github.com/adap/flower/pull/4074))

  SuperNode auth has been improved in several ways, including improved logging, improved testing, and improved error handling.

- **Update** `flwr new` **templates** ([#3933](https://github.com/adap/flower/pull/3933), [#3894](https://github.com/adap/flower/pull/3894), [#3930](https://github.com/adap/flower/pull/3930), [#3931](https://github.com/adap/flower/pull/3931), [#3997](https://github.com/adap/flower/pull/3997), [#3979](https://github.com/adap/flower/pull/3979), [#3965](https://github.com/adap/flower/pull/3965), [#4013](https://github.com/adap/flower/pull/4013), [#4064](https://github.com/adap/flower/pull/4064))

  All `flwr new` templates have been updated to show the latest recommended use of Flower APIs.

- **Improve Simulation Engine** ([#4095](https://github.com/adap/flower/pull/4095), [#3913](https://github.com/adap/flower/pull/3913), [#4059](https://github.com/adap/flower/pull/4059), [#3954](https://github.com/adap/flower/pull/3954), [#4071](https://github.com/adap/flower/pull/4071), [#3985](https://github.com/adap/flower/pull/3985), [#3988](https://github.com/adap/flower/pull/3988))

  The Flower Simulation Engine comes with several updates, including improved run config support, verbose logging, simulation backend configuration via `flwr run`, and more.

- **Improve** `RecordSet` ([#4052](https://github.com/adap/flower/pull/4052), [#3218](https://github.com/adap/flower/pull/3218), [#4016](https://github.com/adap/flower/pull/4016))

  `RecordSet` is the core object to exchange model parameters, configuration values and metrics between `ClientApp` and `ServerApp`. This release ships several smaller improvements to `RecordSet` and related `*Record` types.

- **Update documentation** ([#3972](https://github.com/adap/flower/pull/3972), [#3925](https://github.com/adap/flower/pull/3925), [#4061](https://github.com/adap/flower/pull/4061), [#3984](https://github.com/adap/flower/pull/3984), [#3917](https://github.com/adap/flower/pull/3917), [#3900](https://github.com/adap/flower/pull/3900), [#4066](https://github.com/adap/flower/pull/4066), [#3765](https://github.com/adap/flower/pull/3765), [#4021](https://github.com/adap/flower/pull/4021), [#3906](https://github.com/adap/flower/pull/3906), [#4063](https://github.com/adap/flower/pull/4063), [#4076](https://github.com/adap/flower/pull/4076), [#3920](https://github.com/adap/flower/pull/3920), [#3916](https://github.com/adap/flower/pull/3916))

  Many parts of the documentation, including the main tutorial, have been migrated to show new Flower APIs and other new Flower features like the improved Docker support.

- **Migrate code example to use new Flower APIs** ([#3758](https://github.com/adap/flower/pull/3758), [#3701](https://github.com/adap/flower/pull/3701), [#3919](https://github.com/adap/flower/pull/3919), [#3918](https://github.com/adap/flower/pull/3918), [#3934](https://github.com/adap/flower/pull/3934), [#3893](https://github.com/adap/flower/pull/3893), [#3833](https://github.com/adap/flower/pull/3833), [#3922](https://github.com/adap/flower/pull/3922), [#3846](https://github.com/adap/flower/pull/3846), [#3777](https://github.com/adap/flower/pull/3777), [#3874](https://github.com/adap/flower/pull/3874), [#3873](https://github.com/adap/flower/pull/3873), [#3935](https://github.com/adap/flower/pull/3935), [#3754](https://github.com/adap/flower/pull/3754), [#3980](https://github.com/adap/flower/pull/3980), [#4089](https://github.com/adap/flower/pull/4089), [#4046](https://github.com/adap/flower/pull/4046), [#3314](https://github.com/adap/flower/pull/3314), [#3316](https://github.com/adap/flower/pull/3316), [#3295](https://github.com/adap/flower/pull/3295), [#3313](https://github.com/adap/flower/pull/3313))

  Many code examples have been migrated to use new Flower APIs.

- **Update Flower framework, framework internals and quality infrastructure** ([#4018](https://github.com/adap/flower/pull/4018), [#4053](https://github.com/adap/flower/pull/4053), [#4098](https://github.com/adap/flower/pull/4098), [#4067](https://github.com/adap/flower/pull/4067), [#4105](https://github.com/adap/flower/pull/4105), [#4048](https://github.com/adap/flower/pull/4048), [#4107](https://github.com/adap/flower/pull/4107), [#4069](https://github.com/adap/flower/pull/4069), [#3915](https://github.com/adap/flower/pull/3915), [#4101](https://github.com/adap/flower/pull/4101), [#4108](https://github.com/adap/flower/pull/4108), [#3914](https://github.com/adap/flower/pull/3914), [#4068](https://github.com/adap/flower/pull/4068), [#4041](https://github.com/adap/flower/pull/4041), [#4040](https://github.com/adap/flower/pull/4040), [#3986](https://github.com/adap/flower/pull/3986), [#4026](https://github.com/adap/flower/pull/4026), [#3961](https://github.com/adap/flower/pull/3961), [#3975](https://github.com/adap/flower/pull/3975), [#3983](https://github.com/adap/flower/pull/3983), [#4091](https://github.com/adap/flower/pull/4091), [#3982](https://github.com/adap/flower/pull/3982), [#4079](https://github.com/adap/flower/pull/4079), [#4073](https://github.com/adap/flower/pull/4073), [#4060](https://github.com/adap/flower/pull/4060), [#4106](https://github.com/adap/flower/pull/4106), [#4080](https://github.com/adap/flower/pull/4080), [#3974](https://github.com/adap/flower/pull/3974), [#3996](https://github.com/adap/flower/pull/3996), [#3991](https://github.com/adap/flower/pull/3991), [#3981](https://github.com/adap/flower/pull/3981), [#4093](https://github.com/adap/flower/pull/4093), [#4100](https://github.com/adap/flower/pull/4100), [#3939](https://github.com/adap/flower/pull/3939), [#3955](https://github.com/adap/flower/pull/3955), [#3940](https://github.com/adap/flower/pull/3940), [#4038](https://github.com/adap/flower/pull/4038))

  As always, many parts of the Flower framework and quality infrastructure were improved and updated.

### Deprecations

- **Deprecate accessing `Context` via `Client.context`** ([#3797](https://github.com/adap/flower/pull/3797))

  Now that both `client_fn` and `server_fn` receive a `Context` object, accessing `Context` via `Client.context` is deprecated. `Client.context` will be removed in a future release. If you need to access `Context` in your `Client` implementation, pass it manually when creating the `Client` instance in `client_fn`:

  ```python
  def client_fn(context: Context) -> Client:
      return FlowerClient(context).to_client()
  ```

### Incompatible changes

- **Update CLIs to accept an app directory instead of** `ClientApp` **and** `ServerApp` ([#3952](https://github.com/adap/flower/pull/3952), [#4077](https://github.com/adap/flower/pull/4077), [#3850](https://github.com/adap/flower/pull/3850))

  The CLI commands `flower-supernode` and `flower-server-app` now accept an app directory as argument (instead of references to a `ClientApp` or `ServerApp`). An app directory is any directory containing a `pyproject.toml` file (with the appropriate Flower config fields set). The easiest way to generate a compatible project structure is to use `flwr new`.

- **Disable** `flower-client-app` **CLI command** ([#4022](https://github.com/adap/flower/pull/4022))

  `flower-client-app` has been disabled. Use `flower-supernode` instead.

- **Use spaces instead of commas for separating config args** ([#4000](https://github.com/adap/flower/pull/4000))

  When passing configs (run config, node config) to Flower, you now need to separate key-value pairs using spaces instead of commas. For example:

  ```bash
  flwr run . --run-config "learning-rate=0.01 num_rounds=10"  # Works
  ```

  Previously, you could pass configs using commas, like this:

  ```bash
  flwr run . --run-config "learning-rate=0.01,num_rounds=10"  # Doesn't work
  ```

- **Remove** `flwr example` **CLI command** ([#4084](https://github.com/adap/flower/pull/4084))

  The experimental `flwr example` CLI command has been removed. Use `flwr new` to generate a project and then run it using `flwr run`.

## v1.10.0 (2024-07-24)

### Thanks to our contributors

We would like to give our special thanks to all the contributors who made the new version of Flower possible (in `git shortlog` order):

`Adam Narozniak`, `Charles Beauville`, `Chong Shen Ng`, `Daniel J. Beutel`, `Daniel Nata Nugraha`, `Danny`, `Gustavo Bertoli`, `Heng Pan`, `Ikko Eltociear Ashimine`, `Javier`, `Jiahao Tan`, `Mohammad Naseri`, `Robert Steiner`, `Sebastian van der Voort`, `Taner Topal`, `Yan Gao` <!---TOKEN_v1.10.0-->

### What's new?

- **Introduce** `flwr run` **(beta)** ([#3810](https://github.com/adap/flower/pull/3810), [#3826](https://github.com/adap/flower/pull/3826), [#3880](https://github.com/adap/flower/pull/3880), [#3807](https://github.com/adap/flower/pull/3807), [#3800](https://github.com/adap/flower/pull/3800), [#3814](https://github.com/adap/flower/pull/3814), [#3811](https://github.com/adap/flower/pull/3811), [#3809](https://github.com/adap/flower/pull/3809), [#3819](https://github.com/adap/flower/pull/3819))

  Flower 1.10 ships the first beta release of the new `flwr run` command. `flwr run` can run different projects using `flwr run path/to/project`, it enables you to easily switch between different federations using `flwr run . federation` and it runs your Flower project using either local simulation or the new (experimental) SuperExec service. This allows Flower to scale federatated learning from fast local simulation to large-scale production deployment, seamlessly. All projects generated with `flwr new` are immediately runnable using `flwr run`. Give it a try: use `flwr new` to generate a project and then run it using `flwr run`.

- **Introduce run config** ([#3751](https://github.com/adap/flower/pull/3751), [#3750](https://github.com/adap/flower/pull/3750), [#3845](https://github.com/adap/flower/pull/3845), [#3824](https://github.com/adap/flower/pull/3824), [#3746](https://github.com/adap/flower/pull/3746), [#3728](https://github.com/adap/flower/pull/3728), [#3730](https://github.com/adap/flower/pull/3730), [#3725](https://github.com/adap/flower/pull/3725), [#3729](https://github.com/adap/flower/pull/3729), [#3580](https://github.com/adap/flower/pull/3580), [#3578](https://github.com/adap/flower/pull/3578), [#3576](https://github.com/adap/flower/pull/3576), [#3798](https://github.com/adap/flower/pull/3798), [#3732](https://github.com/adap/flower/pull/3732), [#3815](https://github.com/adap/flower/pull/3815))

  The new run config feature allows you to run your Flower project in different configurations without having to change a single line of code. You can now build a configurable `ServerApp` and `ClientApp` that read configuration values at runtime. This enables you to specify config values like `learning-rate=0.01` in `pyproject.toml` (under the `[tool.flwr.app.config]` key). These config values can then be easily overridden via `flwr run --run-config learning-rate=0.02`, and read from `Context` using `lr = context.run_config["learning-rate"]`. Create a new project using `flwr new` to see run config in action.

- **Generalize** `client_fn` **signature to** `client_fn(context: Context) -> Client` ([#3779](https://github.com/adap/flower/pull/3779), [#3697](https://github.com/adap/flower/pull/3697), [#3694](https://github.com/adap/flower/pull/3694), [#3696](https://github.com/adap/flower/pull/3696))

  The `client_fn` signature has been generalized to `client_fn(context: Context) -> Client`. It now receives a `Context` object instead of the (now depreacated) `cid: str`. `Context` allows accessing `node_id`, `node_config` and `run_config`, among other things. This enables you to build a configurable `ClientApp` that leverages the new run config system.

  The previous signature `client_fn(cid: str)` is now deprecated and support for it will be removed in a future release. Use `client_fn(context: Context) -> Client` everywhere.

- **Introduce new** `server_fn(context)` ([#3773](https://github.com/adap/flower/pull/3773), [#3796](https://github.com/adap/flower/pull/3796), [#3771](https://github.com/adap/flower/pull/3771))

  In addition to the new `client_fn(context:Context)`, a new `server_fn(context: Context) -> ServerAppComponents` can now be passed to `ServerApp` (instead of passing, for example, `Strategy`, directly). This enables you to leverage the full `Context` on the server-side to build a configurable `ServerApp`.

- **Relaunch all** `flwr new` **templates** ([#3877](https://github.com/adap/flower/pull/3877), [#3821](https://github.com/adap/flower/pull/3821), [#3587](https://github.com/adap/flower/pull/3587), [#3795](https://github.com/adap/flower/pull/3795), [#3875](https://github.com/adap/flower/pull/3875), [#3859](https://github.com/adap/flower/pull/3859), [#3760](https://github.com/adap/flower/pull/3760))

  All `flwr new` templates have been significantly updated to showcase new Flower features and best practices. This includes using `flwr run` and the new run config feature. You can now easily create a new project using `flwr new` and, after following the instructions to install it, `flwr run` it.

- **Introduce** `flower-supernode` **(preview)** ([#3353](https://github.com/adap/flower/pull/3353))

  The new `flower-supernode` CLI is here to replace `flower-client-app`. `flower-supernode` brings full multi-app support to the Flower client-side. It also allows to pass `--node-config` to the SuperNode, which is accessible in your `ClientApp` via `Context` (using the new `client_fn(context: Context)` signature).

- **Introduce node config** ([#3782](https://github.com/adap/flower/pull/3782), [#3780](https://github.com/adap/flower/pull/3780), [#3695](https://github.com/adap/flower/pull/3695), [#3886](https://github.com/adap/flower/pull/3886))

  A new node config feature allows you to pass a static configuration to the SuperNode. This configuration is read-only and available to every `ClientApp` running on that SuperNode. A `ClientApp` can access the node config via `Context` (`context.node_config`).

- **Introduce SuperExec (experimental)** ([#3605](https://github.com/adap/flower/pull/3605), [#3723](https://github.com/adap/flower/pull/3723), [#3731](https://github.com/adap/flower/pull/3731), [#3589](https://github.com/adap/flower/pull/3589), [#3604](https://github.com/adap/flower/pull/3604), [#3622](https://github.com/adap/flower/pull/3622), [#3838](https://github.com/adap/flower/pull/3838), [#3720](https://github.com/adap/flower/pull/3720), [#3606](https://github.com/adap/flower/pull/3606), [#3602](https://github.com/adap/flower/pull/3602), [#3603](https://github.com/adap/flower/pull/3603), [#3555](https://github.com/adap/flower/pull/3555), [#3808](https://github.com/adap/flower/pull/3808), [#3724](https://github.com/adap/flower/pull/3724), [#3658](https://github.com/adap/flower/pull/3658), [#3629](https://github.com/adap/flower/pull/3629))

  This is the first experimental release of Flower SuperExec, a new service that executes your runs. It's not ready for production deployment just yet, but don't hesitate to give it a try if you're interested.

- **Add new federated learning with tabular data example** ([#3568](https://github.com/adap/flower/pull/3568))

  A new code example exemplifies a federated learning setup using the Flower framework on the Adult Census Income tabular dataset.

- **Create generic adapter layer (preview)** ([#3538](https://github.com/adap/flower/pull/3538), [#3536](https://github.com/adap/flower/pull/3536), [#3540](https://github.com/adap/flower/pull/3540))

  A new generic gRPC adapter layer allows 3rd-party frameworks to integrate with Flower in a transparent way. This makes Flower more modular and allows for integration into other federated learning solutions and platforms.

- **Refactor Flower Simulation Engine** ([#3581](https://github.com/adap/flower/pull/3581), [#3471](https://github.com/adap/flower/pull/3471), [#3804](https://github.com/adap/flower/pull/3804), [#3468](https://github.com/adap/flower/pull/3468), [#3839](https://github.com/adap/flower/pull/3839), [#3806](https://github.com/adap/flower/pull/3806), [#3861](https://github.com/adap/flower/pull/3861), [#3543](https://github.com/adap/flower/pull/3543), [#3472](https://github.com/adap/flower/pull/3472), [#3829](https://github.com/adap/flower/pull/3829), [#3469](https://github.com/adap/flower/pull/3469))

  The Simulation Engine was significantly refactored. This results in faster and more stable simulations. It is also the foundation for upcoming changes that aim to provide the next level of performance and configurability in federated learning simulations.

- **Optimize Docker containers** ([#3591](https://github.com/adap/flower/pull/3591))

  Flower Docker containers were optimized and updated to use that latest Flower framework features.

- **Improve logging** ([#3776](https://github.com/adap/flower/pull/3776), [#3789](https://github.com/adap/flower/pull/3789))

  Improved logging aims to be more concise and helpful to show you the details you actually care about.

- **Refactor framework internals** ([#3621](https://github.com/adap/flower/pull/3621), [#3792](https://github.com/adap/flower/pull/3792), [#3772](https://github.com/adap/flower/pull/3772), [#3805](https://github.com/adap/flower/pull/3805), [#3583](https://github.com/adap/flower/pull/3583), [#3825](https://github.com/adap/flower/pull/3825), [#3597](https://github.com/adap/flower/pull/3597), [#3802](https://github.com/adap/flower/pull/3802), [#3569](https://github.com/adap/flower/pull/3569))

  As always, many parts of the Flower framework and quality infrastructure were improved and updated.

### Documentation improvements

- **Add 🇰🇷 Korean translations** ([#3680](https://github.com/adap/flower/pull/3680))

- **Update translations** ([#3586](https://github.com/adap/flower/pull/3586), [#3679](https://github.com/adap/flower/pull/3679), [#3570](https://github.com/adap/flower/pull/3570), [#3681](https://github.com/adap/flower/pull/3681), [#3617](https://github.com/adap/flower/pull/3617), [#3674](https://github.com/adap/flower/pull/3674), [#3671](https://github.com/adap/flower/pull/3671), [#3572](https://github.com/adap/flower/pull/3572), [#3631](https://github.com/adap/flower/pull/3631))

- **Update documentation** ([#3864](https://github.com/adap/flower/pull/3864), [#3688](https://github.com/adap/flower/pull/3688), [#3562](https://github.com/adap/flower/pull/3562), [#3641](https://github.com/adap/flower/pull/3641), [#3384](https://github.com/adap/flower/pull/3384), [#3634](https://github.com/adap/flower/pull/3634), [#3823](https://github.com/adap/flower/pull/3823), [#3793](https://github.com/adap/flower/pull/3793), [#3707](https://github.com/adap/flower/pull/3707))

  Updated documentation includes new install instructions for different shells, a new Flower Code Examples documentation landing page, new `flwr` CLI docs and an updated federated XGBoost code example.

### Deprecations

- **Deprecate** `client_fn(cid: str)`

  `client_fn` used to have a signature `client_fn(cid: str) -> Client`. This signature is now deprecated. Use the new signature `client_fn(context: Context) -> Client` instead. The new argument `context` allows accessing `node_id`, `node_config`, `run_config` and other `Context` features. When running using the simulation engine (or using `flower-supernode` with a custom `--node-config partition-id=...`), `context.node_config["partition-id"]` will return an `int` partition ID that can be used with Flower Datasets to load a different partition of the dataset on each simulated or deployed SuperNode.

- **Deprecate passing** `Server/ServerConfig/Strategy/ClientManager` **to** `ServerApp` **directly**

  Creating `ServerApp` using `ServerApp(config=config, strategy=strategy)` is now deprecated. Instead of passing `Server/ServerConfig/Strategy/ClientManager` to `ServerApp` directly, pass them wrapped in a `server_fn(context: Context) -> ServerAppComponents` function, like this: `ServerApp(server_fn=server_fn)`. `ServerAppComponents` can hold references to `Server/ServerConfig/Strategy/ClientManager`. In addition to that, `server_fn` allows you to access `Context` (for example, to read the `run_config`).

### Incompatible changes

- **Remove support for `client_ids` in `start_simulation`** ([#3699](https://github.com/adap/flower/pull/3699))

  The (rarely used) feature that allowed passing custom `client_ids` to the `start_simulation` function was removed. This removal is part of a bigger effort to refactor the simulation engine and unify how the Flower internals work in simulation and deployment.

- **Remove `flower-driver-api` and `flower-fleet-api`** ([#3418](https://github.com/adap/flower/pull/3418))

  The two deprecated CLI commands `flower-driver-api` and `flower-fleet-api` were removed in an effort to streamline the SuperLink developer experience. Use `flower-superlink` instead.

## v1.9.0 (2024-06-10)

### Thanks to our contributors

We would like to give our special thanks to all the contributors who made the new version of Flower possible (in `git shortlog` order):

`Adam Narozniak`, `Charles Beauville`, `Chong Shen Ng`, `Daniel J. Beutel`, `Daniel Nata Nugraha`, `Heng Pan`, `Javier`, `Mahdi Beitollahi`, `Robert Steiner`, `Taner Topal`, `Yan Gao`, `bapic`, `mohammadnaseri` <!---TOKEN_v1.9.0-->

### What's new?

- **Introduce built-in authentication (preview)** ([#2946](https://github.com/adap/flower/pull/2946), [#3388](https://github.com/adap/flower/pull/3388), [#2948](https://github.com/adap/flower/pull/2948), [#2917](https://github.com/adap/flower/pull/2917), [#3386](https://github.com/adap/flower/pull/3386), [#3308](https://github.com/adap/flower/pull/3308), [#3001](https://github.com/adap/flower/pull/3001), [#3409](https://github.com/adap/flower/pull/3409), [#2999](https://github.com/adap/flower/pull/2999), [#2979](https://github.com/adap/flower/pull/2979), [#3389](https://github.com/adap/flower/pull/3389), [#3503](https://github.com/adap/flower/pull/3503), [#3366](https://github.com/adap/flower/pull/3366), [#3357](https://github.com/adap/flower/pull/3357))

  Flower 1.9 introduces the first build-in version of client node authentication. In previous releases, users often wrote glue code to connect Flower to external authentication systems. With this release, the SuperLink can authenticate SuperNodes using a built-in authentication system. A new [how-to guide](https://flower.ai/docs/framework/how-to-authenticate-supernodes.html) and a new [code example](https://github.com/adap/flower/tree/main/examples/flower-authentication) help you to get started.

  This is the first preview release of the Flower-native authentication system. Many additional features are on the roadmap for upcoming Flower releases - stay tuned.

- **Introduce end-to-end Docker support** ([#3483](https://github.com/adap/flower/pull/3483), [#3266](https://github.com/adap/flower/pull/3266), [#3390](https://github.com/adap/flower/pull/3390), [#3283](https://github.com/adap/flower/pull/3283), [#3285](https://github.com/adap/flower/pull/3285), [#3391](https://github.com/adap/flower/pull/3391), [#3403](https://github.com/adap/flower/pull/3403), [#3458](https://github.com/adap/flower/pull/3458), [#3533](https://github.com/adap/flower/pull/3533), [#3453](https://github.com/adap/flower/pull/3453), [#3486](https://github.com/adap/flower/pull/3486), [#3290](https://github.com/adap/flower/pull/3290))

  Full Flower Next Docker support is here! With the release of Flower 1.9, Flower provides stable Docker images for the Flower SuperLink, the Flower SuperNode, and the Flower `ServerApp`. This set of images enables you to run all Flower components in Docker. Check out the new [how-to guide](https://flower.ai/docs/framework/how-to-run-flower-using-docker.html) to get stated.

- **Re-architect Flower Next simulation engine** ([#3307](https://github.com/adap/flower/pull/3307), [#3355](https://github.com/adap/flower/pull/3355), [#3272](https://github.com/adap/flower/pull/3272), [#3273](https://github.com/adap/flower/pull/3273), [#3417](https://github.com/adap/flower/pull/3417), [#3281](https://github.com/adap/flower/pull/3281), [#3343](https://github.com/adap/flower/pull/3343), [#3326](https://github.com/adap/flower/pull/3326))

  Flower Next simulations now use a new in-memory `Driver` that improves the reliability of simulations, especially in notebook environments. This is a significant step towards a complete overhaul of the Flower Next simulation architecture.

- **Upgrade simulation engine** ([#3354](https://github.com/adap/flower/pull/3354), [#3378](https://github.com/adap/flower/pull/3378), [#3262](https://github.com/adap/flower/pull/3262), [#3435](https://github.com/adap/flower/pull/3435), [#3501](https://github.com/adap/flower/pull/3501), [#3482](https://github.com/adap/flower/pull/3482), [#3494](https://github.com/adap/flower/pull/3494))

  The Flower Next simulation engine comes with improved and configurable logging. The Ray-based simulation backend in Flower 1.9 was updated to use Ray 2.10.

- **Introduce FedPFT baseline** ([#3268](https://github.com/adap/flower/pull/3268))

  FedPFT allows you to perform one-shot Federated Learning by leveraging widely available foundational models, dramatically reducing communication costs while delivering high performing models. This is work led by Mahdi Beitollahi from Huawei Noah's Ark Lab (Montreal, Canada). Read all the details in their paper: "Parametric Feature Transfer: One-shot Federated Learning with Foundation Models" ([arxiv](https://arxiv.org/abs/2402.01862))

- **Launch additional** `flwr new` **templates for Apple MLX, Hugging Face Transformers, scikit-learn and TensorFlow** ([#3291](https://github.com/adap/flower/pull/3291), [#3139](https://github.com/adap/flower/pull/3139), [#3284](https://github.com/adap/flower/pull/3284), [#3251](https://github.com/adap/flower/pull/3251), [#3376](https://github.com/adap/flower/pull/3376), [#3287](https://github.com/adap/flower/pull/3287))

  The `flwr` CLI's `flwr new` command is starting to become everone's favorite way of creating new Flower projects. This release introduces additional `flwr new` templates for Apple MLX, Hugging Face Transformers, scikit-learn and TensorFlow. In addition to that, existing templates also received updates.

- **Refine** `RecordSet` **API** ([#3209](https://github.com/adap/flower/pull/3209), [#3331](https://github.com/adap/flower/pull/3331), [#3334](https://github.com/adap/flower/pull/3334), [#3335](https://github.com/adap/flower/pull/3335), [#3375](https://github.com/adap/flower/pull/3375), [#3368](https://github.com/adap/flower/pull/3368))

  `RecordSet` is part of the Flower Next low-level API preview release. In Flower 1.9, `RecordSet` received a number of usability improvements that make it easier to build `RecordSet`-based `ServerApp`s and `ClientApp`s.

- **Beautify logging** ([#3379](https://github.com/adap/flower/pull/3379), [#3430](https://github.com/adap/flower/pull/3430), [#3461](https://github.com/adap/flower/pull/3461), [#3360](https://github.com/adap/flower/pull/3360), [#3433](https://github.com/adap/flower/pull/3433))

  Logs received a substantial update. Not only are logs now much nicer to look at, but they are also more configurable.

- **Improve reliability** ([#3564](https://github.com/adap/flower/pull/3564), [#3561](https://github.com/adap/flower/pull/3561), [#3566](https://github.com/adap/flower/pull/3566), [#3462](https://github.com/adap/flower/pull/3462), [#3225](https://github.com/adap/flower/pull/3225), [#3514](https://github.com/adap/flower/pull/3514), [#3535](https://github.com/adap/flower/pull/3535), [#3372](https://github.com/adap/flower/pull/3372))

  Flower 1.9 includes reliability improvements across many parts of the system. One example is a much improved SuperNode shutdown procedure.

- **Update Swift and C++ SDKs** ([#3321](https://github.com/adap/flower/pull/3321), [#2763](https://github.com/adap/flower/pull/2763))

  In the C++ SDK, communication-related code is now separate from main client logic. A new abstract class `Communicator` has been introduced alongside a gRPC implementation of it.

- **Improve testing, tooling and CI/CD infrastructure** ([#3294](https://github.com/adap/flower/pull/3294), [#3282](https://github.com/adap/flower/pull/3282), [#3311](https://github.com/adap/flower/pull/3311), [#2878](https://github.com/adap/flower/pull/2878), [#3333](https://github.com/adap/flower/pull/3333), [#3255](https://github.com/adap/flower/pull/3255), [#3349](https://github.com/adap/flower/pull/3349), [#3400](https://github.com/adap/flower/pull/3400), [#3401](https://github.com/adap/flower/pull/3401), [#3399](https://github.com/adap/flower/pull/3399), [#3346](https://github.com/adap/flower/pull/3346), [#3398](https://github.com/adap/flower/pull/3398), [#3397](https://github.com/adap/flower/pull/3397), [#3347](https://github.com/adap/flower/pull/3347), [#3502](https://github.com/adap/flower/pull/3502), [#3387](https://github.com/adap/flower/pull/3387), [#3542](https://github.com/adap/flower/pull/3542), [#3396](https://github.com/adap/flower/pull/3396), [#3496](https://github.com/adap/flower/pull/3496), [#3465](https://github.com/adap/flower/pull/3465), [#3473](https://github.com/adap/flower/pull/3473), [#3484](https://github.com/adap/flower/pull/3484), [#3521](https://github.com/adap/flower/pull/3521), [#3363](https://github.com/adap/flower/pull/3363), [#3497](https://github.com/adap/flower/pull/3497), [#3464](https://github.com/adap/flower/pull/3464), [#3495](https://github.com/adap/flower/pull/3495), [#3478](https://github.com/adap/flower/pull/3478), [#3271](https://github.com/adap/flower/pull/3271))

  As always, the Flower tooling, testing, and CI/CD infrastructure has received many updates.

- **Improve documentation** ([#3530](https://github.com/adap/flower/pull/3530), [#3539](https://github.com/adap/flower/pull/3539), [#3425](https://github.com/adap/flower/pull/3425), [#3520](https://github.com/adap/flower/pull/3520), [#3286](https://github.com/adap/flower/pull/3286), [#3516](https://github.com/adap/flower/pull/3516), [#3523](https://github.com/adap/flower/pull/3523), [#3545](https://github.com/adap/flower/pull/3545), [#3498](https://github.com/adap/flower/pull/3498), [#3439](https://github.com/adap/flower/pull/3439), [#3440](https://github.com/adap/flower/pull/3440), [#3382](https://github.com/adap/flower/pull/3382), [#3559](https://github.com/adap/flower/pull/3559), [#3432](https://github.com/adap/flower/pull/3432), [#3278](https://github.com/adap/flower/pull/3278), [#3371](https://github.com/adap/flower/pull/3371), [#3519](https://github.com/adap/flower/pull/3519), [#3267](https://github.com/adap/flower/pull/3267), [#3204](https://github.com/adap/flower/pull/3204), [#3274](https://github.com/adap/flower/pull/3274))

  As always, the Flower documentation has received many updates. Notable new pages include:

  - [How-to upgrate to Flower Next (Flower Next migration guide)](https://flower.ai/docs/framework/how-to-upgrade-to-flower-next.html)

  - [How-to run Flower using Docker](https://flower.ai/docs/framework/how-to-run-flower-using-docker.html)

  - [Flower Mods reference](https://flower.ai/docs/framework/ref-api/flwr.client.mod.html#module-flwr.client.mod)

- **General updates to Flower Examples** ([#3205](https://github.com/adap/flower/pull/3205), [#3226](https://github.com/adap/flower/pull/3226), [#3211](https://github.com/adap/flower/pull/3211), [#3252](https://github.com/adap/flower/pull/3252), [#3427](https://github.com/adap/flower/pull/3427), [#3410](https://github.com/adap/flower/pull/3410), [#3426](https://github.com/adap/flower/pull/3426), [#3228](https://github.com/adap/flower/pull/3228), [#3342](https://github.com/adap/flower/pull/3342), [#3200](https://github.com/adap/flower/pull/3200), [#3202](https://github.com/adap/flower/pull/3202), [#3394](https://github.com/adap/flower/pull/3394), [#3488](https://github.com/adap/flower/pull/3488), [#3329](https://github.com/adap/flower/pull/3329), [#3526](https://github.com/adap/flower/pull/3526), [#3392](https://github.com/adap/flower/pull/3392), [#3474](https://github.com/adap/flower/pull/3474), [#3269](https://github.com/adap/flower/pull/3269))

  As always, Flower code examples have received many updates.

- **General improvements** ([#3532](https://github.com/adap/flower/pull/3532), [#3318](https://github.com/adap/flower/pull/3318), [#3565](https://github.com/adap/flower/pull/3565), [#3296](https://github.com/adap/flower/pull/3296), [#3305](https://github.com/adap/flower/pull/3305), [#3246](https://github.com/adap/flower/pull/3246), [#3224](https://github.com/adap/flower/pull/3224), [#3475](https://github.com/adap/flower/pull/3475), [#3297](https://github.com/adap/flower/pull/3297), [#3317](https://github.com/adap/flower/pull/3317), [#3429](https://github.com/adap/flower/pull/3429), [#3196](https://github.com/adap/flower/pull/3196), [#3534](https://github.com/adap/flower/pull/3534), [#3240](https://github.com/adap/flower/pull/3240), [#3365](https://github.com/adap/flower/pull/3365), [#3407](https://github.com/adap/flower/pull/3407), [#3563](https://github.com/adap/flower/pull/3563), [#3344](https://github.com/adap/flower/pull/3344), [#3330](https://github.com/adap/flower/pull/3330), [#3436](https://github.com/adap/flower/pull/3436), [#3300](https://github.com/adap/flower/pull/3300), [#3327](https://github.com/adap/flower/pull/3327), [#3254](https://github.com/adap/flower/pull/3254), [#3253](https://github.com/adap/flower/pull/3253), [#3419](https://github.com/adap/flower/pull/3419), [#3289](https://github.com/adap/flower/pull/3289), [#3208](https://github.com/adap/flower/pull/3208), [#3245](https://github.com/adap/flower/pull/3245), [#3319](https://github.com/adap/flower/pull/3319), [#3203](https://github.com/adap/flower/pull/3203), [#3423](https://github.com/adap/flower/pull/3423), [#3352](https://github.com/adap/flower/pull/3352), [#3292](https://github.com/adap/flower/pull/3292), [#3261](https://github.com/adap/flower/pull/3261))

### Deprecations

- **Deprecate Python 3.8 support**

  Python 3.8 will stop receiving security fixes in [October 2024](https://devguide.python.org/versions/). Support for Python 3.8 is now deprecated and will be removed in an upcoming release.

- **Deprecate (experimental)** `flower-driver-api` **and** `flower-fleet-api` ([#3416](https://github.com/adap/flower/pull/3416), [#3420](https://github.com/adap/flower/pull/3420))

  Flower 1.9 deprecates the two (experimental) commands `flower-driver-api` and `flower-fleet-api`. Both commands will be removed in an upcoming release. Use `flower-superlink` instead.

- **Deprecate** `--server` **in favor of** `--superlink` ([#3518](https://github.com/adap/flower/pull/3518))

  The commands `flower-server-app` and `flower-client-app` should use `--superlink` instead of the now deprecated `--server`. Support for `--server` will be removed in a future release.

### Incompatible changes

- **Replace** `flower-superlink` **CLI option** `--certificates` **with** `--ssl-ca-certfile` **,** `--ssl-certfile` **and** `--ssl-keyfile` ([#3512](https://github.com/adap/flower/pull/3512), [#3408](https://github.com/adap/flower/pull/3408))

  SSL-related `flower-superlink` CLI arguments were restructured in an incompatible way. Instead of passing a single `--certificates` flag with three values, you now need to pass three flags (`--ssl-ca-certfile`, `--ssl-certfile` and `--ssl-keyfile`) with one value each. Check out the [SSL connections](https://flower.ai/docs/framework/how-to-enable-ssl-connections.html) documentation page for details.

- **Remove SuperLink** `--vce` **option** ([#3513](https://github.com/adap/flower/pull/3513))

  Instead of separately starting a SuperLink and a `ServerApp` for simulation, simulations must now be started using the single `flower-simulation` command.

- **Merge** `--grpc-rere` **and** `--rest` **SuperLink options** ([#3527](https://github.com/adap/flower/pull/3527))

  To simplify the usage of `flower-superlink`, previously separate sets of CLI options for gRPC and REST were merged into one unified set of options. Consult the [Flower CLI reference documentation](https://flower.ai/docs/framework/ref-api-cli.html) for details.

## v1.8.0 (2024-04-03)

### Thanks to our contributors

We would like to give our special thanks to all the contributors who made the new version of Flower possible (in `git shortlog` order):

`Adam Narozniak`, `Charles Beauville`, `Daniel J. Beutel`, `Daniel Nata Nugraha`, `Danny`, `Gustavo Bertoli`, `Heng Pan`, `Ikko Eltociear Ashimine`, `Jack Cook`, `Javier`, `Raj Parekh`, `Robert Steiner`, `Sebastian van der Voort`, `Taner Topal`, `Yan Gao`, `mohammadnaseri`, `tabdar-khan` <!---TOKEN_v1.8.0-->

### What's new?

- **Introduce Flower Next high-level API (stable)** ([#3002](https://github.com/adap/flower/pull/3002), [#2934](https://github.com/adap/flower/pull/2934), [#2958](https://github.com/adap/flower/pull/2958), [#3173](https://github.com/adap/flower/pull/3173), [#3174](https://github.com/adap/flower/pull/3174), [#2923](https://github.com/adap/flower/pull/2923), [#2691](https://github.com/adap/flower/pull/2691), [#3079](https://github.com/adap/flower/pull/3079), [#2961](https://github.com/adap/flower/pull/2961), [#2924](https://github.com/adap/flower/pull/2924), [#3166](https://github.com/adap/flower/pull/3166), [#3031](https://github.com/adap/flower/pull/3031), [#3057](https://github.com/adap/flower/pull/3057), [#3000](https://github.com/adap/flower/pull/3000), [#3113](https://github.com/adap/flower/pull/3113), [#2957](https://github.com/adap/flower/pull/2957), [#3183](https://github.com/adap/flower/pull/3183), [#3180](https://github.com/adap/flower/pull/3180), [#3035](https://github.com/adap/flower/pull/3035), [#3189](https://github.com/adap/flower/pull/3189), [#3185](https://github.com/adap/flower/pull/3185), [#3190](https://github.com/adap/flower/pull/3190), [#3191](https://github.com/adap/flower/pull/3191), [#3195](https://github.com/adap/flower/pull/3195), [#3197](https://github.com/adap/flower/pull/3197))

  The Flower Next high-level API is stable! Flower Next is the future of Flower - all new features (like Flower Mods) will be built on top of it. You can start to migrate your existing projects to Flower Next by using `ServerApp` and `ClientApp` (check out `quickstart-pytorch` or `quickstart-tensorflow`, a detailed migration guide will follow shortly). Flower Next allows you to run multiple projects concurrently (we call this multi-run) and execute the same project in either simulation environments or deployment environments without having to change a single line of code. The best part? It's fully compatible with existing Flower projects that use `Strategy`, `NumPyClient` & co.

- **Introduce Flower Next low-level API (preview)** ([#3062](https://github.com/adap/flower/pull/3062), [#3034](https://github.com/adap/flower/pull/3034), [#3069](https://github.com/adap/flower/pull/3069))

  In addition to the Flower Next *high-level* API that uses `Strategy`, `NumPyClient` & co, Flower 1.8 also comes with a preview version of the new Flower Next *low-level* API. The low-level API allows for granular control of every aspect of the learning process by sending/receiving individual messages to/from client nodes. The new `ServerApp` supports registering a custom `main` function that allows writing custom training loops for methods like async FL, cyclic training, or federated analytics. The new `ClientApp` supports registering `train`, `evaluate` and `query` functions that can access the raw message received from the `ServerApp`. New abstractions like `RecordSet`, `Message` and `Context` further enable sending multiple models, multiple sets of config values and metrics, stateful computations on the client node and implementations of custom SMPC protocols, to name just a few.

- **Introduce Flower Mods (preview)** ([#3054](https://github.com/adap/flower/pull/3054), [#2911](https://github.com/adap/flower/pull/2911), [#3083](https://github.com/adap/flower/pull/3083))

  Flower Modifiers (we call them Mods) can intercept messages and analyze, edit or handle them directly. Mods can be used to develop pluggable modules that work across different projects. Flower 1.8 already includes mods to log the size of a message, the number of parameters sent over the network, differential privacy with fixed clipping and adaptive clipping, local differential privacy and secure aggregation protocols SecAgg and SecAgg+. The Flower Mods API is released as a preview, but researchers can already use it to experiment with arbirtrary SMPC protocols.

- **Fine-tune LLMs with LLM FlowerTune** ([#3029](https://github.com/adap/flower/pull/3029), [#3089](https://github.com/adap/flower/pull/3089), [#3092](https://github.com/adap/flower/pull/3092), [#3100](https://github.com/adap/flower/pull/3100), [#3114](https://github.com/adap/flower/pull/3114), [#3162](https://github.com/adap/flower/pull/3162), [#3172](https://github.com/adap/flower/pull/3172))

  We are introducing LLM FlowerTune, an introductory example that demonstrates federated LLM fine-tuning of pre-trained Llama2 models on the Alpaca-GPT4 dataset. The example is built to be easily adapted to use different models and/or datasets. Read our blog post [LLM FlowerTune: Federated LLM Fine-tuning with Flower](https://flower.ai/blog/2024-03-14-llm-flowertune-federated-llm-finetuning-with-flower/) for more details.

- **Introduce built-in Differential Privacy (preview)** ([#2798](https://github.com/adap/flower/pull/2798), [#2959](https://github.com/adap/flower/pull/2959), [#3038](https://github.com/adap/flower/pull/3038), [#3147](https://github.com/adap/flower/pull/3147), [#2909](https://github.com/adap/flower/pull/2909), [#2893](https://github.com/adap/flower/pull/2893), [#2892](https://github.com/adap/flower/pull/2892), [#3039](https://github.com/adap/flower/pull/3039), [#3074](https://github.com/adap/flower/pull/3074))

  Built-in Differential Privacy is here! Flower supports both central and local differential privacy (DP). Central DP can be configured with either fixed or adaptive clipping. The clipping can happen either on the server-side or the client-side. Local DP does both clipping and noising on the client-side. A new documentation page [explains Differential Privacy approaches](https://flower.ai/docs/framework/explanation-differential-privacy.html) and a new how-to guide describes [how to use the new Differential Privacy components](https://flower.ai/docs/framework/how-to-use-differential-privacy.html) in Flower.

- **Introduce built-in Secure Aggregation (preview)** ([#3120](https://github.com/adap/flower/pull/3120), [#3110](https://github.com/adap/flower/pull/3110), [#3108](https://github.com/adap/flower/pull/3108))

  Built-in Secure Aggregation is here! Flower now supports different secure aggregation protocols out-of-the-box. The best part? You can add secure aggregation to your Flower projects with only a few lines of code. In this initial release, we inlcude support for SecAgg and SecAgg+, but more protocols will be implemented shortly. We'll also add detailed docs that explain secure aggregation and how to use it in Flower. You can already check out the new code example that shows how to use Flower to easily combine Federated Learning, Differential Privacy and Secure Aggregation in the same project.

- **Introduce** `flwr` **CLI (preview)** ([#2942](https://github.com/adap/flower/pull/2942), [#3055](https://github.com/adap/flower/pull/3055), [#3111](https://github.com/adap/flower/pull/3111), [#3130](https://github.com/adap/flower/pull/3130), [#3136](https://github.com/adap/flower/pull/3136), [#3094](https://github.com/adap/flower/pull/3094), [#3059](https://github.com/adap/flower/pull/3059), [#3049](https://github.com/adap/flower/pull/3049), [#3142](https://github.com/adap/flower/pull/3142))

  A new `flwr` CLI command allows creating new Flower projects (`flwr new`) and then running them using the Simulation Engine (`flwr run`).

- **Introduce Flower Next Simulation Engine** ([#3024](https://github.com/adap/flower/pull/3024), [#3061](https://github.com/adap/flower/pull/3061), [#2997](https://github.com/adap/flower/pull/2997), [#2783](https://github.com/adap/flower/pull/2783), [#3184](https://github.com/adap/flower/pull/3184), [#3075](https://github.com/adap/flower/pull/3075), [#3047](https://github.com/adap/flower/pull/3047), [#2998](https://github.com/adap/flower/pull/2998), [#3009](https://github.com/adap/flower/pull/3009), [#3008](https://github.com/adap/flower/pull/3008))

  The Flower Simulation Engine can now run Flower Next projects. For notebook environments, there's also a new `run_simulation` function that can run `ServerApp` and `ClientApp`.

- **Handle SuperNode connection errors** ([#2969](https://github.com/adap/flower/pull/2969))

  A SuperNode will now try to reconnect indefinitely to the SuperLink in case of connection errors. The arguments `--max-retries` and `--max-wait-time` can now be passed to the `flower-client-app` command. `--max-retries` will define the number of tentatives the client should make before it gives up trying to reconnect to the SuperLink, and, `--max-wait-time` defines the time before the SuperNode gives up trying to reconnect to the SuperLink.

- **General updates to Flower Baselines** ([#2904](https://github.com/adap/flower/pull/2904), [#2482](https://github.com/adap/flower/pull/2482), [#2985](https://github.com/adap/flower/pull/2985), [#2968](https://github.com/adap/flower/pull/2968))

  There's a new [FedStar](https://flower.ai/docs/baselines/fedstar.html) baseline. Several other baselined have been updated as well.

- **Improve documentation and translations** ([#3050](https://github.com/adap/flower/pull/3050), [#3044](https://github.com/adap/flower/pull/3044), [#3043](https://github.com/adap/flower/pull/3043), [#2986](https://github.com/adap/flower/pull/2986), [#3041](https://github.com/adap/flower/pull/3041), [#3046](https://github.com/adap/flower/pull/3046), [#3042](https://github.com/adap/flower/pull/3042), [#2978](https://github.com/adap/flower/pull/2978), [#2952](https://github.com/adap/flower/pull/2952), [#3167](https://github.com/adap/flower/pull/3167), [#2953](https://github.com/adap/flower/pull/2953), [#3045](https://github.com/adap/flower/pull/3045), [#2654](https://github.com/adap/flower/pull/2654), [#3082](https://github.com/adap/flower/pull/3082), [#2990](https://github.com/adap/flower/pull/2990), [#2989](https://github.com/adap/flower/pull/2989))

  As usual, we merged many smaller and larger improvements to the documentation. A special thank you goes to [Sebastian van der Voort](https://github.com/svdvoort) for landing a big documentation PR!

- **General updates to Flower Examples** ([3134](https://github.com/adap/flower/pull/3134), [2996](https://github.com/adap/flower/pull/2996), [2930](https://github.com/adap/flower/pull/2930), [2967](https://github.com/adap/flower/pull/2967), [2467](https://github.com/adap/flower/pull/2467), [2910](https://github.com/adap/flower/pull/2910), [#2918](https://github.com/adap/flower/pull/2918), [#2773](https://github.com/adap/flower/pull/2773), [#3063](https://github.com/adap/flower/pull/3063), [#3116](https://github.com/adap/flower/pull/3116), [#3117](https://github.com/adap/flower/pull/3117))

  Two new examples show federated training of a Vision Transformer (ViT) and federated learning in a medical context using the popular MONAI library. `quickstart-pytorch` and `quickstart-tensorflow` demonstrate the new Flower Next `ServerApp` and `ClientApp`. Many other examples received considerable updates as well.

- **General improvements** ([#3171](https://github.com/adap/flower/pull/3171), [3099](https://github.com/adap/flower/pull/3099), [3003](https://github.com/adap/flower/pull/3003), [3145](https://github.com/adap/flower/pull/3145), [3017](https://github.com/adap/flower/pull/3017), [3085](https://github.com/adap/flower/pull/3085), [3012](https://github.com/adap/flower/pull/3012), [3119](https://github.com/adap/flower/pull/3119), [2991](https://github.com/adap/flower/pull/2991), [2970](https://github.com/adap/flower/pull/2970), [2980](https://github.com/adap/flower/pull/2980), [3086](https://github.com/adap/flower/pull/3086), [2932](https://github.com/adap/flower/pull/2932), [2928](https://github.com/adap/flower/pull/2928), [2941](https://github.com/adap/flower/pull/2941), [2933](https://github.com/adap/flower/pull/2933), [3181](https://github.com/adap/flower/pull/3181), [2973](https://github.com/adap/flower/pull/2973), [2992](https://github.com/adap/flower/pull/2992), [2915](https://github.com/adap/flower/pull/2915), [3040](https://github.com/adap/flower/pull/3040), [3022](https://github.com/adap/flower/pull/3022), [3032](https://github.com/adap/flower/pull/3032), [2902](https://github.com/adap/flower/pull/2902), [2931](https://github.com/adap/flower/pull/2931), [3005](https://github.com/adap/flower/pull/3005), [3132](https://github.com/adap/flower/pull/3132), [3115](https://github.com/adap/flower/pull/3115), [2944](https://github.com/adap/flower/pull/2944), [3064](https://github.com/adap/flower/pull/3064), [3106](https://github.com/adap/flower/pull/3106), [2974](https://github.com/adap/flower/pull/2974), [3178](https://github.com/adap/flower/pull/3178), [2993](https://github.com/adap/flower/pull/2993), [3186](https://github.com/adap/flower/pull/3186), [3091](https://github.com/adap/flower/pull/3091), [3125](https://github.com/adap/flower/pull/3125), [3093](https://github.com/adap/flower/pull/3093), [3013](https://github.com/adap/flower/pull/3013), [3033](https://github.com/adap/flower/pull/3033), [3133](https://github.com/adap/flower/pull/3133), [3068](https://github.com/adap/flower/pull/3068), [2916](https://github.com/adap/flower/pull/2916), [2975](https://github.com/adap/flower/pull/2975), [2984](https://github.com/adap/flower/pull/2984), [2846](https://github.com/adap/flower/pull/2846), [3077](https://github.com/adap/flower/pull/3077), [3143](https://github.com/adap/flower/pull/3143), [2921](https://github.com/adap/flower/pull/2921), [3101](https://github.com/adap/flower/pull/3101), [2927](https://github.com/adap/flower/pull/2927), [2995](https://github.com/adap/flower/pull/2995), [2972](https://github.com/adap/flower/pull/2972), [2912](https://github.com/adap/flower/pull/2912), [3065](https://github.com/adap/flower/pull/3065), [3028](https://github.com/adap/flower/pull/3028), [2922](https://github.com/adap/flower/pull/2922), [2982](https://github.com/adap/flower/pull/2982), [2914](https://github.com/adap/flower/pull/2914), [3179](https://github.com/adap/flower/pull/3179), [3080](https://github.com/adap/flower/pull/3080), [2994](https://github.com/adap/flower/pull/2994), [3187](https://github.com/adap/flower/pull/3187), [2926](https://github.com/adap/flower/pull/2926), [3018](https://github.com/adap/flower/pull/3018), [3144](https://github.com/adap/flower/pull/3144), [3011](https://github.com/adap/flower/pull/3011), [#3152](https://github.com/adap/flower/pull/3152), [#2836](https://github.com/adap/flower/pull/2836), [#2929](https://github.com/adap/flower/pull/2929), [#2943](https://github.com/adap/flower/pull/2943), [#2955](https://github.com/adap/flower/pull/2955), [#2954](https://github.com/adap/flower/pull/2954))

### Incompatible changes

None

## v1.7.0 (2024-02-05)

### Thanks to our contributors

We would like to give our special thanks to all the contributors who made the new version of Flower possible (in `git shortlog` order):

`Aasheesh Singh`, `Adam Narozniak`, `Aml Hassan Esmil`, `Charles Beauville`, `Daniel J. Beutel`, `Daniel Nata Nugraha`, `Edoardo Gabrielli`, `Gustavo Bertoli`, `HelinLin`, `Heng Pan`, `Javier`, `M S Chaitanya Kumar`, `Mohammad Naseri`, `Nikos Vlachakis`, `Pritam Neog`, `Robert Kuska`, `Robert Steiner`, `Taner Topal`, `Yahia Salaheldin Shaaban`, `Yan Gao`, `Yasar Abbas` <!---TOKEN_v1.7.0-->

### What's new?

- **Introduce stateful clients (experimental)** ([#2770](https://github.com/adap/flower/pull/2770), [#2686](https://github.com/adap/flower/pull/2686), [#2696](https://github.com/adap/flower/pull/2696), [#2643](https://github.com/adap/flower/pull/2643), [#2769](https://github.com/adap/flower/pull/2769))

  Subclasses of `Client` and `NumPyClient` can now store local state that remains on the client. Let's start with the highlight first: this new feature is compatible with both simulated clients (via `start_simulation`) and networked clients (via `start_client`). It's also the first preview of new abstractions like `Context` and `RecordSet`. Clients can access state of type `RecordSet` via `state: RecordSet = self.context.state`. Changes to this `RecordSet` are preserved across different rounds of execution to enable stateful computations in a unified way across simulation and deployment.

- **Improve performance** ([#2293](https://github.com/adap/flower/pull/2293))

  Flower is faster than ever. All `FedAvg`-derived strategies now use in-place aggregation to reduce memory consumption. The Flower client serialization/deserialization has been rewritten from the ground up, which results in significant speedups, especially when the client-side training time is short.

- **Support Federated Learning with Apple MLX and Flower** ([#2693](https://github.com/adap/flower/pull/2693))

  Flower has official support for federated learning using [Apple MLX](https://ml-explore.github.io/mlx) via the new `quickstart-mlx` code example.

- **Introduce new XGBoost cyclic strategy** ([#2666](https://github.com/adap/flower/pull/2666), [#2668](https://github.com/adap/flower/pull/2668))

  A new strategy called `FedXgbCyclic` supports a client-by-client style of training (often called cyclic). The `xgboost-comprehensive` code example shows how to use it in a full project. In addition to that, `xgboost-comprehensive` now also supports simulation mode. With this, Flower offers best-in-class XGBoost support.

- **Support Python 3.11** ([#2394](https://github.com/adap/flower/pull/2394))

  Framework tests now run on Python 3.8, 3.9, 3.10, and 3.11. This will ensure better support for users using more recent Python versions.

- **Update gRPC and ProtoBuf dependencies** ([#2814](https://github.com/adap/flower/pull/2814))

  The `grpcio` and `protobuf` dependencies were updated to their latest versions for improved security and performance.

- **Introduce Docker image for Flower server** ([#2700](https://github.com/adap/flower/pull/2700), [#2688](https://github.com/adap/flower/pull/2688), [#2705](https://github.com/adap/flower/pull/2705), [#2695](https://github.com/adap/flower/pull/2695), [#2747](https://github.com/adap/flower/pull/2747), [#2746](https://github.com/adap/flower/pull/2746), [#2680](https://github.com/adap/flower/pull/2680), [#2682](https://github.com/adap/flower/pull/2682), [#2701](https://github.com/adap/flower/pull/2701))

  The Flower server can now be run using an official Docker image. A new how-to guide explains [how to run Flower using Docker](https://flower.ai/docs/framework/how-to-run-flower-using-docker.html). An official Flower client Docker image will follow.

- **Introduce** `flower-via-docker-compose` **example** ([#2626](https://github.com/adap/flower/pull/2626))

- **Introduce** `quickstart-sklearn-tabular` **example** ([#2719](https://github.com/adap/flower/pull/2719))

- **Introduce** `custom-metrics` **example** ([#1958](https://github.com/adap/flower/pull/1958))

- **Update code examples to use Flower Datasets** ([#2450](https://github.com/adap/flower/pull/2450), [#2456](https://github.com/adap/flower/pull/2456), [#2318](https://github.com/adap/flower/pull/2318), [#2712](https://github.com/adap/flower/pull/2712))

  Several code examples were updated to use [Flower Datasets](https://flower.ai/docs/datasets/).

- **General updates to Flower Examples** ([#2381](https://github.com/adap/flower/pull/2381), [#2805](https://github.com/adap/flower/pull/2805), [#2782](https://github.com/adap/flower/pull/2782), [#2806](https://github.com/adap/flower/pull/2806), [#2829](https://github.com/adap/flower/pull/2829), [#2825](https://github.com/adap/flower/pull/2825), [#2816](https://github.com/adap/flower/pull/2816), [#2726](https://github.com/adap/flower/pull/2726), [#2659](https://github.com/adap/flower/pull/2659), [#2655](https://github.com/adap/flower/pull/2655))

  Many Flower code examples received substantial updates.

- **Update Flower Baselines**

  - HFedXGBoost ([#2226](https://github.com/adap/flower/pull/2226), [#2771](https://github.com/adap/flower/pull/2771))
  - FedVSSL ([#2412](https://github.com/adap/flower/pull/2412))
  - FedNova ([#2179](https://github.com/adap/flower/pull/2179))
  - HeteroFL ([#2439](https://github.com/adap/flower/pull/2439))
  - FedAvgM ([#2246](https://github.com/adap/flower/pull/2246))
  - FedPara ([#2722](https://github.com/adap/flower/pull/2722))

- **Improve documentation** ([#2674](https://github.com/adap/flower/pull/2674), [#2480](https://github.com/adap/flower/pull/2480), [#2826](https://github.com/adap/flower/pull/2826), [#2727](https://github.com/adap/flower/pull/2727), [#2761](https://github.com/adap/flower/pull/2761), [#2900](https://github.com/adap/flower/pull/2900))

- **Improved testing and development infrastructure** ([#2797](https://github.com/adap/flower/pull/2797), [#2676](https://github.com/adap/flower/pull/2676), [#2644](https://github.com/adap/flower/pull/2644), [#2656](https://github.com/adap/flower/pull/2656), [#2848](https://github.com/adap/flower/pull/2848), [#2675](https://github.com/adap/flower/pull/2675), [#2735](https://github.com/adap/flower/pull/2735), [#2767](https://github.com/adap/flower/pull/2767), [#2732](https://github.com/adap/flower/pull/2732), [#2744](https://github.com/adap/flower/pull/2744), [#2681](https://github.com/adap/flower/pull/2681), [#2699](https://github.com/adap/flower/pull/2699), [#2745](https://github.com/adap/flower/pull/2745), [#2734](https://github.com/adap/flower/pull/2734), [#2731](https://github.com/adap/flower/pull/2731), [#2652](https://github.com/adap/flower/pull/2652), [#2720](https://github.com/adap/flower/pull/2720), [#2721](https://github.com/adap/flower/pull/2721), [#2717](https://github.com/adap/flower/pull/2717), [#2864](https://github.com/adap/flower/pull/2864), [#2694](https://github.com/adap/flower/pull/2694), [#2709](https://github.com/adap/flower/pull/2709), [#2658](https://github.com/adap/flower/pull/2658), [#2796](https://github.com/adap/flower/pull/2796), [#2692](https://github.com/adap/flower/pull/2692), [#2657](https://github.com/adap/flower/pull/2657), [#2813](https://github.com/adap/flower/pull/2813), [#2661](https://github.com/adap/flower/pull/2661), [#2398](https://github.com/adap/flower/pull/2398))

  The Flower testing and development infrastructure has received substantial updates. This makes Flower 1.7 the most tested release ever.

- **Update dependencies** ([#2753](https://github.com/adap/flower/pull/2753), [#2651](https://github.com/adap/flower/pull/2651), [#2739](https://github.com/adap/flower/pull/2739), [#2837](https://github.com/adap/flower/pull/2837), [#2788](https://github.com/adap/flower/pull/2788), [#2811](https://github.com/adap/flower/pull/2811), [#2774](https://github.com/adap/flower/pull/2774), [#2790](https://github.com/adap/flower/pull/2790), [#2751](https://github.com/adap/flower/pull/2751), [#2850](https://github.com/adap/flower/pull/2850), [#2812](https://github.com/adap/flower/pull/2812), [#2872](https://github.com/adap/flower/pull/2872), [#2736](https://github.com/adap/flower/pull/2736), [#2756](https://github.com/adap/flower/pull/2756), [#2857](https://github.com/adap/flower/pull/2857), [#2757](https://github.com/adap/flower/pull/2757), [#2810](https://github.com/adap/flower/pull/2810), [#2740](https://github.com/adap/flower/pull/2740), [#2789](https://github.com/adap/flower/pull/2789))

- **General improvements** ([#2803](https://github.com/adap/flower/pull/2803), [#2847](https://github.com/adap/flower/pull/2847), [#2877](https://github.com/adap/flower/pull/2877), [#2690](https://github.com/adap/flower/pull/2690), [#2889](https://github.com/adap/flower/pull/2889), [#2874](https://github.com/adap/flower/pull/2874), [#2819](https://github.com/adap/flower/pull/2819), [#2689](https://github.com/adap/flower/pull/2689), [#2457](https://github.com/adap/flower/pull/2457), [#2870](https://github.com/adap/flower/pull/2870), [#2669](https://github.com/adap/flower/pull/2669), [#2876](https://github.com/adap/flower/pull/2876), [#2885](https://github.com/adap/flower/pull/2885), [#2858](https://github.com/adap/flower/pull/2858), [#2867](https://github.com/adap/flower/pull/2867), [#2351](https://github.com/adap/flower/pull/2351), [#2886](https://github.com/adap/flower/pull/2886), [#2860](https://github.com/adap/flower/pull/2860), [#2828](https://github.com/adap/flower/pull/2828), [#2869](https://github.com/adap/flower/pull/2869), [#2875](https://github.com/adap/flower/pull/2875), [#2733](https://github.com/adap/flower/pull/2733), [#2488](https://github.com/adap/flower/pull/2488), [#2646](https://github.com/adap/flower/pull/2646), [#2879](https://github.com/adap/flower/pull/2879), [#2821](https://github.com/adap/flower/pull/2821), [#2855](https://github.com/adap/flower/pull/2855), [#2800](https://github.com/adap/flower/pull/2800), [#2807](https://github.com/adap/flower/pull/2807), [#2801](https://github.com/adap/flower/pull/2801), [#2804](https://github.com/adap/flower/pull/2804), [#2851](https://github.com/adap/flower/pull/2851), [#2787](https://github.com/adap/flower/pull/2787), [#2852](https://github.com/adap/flower/pull/2852), [#2672](https://github.com/adap/flower/pull/2672), [#2759](https://github.com/adap/flower/pull/2759))

### Incompatible changes

- **Deprecate** `start_numpy_client` ([#2563](https://github.com/adap/flower/pull/2563), [#2718](https://github.com/adap/flower/pull/2718))

  Until now, clients of type `NumPyClient` needed to be started via `start_numpy_client`. In our efforts to consolidate framework APIs, we have introduced changes, and now all client types should start via `start_client`. To continue using `NumPyClient` clients, you simply need to first call the `.to_client()` method and then pass returned `Client` object to `start_client`. The examples and the documentation have been updated accordingly.

- **Deprecate legacy DP wrappers** ([#2749](https://github.com/adap/flower/pull/2749))

  Legacy DP wrapper classes are deprecated, but still functional. This is in preparation for an all-new pluggable version of differential privacy support in Flower.

- **Make optional arg** `--callable` **in** `flower-client` **a required positional arg** ([#2673](https://github.com/adap/flower/pull/2673))

- **Rename** `certificates` **to** `root_certificates` **in** `Driver` ([#2890](https://github.com/adap/flower/pull/2890))

- **Drop experimental** `Task` **fields** ([#2866](https://github.com/adap/flower/pull/2866), [#2865](https://github.com/adap/flower/pull/2865))

  Experimental fields `sa`, `legacy_server_message` and `legacy_client_message` were removed from `Task` message. The removed fields are superseded by the new `RecordSet` abstraction.

- **Retire MXNet examples** ([#2724](https://github.com/adap/flower/pull/2724))

  The development of the MXNet fremework has ended and the project is now [archived on GitHub](https://github.com/apache/mxnet). Existing MXNet examples won't receive updates.

## v1.6.0 (2023-11-28)

### Thanks to our contributors

We would like to give our special thanks to all the contributors who made the new version of Flower possible (in `git shortlog` order):

`Aashish Kolluri`, `Adam Narozniak`, `Alessio Mora`, `Barathwaja S`, `Charles Beauville`, `Daniel J. Beutel`, `Daniel Nata Nugraha`, `Gabriel Mota`, `Heng Pan`, `Ivan Agarský`, `JS.KIM`, `Javier`, `Marius Schlegel`, `Navin Chandra`, `Nic Lane`, `Peterpan828`, `Qinbin Li`, `Shaz-hash`, `Steve Laskaridis`, `Taner Topal`, `William Lindskog`, `Yan Gao`, `cnxdeveloper`, `k3nfalt` <!---TOKEN_v1.6.0-->

### What's new?

- **Add experimental support for Python 3.12** ([#2565](https://github.com/adap/flower/pull/2565))

- **Add new XGBoost examples** ([#2612](https://github.com/adap/flower/pull/2612), [#2554](https://github.com/adap/flower/pull/2554), [#2617](https://github.com/adap/flower/pull/2617), [#2618](https://github.com/adap/flower/pull/2618), [#2619](https://github.com/adap/flower/pull/2619), [#2567](https://github.com/adap/flower/pull/2567))

  We have added a new `xgboost-quickstart` example alongside a new `xgboost-comprehensive` example that goes more in-depth.

- **Add Vertical FL example** ([#2598](https://github.com/adap/flower/pull/2598))

  We had many questions about Vertical Federated Learning using Flower, so we decided to add an simple example for it on the [Titanic dataset](https://www.kaggle.com/competitions/titanic/data) alongside a tutorial (in the README).

- **Support custom** `ClientManager` **in** `start_driver()` ([#2292](https://github.com/adap/flower/pull/2292))

- **Update REST API to support create and delete nodes** ([#2283](https://github.com/adap/flower/pull/2283))

- **Update the Android SDK** ([#2187](https://github.com/adap/flower/pull/2187))

  Add gRPC request-response capability to the Android SDK.

- **Update the C++ SDK** ([#2537](https://github.com/adap/flower/pull/2537), [#2528](https://github.com/adap/flower/pull/2528), [#2523](https://github.com/adap/flower/pull/2523), [#2522](https://github.com/adap/flower/pull/2522))

  Add gRPC request-response capability to the C++ SDK.

- **Make HTTPS the new default** ([#2591](https://github.com/adap/flower/pull/2591), [#2636](https://github.com/adap/flower/pull/2636))

  Flower is moving to HTTPS by default. The new `flower-server` requires passing `--certificates`, but users can enable `--insecure` to use HTTP for prototyping. The same applies to `flower-client`, which can either use user-provided credentials or gRPC-bundled certificates to connect to an HTTPS-enabled server or requires opt-out via passing `--insecure` to enable insecure HTTP connections.

  For backward compatibility, `start_client()` and `start_numpy_client()` will still start in insecure mode by default. In a future release, insecure connections will require user opt-in by passing `insecure=True`.

- **Unify client API** ([#2303](https://github.com/adap/flower/pull/2303), [#2390](https://github.com/adap/flower/pull/2390), [#2493](https://github.com/adap/flower/pull/2493))

  Using the `client_fn`, Flower clients can interchangeably run as standalone processes (i.e. via `start_client`) or in simulation (i.e. via `start_simulation`) without requiring changes to how the client class is defined and instantiated. The `to_client()` function is introduced to convert a `NumPyClient` to a `Client`.

- **Add new** `Bulyan` **strategy** ([#1817](https://github.com/adap/flower/pull/1817), [#1891](https://github.com/adap/flower/pull/1891))

  The new `Bulyan` strategy implements Bulyan by [El Mhamdi et al., 2018](https://arxiv.org/abs/1802.07927)

- **Add new** `XGB Bagging` **strategy** ([#2611](https://github.com/adap/flower/pull/2611))

- **Introduce `WorkloadState`** ([#2564](https://github.com/adap/flower/pull/2564), [#2632](https://github.com/adap/flower/pull/2632))

- **Introduce `WorkloadState`** ([#2564](https://github.com/adap/flower/pull/2564), [#2632](https://github.com/adap/flower/pull/2632))

- **Update Flower Baselines**

  - FedProx ([#2210](https://github.com/adap/flower/pull/2210), [#2286](https://github.com/adap/flower/pull/2286), [#2509](https://github.com/adap/flower/pull/2509))

  - Baselines Docs ([#2290](https://github.com/adap/flower/pull/2290), [#2400](https://github.com/adap/flower/pull/2400))

  - FedMLB ([#2340](https://github.com/adap/flower/pull/2340), [#2507](https://github.com/adap/flower/pull/2507))

  - TAMUNA ([#2254](https://github.com/adap/flower/pull/2254), [#2508](https://github.com/adap/flower/pull/2508))

  - FedMeta [#2438](https://github.com/adap/flower/pull/2438)

  - FjORD [#2431](https://github.com/adap/flower/pull/2431)

  - MOON [#2421](https://github.com/adap/flower/pull/2421)

  - DepthFL [#2295](https://github.com/adap/flower/pull/2295)

  - FedPer [#2266](https://github.com/adap/flower/pull/2266)

  - FedWav2vec [#2551](https://github.com/adap/flower/pull/2551)

  - niid-Bench [#2428](https://github.com/adap/flower/pull/2428)

  - FedBN ([#2608](https://github.com/adap/flower/pull/2608), [#2615](https://github.com/adap/flower/pull/2615))

- **General updates to Flower Examples** ([#2384](https://github.com/adap/flower/pull/2384), [#2425](https://github.com/adap/flower/pull/2425), [#2526](https://github.com/adap/flower/pull/2526), [#2302](https://github.com/adap/flower/pull/2302), [#2545](https://github.com/adap/flower/pull/2545))

- **General updates to Flower Baselines** ([#2301](https://github.com/adap/flower/pull/2301), [#2305](https://github.com/adap/flower/pull/2305), [#2307](https://github.com/adap/flower/pull/2307), [#2327](https://github.com/adap/flower/pull/2327), [#2435](https://github.com/adap/flower/pull/2435), [#2462](https://github.com/adap/flower/pull/2462), [#2463](https://github.com/adap/flower/pull/2463), [#2461](https://github.com/adap/flower/pull/2461), [#2469](https://github.com/adap/flower/pull/2469), [#2466](https://github.com/adap/flower/pull/2466), [#2471](https://github.com/adap/flower/pull/2471), [#2472](https://github.com/adap/flower/pull/2472), [#2470](https://github.com/adap/flower/pull/2470))

- **General updates to the simulation engine** ([#2331](https://github.com/adap/flower/pull/2331), [#2447](https://github.com/adap/flower/pull/2447), [#2448](https://github.com/adap/flower/pull/2448), [#2294](https://github.com/adap/flower/pull/2294))

- **General updates to Flower SDKs** ([#2288](https://github.com/adap/flower/pull/2288), [#2429](https://github.com/adap/flower/pull/2429), [#2555](https://github.com/adap/flower/pull/2555), [#2543](https://github.com/adap/flower/pull/2543), [#2544](https://github.com/adap/flower/pull/2544), [#2597](https://github.com/adap/flower/pull/2597), [#2623](https://github.com/adap/flower/pull/2623))

- **General improvements** ([#2309](https://github.com/adap/flower/pull/2309), [#2310](https://github.com/adap/flower/pull/2310), [#2313](https://github.com/adap/flower/pull/2313), [#2316](https://github.com/adap/flower/pull/2316), [#2317](https://github.com/adap/flower/pull/2317), [#2349](https://github.com/adap/flower/pull/2349), [#2360](https://github.com/adap/flower/pull/2360), [#2402](https://github.com/adap/flower/pull/2402), [#2446](https://github.com/adap/flower/pull/2446), [#2561](https://github.com/adap/flower/pull/2561), [#2273](https://github.com/adap/flower/pull/2273), [#2267](https://github.com/adap/flower/pull/2267), [#2274](https://github.com/adap/flower/pull/2274), [#2275](https://github.com/adap/flower/pull/2275), [#2432](https://github.com/adap/flower/pull/2432), [#2251](https://github.com/adap/flower/pull/2251), [#2321](https://github.com/adap/flower/pull/2321), [#1936](https://github.com/adap/flower/pull/1936), [#2408](https://github.com/adap/flower/pull/2408), [#2413](https://github.com/adap/flower/pull/2413), [#2401](https://github.com/adap/flower/pull/2401), [#2531](https://github.com/adap/flower/pull/2531), [#2534](https://github.com/adap/flower/pull/2534), [#2535](https://github.com/adap/flower/pull/2535), [#2521](https://github.com/adap/flower/pull/2521), [#2553](https://github.com/adap/flower/pull/2553), [#2596](https://github.com/adap/flower/pull/2596))

  Flower received many improvements under the hood, too many to list here.

### Incompatible changes

- **Remove support for Python 3.7** ([#2280](https://github.com/adap/flower/pull/2280), [#2299](https://github.com/adap/flower/pull/2299), [#2304](https://github.com/adap/flower/pull/2304), [#2306](https://github.com/adap/flower/pull/2306), [#2355](https://github.com/adap/flower/pull/2355), [#2356](https://github.com/adap/flower/pull/2356))

  Python 3.7 support was deprecated in Flower 1.5, and this release removes support. Flower now requires Python 3.8.

- **Remove experimental argument** `rest` **from** `start_client` ([#2324](https://github.com/adap/flower/pull/2324))

  The (still experimental) argument `rest` was removed from `start_client` and `start_numpy_client`. Use `transport="rest"` to opt into the experimental REST API instead.

## v1.5.0 (2023-08-31)

### Thanks to our contributors

We would like to give our special thanks to all the contributors who made the new version of Flower possible (in `git shortlog` order):

`Adam Narozniak`, `Anass Anhari`, `Charles Beauville`, `Dana-Farber`, `Daniel J. Beutel`, `Daniel Nata Nugraha`, `Edoardo Gabrielli`, `Gustavo Bertoli`, `Heng Pan`, `Javier`, `Mahdi`, `Steven Hé (Sīchàng)`, `Taner Topal`, `achiverram28`, `danielnugraha`, `eunchung`, `ruthgal` <!---TOKEN_v1.5.0-->

### What's new?

- **Introduce new simulation engine** ([#1969](https://github.com/adap/flower/pull/1969), [#2221](https://github.com/adap/flower/pull/2221), [#2248](https://github.com/adap/flower/pull/2248))

  The new simulation engine has been rewritten from the ground up, yet it remains fully backwards compatible. It offers much improved stability and memory handling, especially when working with GPUs. Simulations transparently adapt to different settings to scale simulation in CPU-only, CPU+GPU, multi-GPU, or multi-node multi-GPU environments.

  Comprehensive documentation includes a new [how-to run simulations](https://flower.ai/docs/framework/how-to-run-simulations.html) guide, new [simulation-pytorch](https://flower.ai/docs/examples/simulation-pytorch.html) and [simulation-tensorflow](https://flower.ai/docs/examples/simulation-tensorflow.html) notebooks, and a new [YouTube tutorial series](https://www.youtube.com/watch?v=cRebUIGB5RU&list=PLNG4feLHqCWlnj8a_E1A_n5zr2-8pafTB).

- **Restructure Flower Docs** ([#1824](https://github.com/adap/flower/pull/1824), [#1865](https://github.com/adap/flower/pull/1865), [#1884](https://github.com/adap/flower/pull/1884), [#1887](https://github.com/adap/flower/pull/1887), [#1919](https://github.com/adap/flower/pull/1919), [#1922](https://github.com/adap/flower/pull/1922), [#1920](https://github.com/adap/flower/pull/1920), [#1923](https://github.com/adap/flower/pull/1923), [#1924](https://github.com/adap/flower/pull/1924), [#1962](https://github.com/adap/flower/pull/1962), [#2006](https://github.com/adap/flower/pull/2006), [#2133](https://github.com/adap/flower/pull/2133), [#2203](https://github.com/adap/flower/pull/2203), [#2215](https://github.com/adap/flower/pull/2215), [#2122](https://github.com/adap/flower/pull/2122), [#2223](https://github.com/adap/flower/pull/2223), [#2219](https://github.com/adap/flower/pull/2219), [#2232](https://github.com/adap/flower/pull/2232), [#2233](https://github.com/adap/flower/pull/2233), [#2234](https://github.com/adap/flower/pull/2234), [#2235](https://github.com/adap/flower/pull/2235), [#2237](https://github.com/adap/flower/pull/2237), [#2238](https://github.com/adap/flower/pull/2238), [#2242](https://github.com/adap/flower/pull/2242), [#2231](https://github.com/adap/flower/pull/2231), [#2243](https://github.com/adap/flower/pull/2243), [#2227](https://github.com/adap/flower/pull/2227))

  Much effort went into a completely restructured Flower docs experience. The documentation on [flower.ai/docs](https://flower.ai/docs) is now divided into Flower Framework, Flower Baselines, Flower Android SDK, Flower iOS SDK, and code example projects.

- **Introduce Flower Swift SDK** ([#1858](https://github.com/adap/flower/pull/1858), [#1897](https://github.com/adap/flower/pull/1897))

  This is the first preview release of the Flower Swift SDK. Flower support on iOS is improving, and alongside the Swift SDK and code example, there is now also an iOS quickstart tutorial.

- **Introduce Flower Android SDK** ([#2131](https://github.com/adap/flower/pull/2131))

  This is the first preview release of the Flower Kotlin SDK. Flower support on Android is improving, and alongside the Kotlin SDK and code example, there is now also an Android quickstart tutorial.

- **Introduce new end-to-end testing infrastructure** ([#1842](https://github.com/adap/flower/pull/1842), [#2071](https://github.com/adap/flower/pull/2071), [#2072](https://github.com/adap/flower/pull/2072), [#2068](https://github.com/adap/flower/pull/2068), [#2067](https://github.com/adap/flower/pull/2067), [#2069](https://github.com/adap/flower/pull/2069), [#2073](https://github.com/adap/flower/pull/2073), [#2070](https://github.com/adap/flower/pull/2070), [#2074](https://github.com/adap/flower/pull/2074), [#2082](https://github.com/adap/flower/pull/2082), [#2084](https://github.com/adap/flower/pull/2084), [#2093](https://github.com/adap/flower/pull/2093), [#2109](https://github.com/adap/flower/pull/2109), [#2095](https://github.com/adap/flower/pull/2095), [#2140](https://github.com/adap/flower/pull/2140), [#2137](https://github.com/adap/flower/pull/2137), [#2165](https://github.com/adap/flower/pull/2165))

  A new testing infrastructure ensures that new changes stay compatible with existing framework integrations or strategies.

- **Deprecate Python 3.7**

  Since Python 3.7 reached its end of life (EOL) on 2023-06-27, support for Python 3.7 is now deprecated and will be removed in an upcoming release.

- **Add new** `FedTrimmedAvg` **strategy** ([#1769](https://github.com/adap/flower/pull/1769), [#1853](https://github.com/adap/flower/pull/1853))

  The new `FedTrimmedAvg` strategy implements Trimmed Mean by [Dong Yin, 2018](https://arxiv.org/abs/1803.01498).

- **Introduce start_driver** ([#1697](https://github.com/adap/flower/pull/1697))

  In addition to `start_server` and using the raw Driver API, there is a new `start_driver` function that allows for running `start_server` scripts as a Flower driver with only a single-line code change. Check out the `mt-pytorch` code example to see a working example using `start_driver`.

- **Add parameter aggregation to** `mt-pytorch` **code example** ([#1785](https://github.com/adap/flower/pull/1785))

  The `mt-pytorch` example shows how to aggregate parameters when writing a driver script. The included `driver.py` and `server.py` have been aligned to demonstrate both the low-level way and the high-level way of building server-side logic.

- **Migrate experimental REST API to Starlette** ([2171](https://github.com/adap/flower/pull/2171))

  The (experimental) REST API used to be implemented in [FastAPI](https://fastapi.tiangolo.com/), but it has now been migrated to use [Starlette](https://www.starlette.io/) directly.

  Please note: The REST request-response API is still experimental and will likely change significantly over time.

- **Introduce experimental gRPC request-response API** ([#1867](https://github.com/adap/flower/pull/1867), [#1901](https://github.com/adap/flower/pull/1901))

  In addition to the existing gRPC API (based on bidirectional streaming) and the experimental REST API, there is now a new gRPC API that uses a request-response model to communicate with client nodes.

  Please note: The gRPC request-response API is still experimental and will likely change significantly over time.

- **Replace the experimental** `start_client(rest=True)` **with the new** `start_client(transport="rest")` ([#1880](https://github.com/adap/flower/pull/1880))

  The (experimental) `start_client` argument `rest` was deprecated in favour of a new argument `transport`. `start_client(transport="rest")` will yield the same behaviour as `start_client(rest=True)` did before. All code should migrate to the new argument `transport`. The deprecated argument `rest` will be removed in a future release.

- **Add a new gRPC option** ([#2197](https://github.com/adap/flower/pull/2197))

  We now start a gRPC server with the `grpc.keepalive_permit_without_calls` option set to 0 by default. This prevents the clients from sending keepalive pings when there is no outstanding stream.

- **Improve example notebooks** ([#2005](https://github.com/adap/flower/pull/2005))

  There's a new 30min Federated Learning PyTorch tutorial!

- **Example updates** ([#1772](https://github.com/adap/flower/pull/1772), [#1873](https://github.com/adap/flower/pull/1873), [#1981](https://github.com/adap/flower/pull/1981), [#1988](https://github.com/adap/flower/pull/1988), [#1984](https://github.com/adap/flower/pull/1984), [#1982](https://github.com/adap/flower/pull/1982), [#2112](https://github.com/adap/flower/pull/2112), [#2144](https://github.com/adap/flower/pull/2144), [#2174](https://github.com/adap/flower/pull/2174), [#2225](https://github.com/adap/flower/pull/2225), [#2183](https://github.com/adap/flower/pull/2183))

  Many examples have received significant updates, including simplified advanced-tensorflow and advanced-pytorch examples, improved macOS compatibility of TensorFlow examples, and code examples for simulation. A major upgrade is that all code examples now have a `requirements.txt` (in addition to `pyproject.toml`).

- **General improvements** ([#1872](https://github.com/adap/flower/pull/1872), [#1866](https://github.com/adap/flower/pull/1866), [#1884](https://github.com/adap/flower/pull/1884), [#1837](https://github.com/adap/flower/pull/1837), [#1477](https://github.com/adap/flower/pull/1477), [#2171](https://github.com/adap/flower/pull/2171))

  Flower received many improvements under the hood, too many to list here.

### Incompatible changes

None

## v1.4.0 (2023-04-21)

### Thanks to our contributors

We would like to give our special thanks to all the contributors who made the new version of Flower possible (in `git shortlog` order):

`Adam Narozniak`, `Alexander Viala Bellander`, `Charles Beauville`, `Chenyang Ma (Danny)`, `Daniel J. Beutel`, `Edoardo`, `Gautam Jajoo`, `Iacob-Alexandru-Andrei`, `JDRanpariya`, `Jean Charle Yaacoub`, `Kunal Sarkhel`, `L. Jiang`, `Lennart Behme`, `Max Kapsecker`, `Michał`, `Nic Lane`, `Nikolaos Episkopos`, `Ragy`, `Saurav Maheshkar`, `Semo Yang`, `Steve Laskaridis`, `Steven Hé (Sīchàng)`, `Taner Topal`

### What's new?

- **Introduce support for XGBoost (**`FedXgbNnAvg` **strategy and example)** ([#1694](https://github.com/adap/flower/pull/1694), [#1709](https://github.com/adap/flower/pull/1709), [#1715](https://github.com/adap/flower/pull/1715), [#1717](https://github.com/adap/flower/pull/1717), [#1763](https://github.com/adap/flower/pull/1763), [#1795](https://github.com/adap/flower/pull/1795))

  XGBoost is a tree-based ensemble machine learning algorithm that uses gradient boosting to improve model accuracy. We added a new `FedXgbNnAvg` [strategy](https://github.com/adap/flower/tree/main/src/py/flwr/server/strategy/fedxgb_nn_avg.py), and a [code example](https://github.com/adap/flower/tree/main/examples/xgboost-quickstart) that demonstrates the usage of this new strategy in an XGBoost project.

- **Introduce iOS SDK (preview)** ([#1621](https://github.com/adap/flower/pull/1621), [#1764](https://github.com/adap/flower/pull/1764))

  This is a major update for anyone wanting to implement Federated Learning on iOS mobile devices. We now have a swift iOS SDK present under [src/swift/flwr](https://github.com/adap/flower/tree/main/src/swift/flwr) that will facilitate greatly the app creating process. To showcase its use, the [iOS example](https://github.com/adap/flower/tree/main/examples/ios) has also been updated!

- **Introduce new "What is Federated Learning?" tutorial** ([#1657](https://github.com/adap/flower/pull/1657), [#1721](https://github.com/adap/flower/pull/1721))

  A new [entry-level tutorial](https://flower.ai/docs/framework/tutorial-what-is-federated-learning.html) in our documentation explains the basics of Fedetated Learning. It enables anyone who's unfamiliar with Federated Learning to start their journey with Flower. Forward it to anyone who's interested in Federated Learning!

- **Introduce new Flower Baseline: FedProx MNIST** ([#1513](https://github.com/adap/flower/pull/1513), [#1680](https://github.com/adap/flower/pull/1680), [#1681](https://github.com/adap/flower/pull/1681), [#1679](https://github.com/adap/flower/pull/1679))

  This new baseline replicates the MNIST+CNN task from the paper [Federated Optimization in Heterogeneous Networks (Li et al., 2018)](https://arxiv.org/abs/1812.06127). It uses the `FedProx` strategy, which aims at making convergence more robust in heterogeneous settings.

- **Introduce new Flower Baseline: FedAvg FEMNIST** ([#1655](https://github.com/adap/flower/pull/1655))

  This new baseline replicates an experiment evaluating the performance of the FedAvg algorithm on the FEMNIST dataset from the paper [LEAF: A Benchmark for Federated Settings (Caldas et al., 2018)](https://arxiv.org/abs/1812.01097).

- **Introduce (experimental) REST API** ([#1594](https://github.com/adap/flower/pull/1594), [#1690](https://github.com/adap/flower/pull/1690), [#1695](https://github.com/adap/flower/pull/1695), [#1712](https://github.com/adap/flower/pull/1712), [#1802](https://github.com/adap/flower/pull/1802), [#1770](https://github.com/adap/flower/pull/1770), [#1733](https://github.com/adap/flower/pull/1733))

  A new REST API has been introduced as an alternative to the gRPC-based communication stack. In this initial version, the REST API only supports anonymous clients.

  Please note: The REST API is still experimental and will likely change significantly over time.

- **Improve the (experimental) Driver API** ([#1663](https://github.com/adap/flower/pull/1663), [#1666](https://github.com/adap/flower/pull/1666), [#1667](https://github.com/adap/flower/pull/1667), [#1664](https://github.com/adap/flower/pull/1664), [#1675](https://github.com/adap/flower/pull/1675), [#1676](https://github.com/adap/flower/pull/1676), [#1693](https://github.com/adap/flower/pull/1693), [#1662](https://github.com/adap/flower/pull/1662), [#1794](https://github.com/adap/flower/pull/1794))

  The Driver API is still an experimental feature, but this release introduces some major upgrades. One of the main improvements is the introduction of an SQLite database to store server state on disk (instead of in-memory). Another improvement is that tasks (instructions or results) that have been delivered will now be deleted. This greatly improves the memory efficiency of a long-running Flower server.

- **Fix spilling issues related to Ray during simulations** ([#1698](https://github.com/adap/flower/pull/1698))

  While running long simulations, `ray` was sometimes spilling huge amounts of data that would make the training unable to continue. This is now fixed! 🎉

- **Add new example using** `TabNet` **and Flower** ([#1725](https://github.com/adap/flower/pull/1725))

  TabNet is a powerful and flexible framework for training machine learning models on tabular data. We now have a federated example using Flower: [quickstart-tabnet](https://github.com/adap/flower/tree/main/examples/quickstart-tabnet).

- **Add new how-to guide for monitoring simulations** ([#1649](https://github.com/adap/flower/pull/1649))

  We now have a documentation guide to help users monitor their performance during simulations.

- **Add training metrics to** `History` **object during simulations** ([#1696](https://github.com/adap/flower/pull/1696))

  The `fit_metrics_aggregation_fn` can be used to aggregate training metrics, but previous releases did not save the results in the `History` object. This is now the case!

- **General improvements** ([#1659](https://github.com/adap/flower/pull/1659), [#1646](https://github.com/adap/flower/pull/1646), [#1647](https://github.com/adap/flower/pull/1647), [#1471](https://github.com/adap/flower/pull/1471), [#1648](https://github.com/adap/flower/pull/1648), [#1651](https://github.com/adap/flower/pull/1651), [#1652](https://github.com/adap/flower/pull/1652), [#1653](https://github.com/adap/flower/pull/1653), [#1659](https://github.com/adap/flower/pull/1659), [#1665](https://github.com/adap/flower/pull/1665), [#1670](https://github.com/adap/flower/pull/1670), [#1672](https://github.com/adap/flower/pull/1672), [#1677](https://github.com/adap/flower/pull/1677), [#1684](https://github.com/adap/flower/pull/1684), [#1683](https://github.com/adap/flower/pull/1683), [#1686](https://github.com/adap/flower/pull/1686), [#1682](https://github.com/adap/flower/pull/1682), [#1685](https://github.com/adap/flower/pull/1685), [#1692](https://github.com/adap/flower/pull/1692), [#1705](https://github.com/adap/flower/pull/1705), [#1708](https://github.com/adap/flower/pull/1708), [#1711](https://github.com/adap/flower/pull/1711), [#1713](https://github.com/adap/flower/pull/1713), [#1714](https://github.com/adap/flower/pull/1714), [#1718](https://github.com/adap/flower/pull/1718), [#1716](https://github.com/adap/flower/pull/1716), [#1723](https://github.com/adap/flower/pull/1723), [#1735](https://github.com/adap/flower/pull/1735), [#1678](https://github.com/adap/flower/pull/1678), [#1750](https://github.com/adap/flower/pull/1750), [#1753](https://github.com/adap/flower/pull/1753), [#1736](https://github.com/adap/flower/pull/1736), [#1766](https://github.com/adap/flower/pull/1766), [#1760](https://github.com/adap/flower/pull/1760), [#1775](https://github.com/adap/flower/pull/1775), [#1776](https://github.com/adap/flower/pull/1776), [#1777](https://github.com/adap/flower/pull/1777), [#1779](https://github.com/adap/flower/pull/1779), [#1784](https://github.com/adap/flower/pull/1784), [#1773](https://github.com/adap/flower/pull/1773), [#1755](https://github.com/adap/flower/pull/1755), [#1789](https://github.com/adap/flower/pull/1789), [#1788](https://github.com/adap/flower/pull/1788), [#1798](https://github.com/adap/flower/pull/1798), [#1799](https://github.com/adap/flower/pull/1799), [#1739](https://github.com/adap/flower/pull/1739), [#1800](https://github.com/adap/flower/pull/1800), [#1804](https://github.com/adap/flower/pull/1804), [#1805](https://github.com/adap/flower/pull/1805))

  Flower received many improvements under the hood, too many to list here.

### Incompatible changes

None

## v1.3.0 (2023-02-06)

### Thanks to our contributors

We would like to give our special thanks to all the contributors who made the new version of Flower possible (in `git shortlog` order):

`Adam Narozniak`, `Alexander Viala Bellander`, `Charles Beauville`, `Daniel J. Beutel`, `JDRanpariya`, `Lennart Behme`, `Taner Topal`

### What's new?

- **Add support for** `workload_id` **and** `group_id` **in Driver API** ([#1595](https://github.com/adap/flower/pull/1595))

  The (experimental) Driver API now supports a `workload_id` that can be used to identify which workload a task belongs to. It also supports a new `group_id` that can be used, for example, to indicate the current training round. Both the `workload_id` and `group_id` enable client nodes to decide whether they want to handle a task or not.

- **Make Driver API and Fleet API address configurable** ([#1637](https://github.com/adap/flower/pull/1637))

  The (experimental) long-running Flower server (Driver API and Fleet API) can now configure the server address of both Driver API (via `--driver-api-address`) and Fleet API (via `--fleet-api-address`) when starting:

  `flower-server --driver-api-address "0.0.0.0:8081" --fleet-api-address "0.0.0.0:8086"`

  Both IPv4 and IPv6 addresses are supported.

- **Add new example of Federated Learning using fastai and Flower** ([#1598](https://github.com/adap/flower/pull/1598))

  A new code example (`quickstart-fastai`) demonstrates federated learning with [fastai](https://www.fast.ai/) and Flower. You can find it here: [quickstart-fastai](https://github.com/adap/flower/tree/main/examples/quickstart-fastai).

- **Make Android example compatible with** `flwr >= 1.0.0` **and the latest versions of Android** ([#1603](https://github.com/adap/flower/pull/1603))

  The Android code example has received a substantial update: the project is compatible with Flower 1.0 (and later), the UI received a full refresh, and the project is updated to be compatible with newer Android tooling.

- **Add new `FedProx` strategy** ([#1619](https://github.com/adap/flower/pull/1619))

  This [strategy](https://github.com/adap/flower/blob/main/src/py/flwr/server/strategy/fedprox.py) is almost identical to [`FedAvg`](https://github.com/adap/flower/blob/main/src/py/flwr/server/strategy/fedavg.py), but helps users replicate what is described in this [paper](https://arxiv.org/abs/1812.06127). It essentially adds a parameter called `proximal_mu` to regularize the local models with respect to the global models.

- **Add new metrics to telemetry events** ([#1640](https://github.com/adap/flower/pull/1640))

  An updated event structure allows, for example, the clustering of events within the same workload.

- **Add new custom strategy tutorial section** [#1623](https://github.com/adap/flower/pull/1623)

  The Flower tutorial now has a new section that covers implementing a custom strategy from scratch: [Open in Colab](https://colab.research.google.com/github/adap/flower/blob/main/doc/source/tutorial-build-a-strategy-from-scratch-pytorch.ipynb)

- **Add new custom serialization tutorial section** ([#1622](https://github.com/adap/flower/pull/1622))

  The Flower tutorial now has a new section that covers custom serialization: [Open in Colab](https://colab.research.google.com/github/adap/flower/blob/main/doc/source/tutorial-customize-the-client-pytorch.ipynb)

- **General improvements** ([#1638](https://github.com/adap/flower/pull/1638), [#1634](https://github.com/adap/flower/pull/1634), [#1636](https://github.com/adap/flower/pull/1636), [#1635](https://github.com/adap/flower/pull/1635), [#1633](https://github.com/adap/flower/pull/1633), [#1632](https://github.com/adap/flower/pull/1632), [#1631](https://github.com/adap/flower/pull/1631), [#1630](https://github.com/adap/flower/pull/1630), [#1627](https://github.com/adap/flower/pull/1627), [#1593](https://github.com/adap/flower/pull/1593), [#1616](https://github.com/adap/flower/pull/1616), [#1615](https://github.com/adap/flower/pull/1615), [#1607](https://github.com/adap/flower/pull/1607), [#1609](https://github.com/adap/flower/pull/1609), [#1608](https://github.com/adap/flower/pull/1608), [#1603](https://github.com/adap/flower/pull/1603), [#1590](https://github.com/adap/flower/pull/1590), [#1580](https://github.com/adap/flower/pull/1580), [#1599](https://github.com/adap/flower/pull/1599), [#1600](https://github.com/adap/flower/pull/1600), [#1601](https://github.com/adap/flower/pull/1601), [#1597](https://github.com/adap/flower/pull/1597), [#1595](https://github.com/adap/flower/pull/1595), [#1591](https://github.com/adap/flower/pull/1591), [#1588](https://github.com/adap/flower/pull/1588), [#1589](https://github.com/adap/flower/pull/1589), [#1587](https://github.com/adap/flower/pull/1587), [#1573](https://github.com/adap/flower/pull/1573), [#1581](https://github.com/adap/flower/pull/1581), [#1578](https://github.com/adap/flower/pull/1578), [#1574](https://github.com/adap/flower/pull/1574), [#1572](https://github.com/adap/flower/pull/1572), [#1586](https://github.com/adap/flower/pull/1586))

  Flower received many improvements under the hood, too many to list here.

- **Updated documentation** ([#1629](https://github.com/adap/flower/pull/1629), [#1628](https://github.com/adap/flower/pull/1628), [#1620](https://github.com/adap/flower/pull/1620), [#1618](https://github.com/adap/flower/pull/1618), [#1617](https://github.com/adap/flower/pull/1617), [#1613](https://github.com/adap/flower/pull/1613), [#1614](https://github.com/adap/flower/pull/1614))

  As usual, the documentation has improved quite a bit. It is another step in our effort to make the Flower documentation the best documentation of any project. Stay tuned and as always, feel free to provide feedback!

### Incompatible changes

None

## v1.2.0 (2023-01-13)

### Thanks to our contributors

We would like to give our special thanks to all the contributors who made the new version of Flower possible (in `git shortlog` order):

`Adam Narozniak`, `Charles Beauville`, `Daniel J. Beutel`, `Edoardo`, `L. Jiang`, `Ragy`, `Taner Topal`, `dannymcy`

### What's new?

- **Introduce new Flower Baseline: FedAvg MNIST** ([#1497](https://github.com/adap/flower/pull/1497), [#1552](https://github.com/adap/flower/pull/1552))

  Over the coming weeks, we will be releasing a number of new reference implementations useful especially to FL newcomers. They will typically revisit well known papers from the literature, and be suitable for integration in your own application or for experimentation, in order to deepen your knowledge of FL in general. Today's release is the first in this series. [Read more.](https://flower.ai/blog/2023-01-12-fl-starter-pack-fedavg-mnist-cnn/)

- **Improve GPU support in simulations** ([#1555](https://github.com/adap/flower/pull/1555))

  The Ray-based Virtual Client Engine (`start_simulation`) has been updated to improve GPU support. The update includes some of the hard-earned lessons from scaling simulations in GPU cluster environments. New defaults make running GPU-based simulations substantially more robust.

- **Improve GPU support in Jupyter Notebook tutorials** ([#1527](https://github.com/adap/flower/pull/1527), [#1558](https://github.com/adap/flower/pull/1558))

  Some users reported that Jupyter Notebooks have not always been easy to use on GPU instances. We listened and made improvements to all of our Jupyter notebooks! Check out the updated notebooks here:

  - [An Introduction to Federated Learning](https://flower.ai/docs/framework/tutorial-get-started-with-flower-pytorch.html)
  - [Strategies in Federated Learning](https://flower.ai/docs/framework/tutorial-use-a-federated-learning-strategy-pytorch.html)
  - [Building a Strategy](https://flower.ai/docs/framework/tutorial-build-a-strategy-from-scratch-pytorch.html)
  - [Client and NumPyClient](https://flower.ai/docs/framework/tutorial-customize-the-client-pytorch.html)

- **Introduce optional telemetry** ([#1533](https://github.com/adap/flower/pull/1533), [#1544](https://github.com/adap/flower/pull/1544), [#1584](https://github.com/adap/flower/pull/1584))

  After a [request for feedback](https://github.com/adap/flower/issues/1534) from the community, the Flower open-source project introduces optional collection of *anonymous* usage metrics to make well-informed decisions to improve Flower. Doing this enables the Flower team to understand how Flower is used and what challenges users might face.

  **Flower is a friendly framework for collaborative AI and data science.** Staying true to this statement, Flower makes it easy to disable telemetry for users who do not want to share anonymous usage metrics. [Read more.](https://flower.ai/docs/telemetry.html).

- **Introduce (experimental) Driver API** ([#1520](https://github.com/adap/flower/pull/1520), [#1525](https://github.com/adap/flower/pull/1525), [#1545](https://github.com/adap/flower/pull/1545), [#1546](https://github.com/adap/flower/pull/1546), [#1550](https://github.com/adap/flower/pull/1550), [#1551](https://github.com/adap/flower/pull/1551), [#1567](https://github.com/adap/flower/pull/1567))

  Flower now has a new (experimental) Driver API which will enable fully programmable, async, and multi-tenant Federated Learning and Federated Analytics applications. Phew, that's a lot! Going forward, the Driver API will be the abstraction that many upcoming features will be built on - and you can start building those things now, too.

  The Driver API also enables a new execution mode in which the server runs indefinitely. Multiple individual workloads can run concurrently and start and stop their execution independent of the server. This is especially useful for users who want to deploy Flower in production.

  To learn more, check out the `mt-pytorch` code example. We look forward to you feedback!

  Please note: *The Driver API is still experimental and will likely change significantly over time.*

- **Add new Federated Analytics with Pandas example** ([#1469](https://github.com/adap/flower/pull/1469), [#1535](https://github.com/adap/flower/pull/1535))

  A new code example (`quickstart-pandas`) demonstrates federated analytics with Pandas and Flower. You can find it here: [quickstart-pandas](https://github.com/adap/flower/tree/main/examples/quickstart-pandas).

- **Add new strategies: Krum and MultiKrum** ([#1481](https://github.com/adap/flower/pull/1481))

  Edoardo, a computer science student at the Sapienza University of Rome, contributed a new `Krum` strategy that enables users to easily use Krum and MultiKrum in their workloads.

- **Update C++ example to be compatible with Flower v1.2.0** ([#1495](https://github.com/adap/flower/pull/1495))

  The C++ code example has received a substantial update to make it compatible with the latest version of Flower.

- **General improvements** ([#1491](https://github.com/adap/flower/pull/1491), [#1504](https://github.com/adap/flower/pull/1504), [#1506](https://github.com/adap/flower/pull/1506), [#1514](https://github.com/adap/flower/pull/1514), [#1522](https://github.com/adap/flower/pull/1522), [#1523](https://github.com/adap/flower/pull/1523), [#1526](https://github.com/adap/flower/pull/1526), [#1528](https://github.com/adap/flower/pull/1528), [#1547](https://github.com/adap/flower/pull/1547), [#1549](https://github.com/adap/flower/pull/1549), [#1560](https://github.com/adap/flower/pull/1560), [#1564](https://github.com/adap/flower/pull/1564), [#1566](https://github.com/adap/flower/pull/1566))

  Flower received many improvements under the hood, too many to list here.

- **Updated documentation** ([#1494](https://github.com/adap/flower/pull/1494), [#1496](https://github.com/adap/flower/pull/1496), [#1500](https://github.com/adap/flower/pull/1500), [#1503](https://github.com/adap/flower/pull/1503), [#1505](https://github.com/adap/flower/pull/1505), [#1524](https://github.com/adap/flower/pull/1524), [#1518](https://github.com/adap/flower/pull/1518), [#1519](https://github.com/adap/flower/pull/1519), [#1515](https://github.com/adap/flower/pull/1515))

  As usual, the documentation has improved quite a bit. It is another step in our effort to make the Flower documentation the best documentation of any project. Stay tuned and as always, feel free to provide feedback!

  One highlight is the new [first time contributor guide](https://flower.ai/docs/first-time-contributors.html): if you've never contributed on GitHub before, this is the perfect place to start!

### Incompatible changes

None

## v1.1.0 (2022-10-31)

### Thanks to our contributors

We would like to give our **special thanks** to all the contributors who made the new version of Flower possible (in `git shortlog` order):

`Akis Linardos`, `Christopher S`, `Daniel J. Beutel`, `George`, `Jan Schlicht`, `Mohammad Fares`, `Pedro Porto Buarque de Gusmão`, `Philipp Wiesner`, `Rob Luke`, `Taner Topal`, `VasundharaAgarwal`, `danielnugraha`, `edogab33`

### What's new?

- **Introduce Differential Privacy wrappers (preview)** ([#1357](https://github.com/adap/flower/pull/1357), [#1460](https://github.com/adap/flower/pull/1460))

  The first (experimental) preview of pluggable Differential Privacy wrappers enables easy configuration and usage of differential privacy (DP). The pluggable DP wrappers enable framework-agnostic **and** strategy-agnostic usage of both client-side DP and server-side DP. Head over to the Flower docs, a new explainer goes into more detail.

- **New iOS CoreML code example** ([#1289](https://github.com/adap/flower/pull/1289))

  Flower goes iOS! A massive new code example shows how Flower clients can be built for iOS. The code example contains both Flower iOS SDK components that can be used for many tasks, and one task example running on CoreML.

- **New FedMedian strategy** ([#1461](https://github.com/adap/flower/pull/1461))

  The new `FedMedian` strategy implements Federated Median (FedMedian) by [Yin et al., 2018](https://arxiv.org/pdf/1803.01498v1.pdf).

- **Log** `Client` **exceptions in Virtual Client Engine** ([#1493](https://github.com/adap/flower/pull/1493))

  All `Client` exceptions happening in the VCE are now logged by default and not just exposed to the configured `Strategy` (via the `failures` argument).

- **Improve Virtual Client Engine internals** ([#1401](https://github.com/adap/flower/pull/1401), [#1453](https://github.com/adap/flower/pull/1453))

  Some internals of the Virtual Client Engine have been revamped. The VCE now uses Ray 2.0 under the hood, the value type of the `client_resources` dictionary changed to `float` to allow fractions of resources to be allocated.

- **Support optional** `Client`**/**`NumPyClient` **methods in Virtual Client Engine**

  The Virtual Client Engine now has full support for optional `Client` (and `NumPyClient`) methods.

- **Provide type information to packages using** `flwr` ([#1377](https://github.com/adap/flower/pull/1377))

  The package `flwr` is now bundled with a `py.typed` file indicating that the package is typed. This enables typing support for projects or packages that use `flwr` by enabling them to improve their code using static type checkers like `mypy`.

- **Updated code example** ([#1344](https://github.com/adap/flower/pull/1344), [#1347](https://github.com/adap/flower/pull/1347))

  The code examples covering scikit-learn and PyTorch Lightning have been updated to work with the latest version of Flower.

- **Updated documentation** ([#1355](https://github.com/adap/flower/pull/1355), [#1558](https://github.com/adap/flower/pull/1558), [#1379](https://github.com/adap/flower/pull/1379), [#1380](https://github.com/adap/flower/pull/1380), [#1381](https://github.com/adap/flower/pull/1381), [#1332](https://github.com/adap/flower/pull/1332), [#1391](https://github.com/adap/flower/pull/1391), [#1403](https://github.com/adap/flower/pull/1403), [#1364](https://github.com/adap/flower/pull/1364), [#1409](https://github.com/adap/flower/pull/1409), [#1419](https://github.com/adap/flower/pull/1419), [#1444](https://github.com/adap/flower/pull/1444), [#1448](https://github.com/adap/flower/pull/1448), [#1417](https://github.com/adap/flower/pull/1417), [#1449](https://github.com/adap/flower/pull/1449), [#1465](https://github.com/adap/flower/pull/1465), [#1467](https://github.com/adap/flower/pull/1467))

  There have been so many documentation updates that it doesn't even make sense to list them individually.

- **Restructured documentation** ([#1387](https://github.com/adap/flower/pull/1387))

  The documentation has been restructured to make it easier to navigate. This is just the first step in a larger effort to make the Flower documentation the best documentation of any project ever. Stay tuned!

- **Open in Colab button** ([#1389](https://github.com/adap/flower/pull/1389))

  The four parts of the Flower Federated Learning Tutorial now come with a new `Open in Colab` button. No need to install anything on your local machine, you can now use and learn about Flower in your browser, it's only a single click away.

- **Improved tutorial** ([#1468](https://github.com/adap/flower/pull/1468), [#1470](https://github.com/adap/flower/pull/1470), [#1472](https://github.com/adap/flower/pull/1472), [#1473](https://github.com/adap/flower/pull/1473), [#1474](https://github.com/adap/flower/pull/1474), [#1475](https://github.com/adap/flower/pull/1475))

  The Flower Federated Learning Tutorial has two brand-new parts covering custom strategies (still WIP) and the distinction between `Client` and `NumPyClient`. The existing parts one and two have also been improved (many small changes and fixes).

### Incompatible changes

None

## v1.0.0 (2022-07-28)

### Highlights

- Stable **Virtual Client Engine** (accessible via `start_simulation`)
- All `Client`/`NumPyClient` methods are now optional
- Configurable `get_parameters`
- Tons of small API cleanups resulting in a more coherent developer experience

### Thanks to our contributors

We would like to give our **special thanks** to all the contributors who made Flower 1.0 possible (in reverse [GitHub Contributors](https://github.com/adap/flower/graphs/contributors) order):

[@rtaiello](https://github.com/rtaiello), [@g-pichler](https://github.com/g-pichler), [@rob-luke](https://github.com/rob-luke), [@andreea-zaharia](https://github.com/andreea-zaharia), [@kinshukdua](https://github.com/kinshukdua), [@nfnt](https://github.com/nfnt), [@tatiana-s](https://github.com/tatiana-s), [@TParcollet](https://github.com/TParcollet), [@vballoli](https://github.com/vballoli), [@negedng](https://github.com/negedng), [@RISHIKESHAVAN](https://github.com/RISHIKESHAVAN), [@hei411](https://github.com/hei411), [@SebastianSpeitel](https://github.com/SebastianSpeitel), [@AmitChaulwar](https://github.com/AmitChaulwar), [@Rubiel1](https://github.com/Rubiel1), [@FANTOME-PAN](https://github.com/FANTOME-PAN), [@Rono-BC](https://github.com/Rono-BC), [@lbhm](https://github.com/lbhm), [@sishtiaq](https://github.com/sishtiaq), [@remde](https://github.com/remde), [@Jueun-Park](https://github.com/Jueun-Park), [@architjen](https://github.com/architjen), [@PratikGarai](https://github.com/PratikGarai), [@mrinaald](https://github.com/mrinaald), [@zliel](https://github.com/zliel), [@MeiruiJiang](https://github.com/MeiruiJiang), [@sancarlim](https://github.com/sancarlim), [@gubertoli](https://github.com/gubertoli), [@Vingt100](https://github.com/Vingt100), [@MakGulati](https://github.com/MakGulati), [@cozek](https://github.com/cozek), [@jafermarq](https://github.com/jafermarq), [@sisco0](https://github.com/sisco0), [@akhilmathurs](https://github.com/akhilmathurs), [@CanTuerk](https://github.com/CanTuerk), [@mariaboerner1987](https://github.com/mariaboerner1987), [@pedropgusmao](https://github.com/pedropgusmao), [@tanertopal](https://github.com/tanertopal), [@danieljanes](https://github.com/danieljanes).

### Incompatible changes

- **All arguments must be passed as keyword arguments** ([#1338](https://github.com/adap/flower/pull/1338))

  Pass all arguments as keyword arguments, positional arguments are not longer supported. Code that uses positional arguments (e.g., `start_client("127.0.0.1:8080", FlowerClient())`) must add the keyword for each positional argument (e.g., `start_client(server_address="127.0.0.1:8080", client=FlowerClient())`).

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

### What's new?

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

### What's new?

- **Flower Baselines (preview): FedOpt, FedBN, FedAvgM** ([#919](https://github.com/adap/flower/pull/919), [#1127](https://github.com/adap/flower/pull/1127), [#914](https://github.com/adap/flower/pull/914))

  The first preview release of Flower Baselines has arrived! We're kickstarting Flower Baselines with implementations of FedOpt (FedYogi, FedAdam, FedAdagrad), FedBN, and FedAvgM. Check the documentation on how to use [Flower Baselines](https://flower.ai/docs/using-baselines.html). With this first preview release we're also inviting the community to [contribute their own baselines](https://flower.ai/docs/baselines/how-to-contribute-baselines.html).

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

  The new `FedAvgM` strategy implements Federated Averaging with Server Momentum \[Hsu et al., 2019\].

- **New advanced PyTorch code example** ([#1007](https://github.com/adap/flower/pull/1007))

  A new code example (`advanced_pytorch`) demonstrates advanced Flower concepts with PyTorch.

- **New JAX code example** ([#906](https://github.com/adap/flower/pull/906), [#1143](https://github.com/adap/flower/pull/1143))

  A new code example (`jax_from_centralized_to_federated`) shows federated learning with JAX and Flower.

- **Minor updates**

  - New option to keep Ray running if Ray was already initialized in `start_simulation` ([#1177](https://github.com/adap/flower/pull/1177))
  - Add support for custom `ClientManager` as a `start_simulation` parameter ([#1171](https://github.com/adap/flower/pull/1171))
  - New documentation for [implementing strategies](https://flower.ai/docs/framework/how-to-implement-strategies.html) ([#1097](https://github.com/adap/flower/pull/1097), [#1175](https://github.com/adap/flower/pull/1175))
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

  Simulations (using the Virtual Client Engine through `start_simulation`) now work more smoothly on Jupyter Notebooks (incl. Google Colab) after installing Flower with the `simulation` extra (`pip install 'flwr[simulation]'`).

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

### Incompatible changes

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

### Incompatible changes

- **Disabled final distributed evaluation** ([#800](https://github.com/adap/flower/pull/800))

  Prior behaviour was to perform a final round of distributed evaluation on all connected clients, which is often not required (e.g., when using server-side evaluation). The prior behaviour can be enabled by passing `force_final_distributed_eval=True` to `start_server`.

- **Renamed q-FedAvg strategy** ([#802](https://github.com/adap/flower/pull/802))

  The strategy named `QffedAvg` was renamed to `QFedAvg` to better reflect the notation given in the original paper (q-FFL is the optimization objective, q-FedAvg is the proposed solver). Note the original (now deprecated) `QffedAvg` class is still available for compatibility reasons (it will be removed in a future release).

- **Deprecated and renamed code example** `simulation_pytorch` **to** `simulation_pytorch_legacy` ([#791](https://github.com/adap/flower/pull/791))

  This example has been replaced by a new example. The new example is based on the experimental virtual client engine, which will become the new default way of doing most types of large-scale simulations in Flower. The existing example was kept for reference purposes, but it might be removed in the future.

## v0.16.0 (2021-05-11)

### What's new?

- **New built-in strategies** ([#549](https://github.com/adap/flower/pull/549))

  - (abstract) FedOpt
  - FedAdagrad

- **Custom metrics for server and strategies** ([#717](https://github.com/adap/flower/pull/717))

  The Flower server is now fully task-agnostic, all remaining instances of task-specific metrics (such as `accuracy`) have been replaced by custom metrics dictionaries. Flower 0.15 introduced the capability to pass a dictionary containing custom metrics from client to server. As of this release, custom metrics replace task-specific metrics on the server.

  Custom metric dictionaries are now used in two user-facing APIs: they are returned from Strategy methods `aggregate_fit`/`aggregate_evaluate` and they enable evaluation functions passed to built-in strategies (via `eval_fn`) to return more than two evaluation metrics. Strategies can even return *aggregated* metrics dictionaries for the server to keep track of.

  Strategy implementations should migrate their `aggregate_fit` and `aggregate_evaluate` methods to the new return type (e.g., by simply returning an empty `{}`), server-side evaluation functions should migrate from `return loss, accuracy` to `return loss, {"accuracy": accuracy}`.

  Flower 0.15-style return types are deprecated (but still supported), compatibility will be removed in a future release.

- **Migration warnings for deprecated functionality** ([#690](https://github.com/adap/flower/pull/690))

  Earlier versions of Flower were often migrated to new APIs, while maintaining compatibility with legacy APIs. This release introduces detailed warning messages if usage of deprecated APIs is detected. The new warning messages often provide details on how to migrate to more recent APIs, thus easing the transition from one release to another.

- Improved docs and docstrings ([#691](https://github.com/adap/flower/pull/691) [#692](https://github.com/adap/flower/pull/692) [#713](https://github.com/adap/flower/pull/713))

- MXNet example and documentation

- FedBN implementation in example PyTorch: From Centralized To Federated ([#696](https://github.com/adap/flower/pull/696) [#702](https://github.com/adap/flower/pull/702) [#705](https://github.com/adap/flower/pull/705))

### Incompatible changes

- **Serialization-agnostic server** ([#721](https://github.com/adap/flower/pull/721))

  The Flower server is now fully serialization-agnostic. Prior usage of class `Weights` (which represents parameters as deserialized NumPy ndarrays) was replaced by class `Parameters` (e.g., in `Strategy`). `Parameters` objects are fully serialization-agnostic and represents parameters as byte arrays, the `tensor_type` attributes indicates how these byte arrays should be interpreted (e.g., for serialization/deserialization).

  Built-in strategies implement this approach by handling serialization and deserialization to/from `Weights` internally. Custom/3rd-party Strategy implementations should update to the slightly changed Strategy method definitions. Strategy authors can consult PR [#721](https://github.com/adap/flower/pull/721) to see how strategies can easily migrate to the new format.

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

  # Create strategy and initialize parameters on the server-side
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
