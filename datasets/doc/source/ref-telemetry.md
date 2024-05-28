# Telemetry

The Flower Datasets open-source project collects **anonymous** usage metrics to make well-informed decisions to improve Flower Datasets. Doing this enables the Flower team to understand how Flower Datasets is used and what challenges users might face.

**Flower is a friendly framework for collaborative AI and data science.** Staying true to this statement, Flower makes it easy to disable telemetry for users that do not want to share anonymous usage metrics.

## Principles

We follow strong principles guarding anonymous usage metrics collection:

- **Optional:** You will always be able to disable telemetry; read on to learn “[How to opt-out](#how-to-opt-out)”.
- **Anonymous:** The reported usage metrics are anonymous and do not contain any personally identifiable information (PII). See “[Collected metrics](#collected-metrics)” to understand what metrics are being reported.
- **Transparent:** You can easily inspect what anonymous metrics are being reported; see the section “[How to inspect what is being reported](#how-to-inspect-what-is-being-reported)”
- **Open for feedback:** You can always reach out to us if you have feedback; see the section “[How to contact us](#how-to-contact-us)” for details.

## How to opt-out

When Flower Datasets starts, it will check for an environment variable called `FLWR_TELEMETRY_ENABLED`. Telemetry can easily be disabled by setting `FLWR_TELEMETRY_ENABLED=0`. Assuming you are using Flower Datasets in a Flower server or client, simply do so by prepending your command as in:

```bash
FLWR_TELEMETRY_ENABLED=0 python server.py # or client.py
```

Alternatively, you can export `FLWR_TELEMETRY_ENABLED=0` in, for example, `.bashrc` (or whatever configuration file applies to your environment) to disable Flower Datasets telemetry permanently.

## Collected metrics

Flower telemetry collects the following metrics:

**Flower version.** Understand which versions of Flower Datasets are currently being used. This helps us to decide whether we should invest effort into releasing a patch version for an older version of Flower Datasets or instead use the bandwidth to build new features.

**Operating system.** Enables us to answer questions such as: *Should we create more guides for Linux, macOS, or Windows?*

**Python version.** Knowing the Python version helps us, for example, to decide whether we should invest effort into supporting old versions of Python or stop supporting them and start taking advantage of new Python features.

**Hardware properties.** Understanding the hardware environment that Flower Datasets is being used in helps to decide whether we should, for example, put more effort into supporting low-resource environments.

**Execution mode.** Knowing what execution mode Flower starts in enables us to understand how heavily certain features are being used and better prioritize based on that.

**Cluster.** Flower telemetry assigns a random in-memory cluster ID each time a Flower workload starts. This allows us to understand which device types not only start Flower workloads but also successfully complete them.

**Source.** Flower telemetry tries to store a random source ID in `~/.flwr/source` the first time a telemetry event is generated. The source ID is important to identify whether an issue is recurring or whether an issue is triggered by multiple clusters running concurrently (which often happens in simulation). For example, if a device runs multiple workloads at the same time, and this results in an issue, then, in order to reproduce the issue, multiple workloads must be started at the same time.

You may delete the source ID at any time. If you wish for all events logged under a specific source ID to be deleted, you can send a deletion request mentioning the source ID to `telemetry@flower.ai`. All events related to that source ID will then be permanently deleted.

We will not collect any personally identifiable information. If you think any of the metrics collected could be misused in any way, please [get in touch with us](#how-to-contact-us). We will update this page to reflect any changes to the metrics collected and publish changes in the changelog.

If you think other metrics would be helpful for us to better guide our decisions, please let us know! We will carefully review them; if we are confident that they do not compromise user privacy, we may add them.

## How to inspect what is being reported

We wanted to make it very easy for you to inspect what anonymous usage metrics are reported. You can view all the reported telemetry information by setting the environment variable `FLWR_TELEMETRY_LOGGING=1`. Logging is disabled by default. You may use logging independently from `FLWR_TELEMETRY_ENABLED` so that you can inspect the telemetry feature without sending any metrics.

```bash
FLWR_TELEMETRY_LOGGING=1 python server.py # or client.py
```

The inspect Flower telemetry without sending any anonymous usage metrics, use both environment variables:

```bash
FLWR_TELEMETRY_ENABLED=0 FLWR_TELEMETRY_LOGGING=1 python server.py # or client.py
```

## How to contact us

We want to hear from you. If you have any feedback or ideas on how to improve the way we handle anonymous usage metrics, reach out to us via [Slack](https://flower.ai/join-slack/) (channel `#telemetry`) or email (`telemetry@flower.ai`).
