# $project_name: A Flower / $framework_str app

## Install dependencies and project

```bash
pip install -e .
```

## Run with the Simulation Engine

In the `$project_name` directory, use `flwr run` to run a local simulation:

```bash
flwr run .
```

Refer to the [How to Run Simulations](https://flower.ai/docs/framework/how-to-run-simulations.html) guide in the documentation for advice on how to optimize your simulations.

## Run with the Deployment Engine

Follow this [how-to guide](https://flower.ai/docs/framework/how-to-run-flower-with-deployment-engine.html) to run the same app in this example but with Flower's Deployment Engine. After that, you might be interested in setting up [secure TLS-enabled communications](https://flower.ai/docs/framework/how-to-enable-tls-connections.html) and [SuperNode authentication](https://flower.ai/docs/framework/how-to-authenticate-supernodes.html) in your federation.

You can run Flower on Docker too! Check out the [Flower with Docker](https://flower.ai/docs/framework/docker/index.html) documentation.

## TOML Configuration

When using `flwr new`, a `pyproject.toml` file is generated in the directory. This file defines your app’s dependencies, configuration, and federation setup.

Here are a few key sections to look out for:

### Project Dependencies

```toml
[project]
dependencies = [
    "flwr[simulation]>=1.20.0",
    "numpy>=2.0.2",
]
```

Add any Python packages your app needs here. These will be installed when you run `pip install -e . `.

### App Components

```toml
[tool.flwr.app.components]
serverapp = "$project_name.server_app:app"
clientapp = "$project_name.client_app:app"
```

These entries point to your `ServerApp` and `ClientApp` definitions, using the format `<module>:<object>`. Only update these import paths if you rename your modules or the variables that reference your `ServerApp` or `ClientApp`.

### App Configuration

```toml
[tool.flwr.app.config]
num-server-rounds = 3
any-name-you-like = "any value supported by TOML"
```

Define any configuration values you want to make available to your app at runtime. You can access them in your code using `context.run_config`, for example:

```python
server_rounds = context.run_config["num-server-rounds"]
```

### Federation Configuration

```toml
[tool.flwr.federations]
default = "local-simulation"

[tool.flwr.federations.local-simulation]
options.num-supernodes = 10
```

By default, a federation named `"local-simulation"` is provided and set as the default via the `default = "local-simulation"` line in your TOML file. You can rename federations to anything you like. The example above sets up a local simulation federation with 10 virtual SuperNodes via the line `options.num-supernodes = 10`. Learn more about customizing your simulation in the [How to Run Simulations](https://flower.ai/docs/framework/how-to-run-simulations.html).

You can define multiple federations here, including configurations for remote deployments:

```toml
[tool.flwr.federations.remote-deployment]
address = "<SUPERLINK-ADDRESS>:<PORT>"
insecure = true
```

To enable TLS, refer to the [deployment documentation](https://flower.ai/docs/framework/deploy.html).

## Resources

- Flower website: [flower.ai](https://flower.ai/)
- Check the documentation: [flower.ai/docs](https://flower.ai/docs/)
- Give Flower a ⭐️ on GitHub: [GitHub](https://github.com/adap/flower)
- Join the Flower community!
  - [Flower Slack](https://flower.ai/join-slack/)
  - [Flower Discuss](https://discuss.flower.ai/)
