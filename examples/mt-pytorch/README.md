# Flower App (PyTorch) 🧪

🧪 = This example covers experimental features that might change in future versions of Flower

Please consult the regular PyTorch code examples ([quickstart](https://github.com/adap/flower/tree/main/examples/quickstart-pytorch), [advanced](https://github.com/adap/flower/tree/main/examples/advanced-pytorch)) to learn how to use Flower with PyTorch.

This how-to guide describes the deployment of a long-running Flower server.

## Preconditions

Let's assume the following project structure:

```bash
$ tree .
.
├── client.py
├── server.py
├── task.py
└── requirements.txt
```

## Install dependencies

```bash
pip install -r requirements.txt
```

## Start the SuperLink

```bash
flower-superlink --insecure
```

## Start the long-running Flower client

In a new terminal window, start the first long-running Flower client:

```bash
flower-client client:app --insecure
```

In yet another new terminal window, start the second long-running Flower client:

```bash
flower-client client:app --insecure
```

## Start the driver

```bash
python driver.py
```
