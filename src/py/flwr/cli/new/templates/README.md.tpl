# $project_name

TODO: intro text

If you are new to Flower and you are looking for more advanced examples please consult the regular PyTorch code examples ([quickstart](https://github.com/adap/flower/tree/main/examples/quickstart-pytorch), [advanced](https://github.com/adap/flower/tree/main/examples/advanced-pytorch)) to learn how to use Flower with PyTorch.

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
