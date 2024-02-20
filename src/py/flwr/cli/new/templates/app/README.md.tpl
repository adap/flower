# $project_name

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
flower-client-app client:app --insecure
```

In yet another new terminal window, start the second long-running Flower client:

```bash
flower-client-app client:app --insecure
```

## Start the ServerApp

```bash
flower-server-app server:app --insecure
```
