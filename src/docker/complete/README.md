**Run in insecure mode**

```
docker compose up --build -d
```

**Run with persistent state**

```
docker compose -f compose.yml -f with-state.yml up --build -d
```

**Run with TLS**

```
docker compose -f certs.yml up --build
docker compose -f compose.yml -f with-tls.yml up --build -d
```

```toml
[tool.flwr.federations.superexec]
address = "127.0.0.1:9093"
insecure = true

[tool.flwr.federations.superexec-sec]
address = "127.0.0.1:9093"
root-certificates = "../superexec-certificates/ca.crt"

```

**Render multiple files into a single file**

```
docker compose -f compose.yml -f with-tls.yml config --no-path-resolution > my_compose.yml
```
