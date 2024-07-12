**Run in insecure mode**

```
docker compose up --build
```

**Run with persistent state**

```
docker compose -f compose.yml -f with-state.yml up --build
```

**Run with TLS**

```
docker compose -f certs.yml up --build
docker compose -f compose.yml -f with-tls.yml up --build
```

**Render multiple files into a single file**

```
docker compose -f compose.yml -f with-tls.yml config --no-path-resolution > my_compose.yml
```
