# Launch Flower REST Server
uvicorn flwr.server.rest_server.rest_api:app --reload --host 0.0.0.0 --port 8080