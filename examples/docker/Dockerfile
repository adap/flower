FROM flwr/base:latest

WORKDIR /app

# Copy dependencies list and install them.
COPY pyproject.toml ./
RUN poetry install

# Copy the application code
COPY waterlily waterlily
