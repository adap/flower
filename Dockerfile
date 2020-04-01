FROM python:3.7.7-slim-stretch AS base

RUN apt-get update && apt-get install -y git

# Prepare general dependencies
RUN python -m pip install -U pip==20.0.2 setuptools==45.2.0 poetry==1.0.5
RUN python -m poetry config virtualenvs.create false

# Set a work directory for flower and copy application dependency list
# and create empty flower application src with __init__.py
WORKDIR /opt/flower
COPY pyproject.toml pyproject.toml
RUN mkdir -p src/flower && touch src/flower/__init__.py
RUN touch README.md

# Install flower dependencies with poetry
RUN python -m poetry install --extras "examples-tensorflow"
RUN python -m pip install -U tensorflow-cpu==2.1.0

FROM python:3.7.7-slim-stretch

WORKDIR /opt/flower

COPY --from=base /usr/local/lib/python3.7/site-packages /usr/local/lib/python3.7/site-packages

COPY src src
COPY dev dev

ENTRYPOINT [ "python", "-m" ]
