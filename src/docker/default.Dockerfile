FROM python:3.7.12-slim-stretch

# Install the biggest dependencies before copying the wheel
RUN pip install tensorflow-cpu==2.6.2 numpy==1.19.5

COPY dist/flwr-1.1.0-py3-none-any.whl flwr-1.1.0-py3-none-any.whl
RUN python3.7 -m pip install --no-cache-dir 'flwr-1.1.0-py3-none-any.whl[examples-pytorch,examples-tensorflow,http-logger,baseline,ops]'
RUN rm flwr-1.1.0-py3-none-any.whl
