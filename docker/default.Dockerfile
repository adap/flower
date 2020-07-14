FROM python:3.7.7-slim-stretch

# Install the biggest dependencies before copying the wheel
RUN pip install tensorflow-cpu==2.2.0 numpy==1.18.3

COPY dist/flwr-0.0.1-py3-none-any.whl flwr-0.0.1-py3-none-any.whl
RUN python -m pip install --no-cache-dir 'flwr-0.0.1-py3-none-any.whl[examples-tensorflow,http-logger,benchmark,ops]'
RUN rm flwr-0.0.1-py3-none-any.whl
