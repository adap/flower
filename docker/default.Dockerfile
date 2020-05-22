FROM python:3.7.7-slim-stretch

# Install the biggest dependencies before copying the wheel
RUN pip install tensorflow-cpu==2.1.0 numpy==1.18.3

COPY dist/flower-0.0.1-py3-none-any.whl flower-0.0.1-py3-none-any.whl
RUN python -m pip install --no-cache-dir 'flower-0.0.1-py3-none-any.whl[examples-tensorflow,http-logger,benchmark,ops]'
RUN rm flower-0.0.1-py3-none-any.whl
