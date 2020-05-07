FROM python:3.7.7-slim-stretch

COPY dist/flower-0.0.1-py3-none-any.whl flower-0.0.1-py3-none-any.whl
RUN python -m pip install --no-cache-dir 'flower-0.0.1-py3-none-any.whl[examples-tensorflow]'
RUN rm flower-0.0.1-py3-none-any.whl
