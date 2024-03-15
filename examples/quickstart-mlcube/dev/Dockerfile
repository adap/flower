FROM ubuntu:22.04
LABEL author="The Flower Authors"

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
    python3=3.10.6-1~22.04 \
    python3-pip=22.0.2+dfsg-1ubuntu0.4 \
    python3-distutils=3.10.8-1~22.04 \
    curl=7.81.0-1ubuntu1.15 \
    build-essential=12.9ubuntu3 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /requirements.txt

# fixes pyyaml installation issues
RUN pip3 install --no-cache-dir --no-build-isolation \
    pyyaml==5.4.1 \
    && pip3 install --no-cache-dir -r /requirements.txt

COPY mnist.py /workspace/mnist.py
ENTRYPOINT ["python3", "/workspace/mnist.py"]
