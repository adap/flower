FROM ubuntu:22.04
LABEL author="The Flower Authors"

RUN apt-get update && \
    # install Python
    apt-get install -y python3 python3-distutils curl && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
    curl -sSL -O https://bootstrap.pypa.io/get-pip.py && \
    python3 get-pip.py && \
    rm get-pip.py && \
    # clean apt cache
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /requirements.txt
RUN pip3 install --no-cache-dir -r /requirements.txt

COPY mnist.py /workspace/mnist.py
ENTRYPOINT ["python3", "/workspace/mnist.py"]
