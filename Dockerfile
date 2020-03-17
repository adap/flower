FROM python:3.7.7-stretch

RUN apt-get update && apt-get install -y vim git

WORKDIR /opt/flower

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . /opt/flower

RUN NO_VIRTUALENV=1 ./dev/bootstrap.sh
RUN pip install tensorflow==2.1.0

RUN ./src/flower_examples/tf_mnist/run-download.sh
