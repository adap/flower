FROM python:3.7.7-stretch

RUN apt-get update && apt-get install -y vim git

COPY . /opt/flower

WORKDIR /opt/flower

RUN NO_VIRTUALENV=1 ./dev/bootstrap.sh

RUN pip install tensorflow==2.1
