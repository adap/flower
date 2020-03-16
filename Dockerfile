FROM ubuntu:18.04

# Make sure we don't have to do anything manually
# while installing
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get -y upgrade

RUN apt-get install -y make build-essential libssl-dev zlib1g-dev libbz2-dev \
    libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev \
    xz-utils tk-dev libffi-dev liblzma-dev python-openssl git vim

RUN curl https://pyenv.run | bash


ENV HOME /root
ENV PYENV_ROOT $HOME/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH

RUN pyenv install 3.7.6
RUN pyenv global 3.7.6

RUN git clone https://github.com/adap/flower.git /root/flower

WORKDIR /root/flower

RUN ./dev/bootstrap.sh
