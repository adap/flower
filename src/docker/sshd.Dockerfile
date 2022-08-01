FROM python:3.7.12-slim-stretch

RUN apt-get update
RUN apt-get install -y openssh-server screen
RUN mkdir /var/run/sshd

RUN echo 'root:root' | chpasswd

RUN sed -ri 's/^#?PermitRootLogin\s+.*/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN sed -ri 's/UsePAM yes/#UsePAM yes/g' /etc/ssh/sshd_config

RUN mkdir /root/.ssh

ARG SSH_PUBLIC_KEY
RUN echo $SSH_PUBLIC_KEY > /root/.ssh/authorized_keys

RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

WORKDIR /root

RUN python3.7 -m pip install tensorflow-cpu==2.6.0 torch==1.7.1 torchvision==0.8.2 numpy==1.19.5
COPY dist/flwr-1.1.0-py3-none-any.whl flwr-1.1.0-py3-none-any.whl
RUN python3.7 -m pip install --no-cache-dir 'flwr-1.1.0-py3-none-any.whl[examples-pytorch,examples-tensorflow,http-logger,baseline,ops]' && \
    rm flwr-1.1.0-py3-none-any.whl

RUN python3.7 -m flwr_experimental.baseline.tf_fashion_mnist.download
RUN python3.7 -m flwr_experimental.baseline.tf_cifar.download

EXPOSE 22

CMD ["/usr/sbin/sshd", "-D"]
