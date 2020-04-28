FROM python:3.7.7-slim-stretch

COPY dist/flower-0.0.1-py3-none-any.whl flower-0.0.1-py3-none-any.whl
RUN python -m pip install --no-cache-dir 'flower-0.0.1-py3-none-any.whl[examples-tensorflow]'

RUN apt-get update
RUN apt-get install -y openssh-server
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

EXPOSE 22

CMD    ["/usr/sbin/sshd", "-D"]