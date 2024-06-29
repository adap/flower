FROM ubuntu:22.04

# https://code.visualstudio.com/docs/remote/containers-advanced#_creating-a-nonroot-user
ARG USERNAME=flwr-vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
  && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
  #
  # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
  && apt-get update \
  && apt-get install -y sudo bash \
  && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
  && chmod 0440 /etc/sudoers.d/$USERNAME

# A persistent volume will be configured in devcontainer.json so we don't loose the commandhistory
# after rebuilding the container
RUN SNIPPET="export PROMPT_COMMAND='history -a' && export HISTFILE=/commandhistory/.bash_history" \
  && mkdir /commandhistory \
  && touch /commandhistory/.bash_history \
  && chown -R $USERNAME /commandhistory \
  && echo $SNIPPET >> "/home/$USERNAME/.bashrc"

# Install system dependencies
RUN apt-get update
RUN apt-get install -y curl wget gnupg python3 python-is-python3 python3-pip git \
  build-essential tmux vim

RUN python -m pip install \
  pip==24.0.0 \
  setuptools==69.5.1 \
  poetry==1.8.3

USER $USERNAME
ENV PATH="/home/$USERNAME/.local/bin:${PATH}"
