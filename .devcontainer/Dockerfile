FROM ubuntu:22.04

# https://code.visualstudio.com/docs/remote/containers-advanced#_creating-a-nonroot-user
ARG USERNAME=flwr-vscode
ARG USER_UID=1000
ARG USER_GID=${USER_UID}
ENV PATH="/home/$USERNAME/.local/bin:${PATH}"

# Create the user
RUN groupadd --gid ${USER_GID} ${USERNAME} \
    && useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME} \
    #
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && apt-get install --no-install-recommends -y \
    sudo=1.9.9-1ubuntu2.4 \
    bash=5.1-6ubuntu1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && echo ${USERNAME} ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME}

# A persistent volume will be configured in devcontainer.json so we don't loose the commandhistory
# after rebuilding the container
RUN SNIPPET="export PROMPT_COMMAND='history -a' && export HISTFILE='/commandhistory/.bash_history'" \
    && mkdir /commandhistory \
    && touch /commandhistory/.bash_history \
    && chown -R ${USERNAME} /commandhistory \
    && echo ${SNIPPET} >> "/home/${USERNAME}/.bashrc"

# Install system dependencies
RUN apt-get update \  
    && apt-get install --no-install-recommends -y  \
    python3=3.10.6-1~22.04  \
    python-is-python3=3.9.2-2  \
    python3-pip=22.0.2+dfsg-1ubuntu0.4 \
    git=1:2.34.1-1ubuntu1.10 \
    curl=7.81.0-1ubuntu1.15 \
    wget=1.21.2-2ubuntu1 \
    gnupg=2.2.27-3ubuntu2.1 \
    build-essential=12.9ubuntu3 \
    tmux=3.2a-4ubuntu0.2 \
    vim=2:8.2.3995-1ubuntu2.15 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir \
    pip==23.3.1 \
    setuptools==68.2.2 \
    poetry==1.7.1

USER ${USERNAME}
