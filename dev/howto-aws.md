1. Start ubuntu machine on AWS

2. RUN: `sudo apt update && sudo apt -y upgrade`

3. RUN: 
    ```
    sudo apt-get install -y make build-essential libssl-dev zlib1g-dev libbz2-dev \
    libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev \
    xz-utils tk-dev libffi-dev liblzma-dev python-openssl git
    ```

4. Install pyenv with:
    ```
    curl https://pyenv.run | bash
    ```
and follow the instructions

5. Install pyenv-virtualenv
    ```
    git clone https://github.com/pyenv/pyenv-virtualenv.git $(pyenv root)/plugins/pyenv-virtualenv
    ```

6. Install python 3.7.6 with:
    ```
    pyenv install 3.7.6
    ```

7. Clone the repo
    ```
    git clone https://github.com/adap/flower.git
    ```

8. Create venv and run boostrap
    ```
    cd flower
    ./dev/venv-create.sh
    ./dev/bootstrap.sh
    ```

9. Adjust example and run
