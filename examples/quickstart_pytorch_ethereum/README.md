# BlockChain based Flower Example using PyTorch

This introductory example for blockchain-powered Flower uses Ethereum, but in-depth knowledge of Ethereum is not necessarily required to run the example. However, it will help you understand how to adapt Ethereum to Flower's project to suit your use case.
Running this example on its own is pretty easy.

## Project Setup

Start by cloning the example project. We prepared a single-line command that you can copy into your shell which will checkout the example for you:

```shell
git clone --depth=1 https://github.com/adap/flower.git && mv flower/examples/quickstart_pytorch_ethereum . && rm -rf flower && cd quickstart_pytorch_ethereum
```

This will create a new directory called `quickstart_pytorch_ethereum` containing the following files:

```shell
-- pyproject.toml
-- client.py
-- server.py
-- README.md
```

Project dependencies (such as `torch`,`web3` and `flwr`) are defined in `pyproject.toml`. We recommend [Poetry](https://python-poetry.org/docs/) to install those dependencies and manage your virtual environment ([Poetry installation](https://python-poetry.org/docs/#installation)), but feel free to use a different way of installing dependencies and managing virtual environments if you have other preferences.

```shell
poetry install
poetry shell
```

Poetry will install all your dependencies in a newly created virtual environment. To verify that everything works correctly you can run the following command:

```shell
python3 -c "import flwr"
```

`ganache` installation is required to build Ethereum local environment.
`truffle` installation is required to build Solidity.

```shell
npm i -g ganache-cli
npm i -g truffle
```

And then, `ipfs daemon` installation is required to save deep learning model history. before installation must install nvm.
```shell
wget https://dist.ipfs.tech/kubo/v0.15.0/kubo_v0.12.2_linux-amd64.tar.gz
tar -xvzf kubo_v0.12.2_linux-amd64.tar.gz
cd kubo
sudo bash install.sh
```



If you don't see any errors you're good to go!

# Run Federated Learning with PyTorch and Flower

First, run ipfs daemon to store the Deep Learning Network
```shell
ipfs daemon
```
Second, Run Ganache-cli
```shell
ganache-cli --port 7545 --networkId 5777 -a 31 -d
```
Last, Deploy smart contract
```
cd ~/flowr/py/flwr/client/eth_client
npm i # first time only
truffle migrate --network development --reset
```
After deployment, you can get the result like below.
```shell
2_deploy_contracts.js
=====================

   Replacing 'Crowdsource'
   -----------------------
   > transaction hash:    0x5e05aae436f07591464a11a0ba301220575038e8ff77cb6705401d521940aa5c
   > Blocks: 0            Seconds: 0
   > contract address:    0xCfEB869F69431e42cdB54A4F4f105C19C080A601
   > block number:        3
   > block timestamp:     1682418499
   > account:             0x90F8bf6A479f320ead074411a4B0e7944Ea8c9C1
   > balance:             99.95304242
   > gas used:            2103520 (0x2018e0)
   > gas price:           20 gwei
   > value sent:          0 ETH
   > total cost:          0.0420704 ETH
```
If you modify `CONTRACT_ADDRESS` of `~/flowr/src/py/flwr/client/eth_client/eth_client.py` to the contract address, everything is ready.

Afterwards you are ready to start the Flower server as well as the clients. You can simply start the server in a terminal as follows:

```shell
python3 server.py
```

Now you are ready to start the Flower clients which will participate in the learning. To do so simply open two more terminal windows and run the following commands.

Start client 1 in the first terminal with cid 0:

```shell
python3 client.py
```

Start client 2 in the second terminal with cid 1:

```shell
python3 client.py
```

You will see that PyTorch is starting a federated training. Have a look to the [Flower Quickstarter documentation](https://flower.dev/docs/quickstart-pytorch.html) for a detailed explanation.
