import json
import os
import base58
from web3 import HTTPProvider, Web3
from dotenv import load_dotenv

load_dotenv()

# 1. 이더리움 클라이언트
class _BaseEthClient:
    """
    An ethereum client.
    """

    # PROVIDER_ADDRESS = os.environ.get("RPC_URL")
    # NETWORK_ID = os.environ.get("NETWORK_ID")
    PROVIDER_ADDRESS = "http://127.0.0.1:7545" # ganache
    NETWORK_ID = "5777"

    def __init__(self, account_idx):
        self._w3 = Web3(HTTPProvider(self.PROVIDER_ADDRESS))  # json rpc 서버 연결(가나슈)
        self.address = self._w3.eth.accounts[int(account_idx)]  # 이더리움 지갑주소
        self._w3.eth.defaultAccount = self.address  # 지정된 지갑주소를 기본주소로
        self.txs = []  # 트랜잭션 담을 리스트

    # 트랜잭션 해쉬를 입력해서 트랜잭션 상태를 받아옴
    def wait_for_tx(self, tx_hash):
        receipt = self._w3.eth.waitForTransactionReceipt(tx_hash)
        return receipt

    # 특정 어카운트가 사용한 총 가스비 출력
    def get_gas_used(self):
        receipts = [self._w3.eth.getTransactionReceipt(tx) for tx in self.txs]
        gas_amounts = [receipt['gasUsed'] for receipt in receipts]
        return sum(gas_amounts)

    def get_account(self):
        # print(self.address)
        return self.address

    def get_accounts(self, max_num_updates):
        account_list = []
        for i in range(1, max_num_updates + 1):
            account_list.append(self._w3.eth.accounts[i])
        return account_list


class _BaseContractClient(_BaseEthClient):
    """
    Contains common features of both contract clients.
    Handles contract setup, conversions to and from bytes32 and other utils.
    """

    IPFS_HASH_PREFIX = bytes.fromhex('1220')  # IPFS 해쉬값 접두사

    def __init__(self, contract_json_path, account_idx, contract_address, deploy):
        super().__init__(account_idx)

        self._contract_json_path = contract_json_path

        self._contract, self.contract_address = self._instantiate_contract(contract_address, deploy)

    def _instantiate_contract(self, address=None, deploy=False):
        # 가나슈에 배포된 컨트랙트의 json 파일을 인스턴스화
        with open(self._contract_json_path) as json_file:
            crt_json = json.load(json_file)
            abi = crt_json['abi']
            bytecode = crt_json['bytecode']
            if address is None:
                if deploy:
                    # 배포 트랜잭션에 대한 해쉬 반환
                    tx_hash = self._w3.eth.contract(
                        abi=abi,
                        bytecode=bytecode
                    ).constructor().transact()  # 계약의 인스턴스가 배포됨
                    self.txs.append(tx_hash)
                    tx_receipt = self.wait_for_tx(tx_hash)
                    address = tx_receipt.contractAddress
                else:
                    address = crt_json['networks'][self.NETWORK_ID]['address']
        instance = self._w3.eth.contract(
            abi=abi,
            address=address
        )
        return instance, address

    def _to_bytes32(self, model_cid):
        bytes34 = base58.b58decode(model_cid)
        assert bytes34[:2] == self.IPFS_HASH_PREFIX, \
            f"IPFS cid should begin with {self.IPFS_HASH_PREFIX} but got {bytes34[:2].hex()}"
        bytes32 = bytes34[2:]
        return bytes32

    def _from_bytes32(self, bytes32):
        bytes34 = self.IPFS_HASH_PREFIX + bytes32
        model_cid = base58.b58encode(bytes34).decode()
        return model_cid


class _EthClient(_BaseContractClient):
    """
    Wrapper over the Crowdsource.sol ABI, to gracefully bridge Python data to Solidity.
    The API of this class should match that of the smart contract.
    """

    def __init__(self, account_idx, address, deploy):
        super().__init__(
            os.path.dirname(os.path.abspath(__file__))+"/build/contracts/Crowdsource.json",
            account_idx,
            address,
            deploy
        )

    # os.system("pwd")
    def evaluator(self):
        return self._contract.functions.evaluator().call()

    def genesis(self):
        cid_bytes = self._contract.functions.genesis().call()
        return self._from_bytes32(cid_bytes)

    def updates(self, training_round):
        cid_bytes = self._contract.functions.updates(training_round).call()
        return [self._from_bytes32(b) for b in cid_bytes]

    def saveGlobalmodel(self, model_cid, training_round):
        cid_bytes = self._to_bytes32(model_cid)
        tx = self._contract.functions.saveGlobalmodel(cid_bytes, training_round).transact()
        self.txs.append(tx)
        return tx

    def getGlobalmodel(self, training_round):
        cid_bytes = self._contract.functions.getGlobalmodel(training_round).call()
        model_cid = self._from_bytes32(cid_bytes)
        return model_cid

    def saveScores(self, model_cid, account_address, score):
        cid_bytes = self._to_bytes32(model_cid)
        tx = self._contract.functions.saveScores(cid_bytes, account_address, score).transact()
        self.txs.append(tx)
        return tx

    def getScores(self, model_cid):
        cid_bytes = self._to_bytes32(model_cid)
        account, score = self._contract.functions.getScores(cid_bytes).call()
        return account, score

    def completeEval(self, training_round):
        tx = self._contract.functions.completeEval(training_round).transact()
        self.txs.append(tx)
        return tx

    def getCurTrainers(self, training_round):
        current_trainers = self._contract.functions.getCurTrainers(training_round).call()
        return current_trainers

    def isTrainer(self, training_round, account_address=None):
        if account_address is None:
            account_address = self.address
        trainCheckFlag = self._contract.functions.isTrainer(account_address, training_round).call()
        return trainCheckFlag

    def setCurTrainer(self, training_round, account_address=None, contract_address=None):
        if account_address is None:
            account_address = self.address
        tx = self._contract.functions.setCurTrainer(account_address, training_round).transact()
        self.txs.append(tx)
        return tx

    def changeMaxNumUpdates(self, max_num):
        tx = self._contract.functions.changeMaxNumUpdates(max_num).transact()
        self.txs.append(tx)
        return tx

    def currentRound(self):
        return self._contract.functions.currentRound().call()

    def secondsRemaining(self):
        return self._contract.functions.secondsRemaining().call()

    def countTokens(self, address=None, training_round=None):
        if address is None:
            address = self.address
        if training_round is None:
            training_round = self.currentRound()
        return self._contract.functions.countTokens(address, training_round).call()

    def countTotalTokens(self, training_round=None):
        if training_round is None:
            training_round = self.currentRound()
        return self._contract.functions.countTotalTokens(training_round).call()

    def madeContribution(self, address, training_round):
        return self._contract.functions.madecontribution(address, training_round).call()

    def setGenesis(self, model_cid):
        cid_bytes = self._to_bytes32(model_cid)
        self._contract.functions.setGenesis(
            cid_bytes).call()
        tx = self._contract.functions.setGenesis(cid_bytes).transact()
        self.txs.append(tx)
        return tx

    def setModelArch(self, model_cid):
        cid_bytes = self._to_bytes32(model_cid)
        self._contract.functions.setModelArchitecture(
            cid_bytes).call()
        tx = self._contract.functions.setModelArchitecture(cid_bytes).transact()
        self.txs.append(tx)
        return tx

    def addModelUpdate(self, model_cid, training_round):
        cid_bytes = self._to_bytes32(model_cid)
        self._contract.functions.addModelUpdate(
            cid_bytes, training_round).call()
        tx = self._contract.functions.addModelUpdate(
            cid_bytes, training_round).transact()
        self.txs.append(tx)
        return tx

    def skipRound(self, training_round):
        tx = self._contract.functions.skipRound(training_round).transact()
        self.txs.append(tx)
        return tx

    def waitTrainers(self, training_round):
        train_flag = self._contract.functions.waitTrainers(training_round).call()
        return train_flag

    def getAccountfromUpdate(self, model_cid):
        cid_bytes = self._to_bytes32(model_cid)
        account = self._contract.functions.getAccountfromUpdate(cid_bytes).call()
        return account

    def setTokens(self, model_cid, num_tokens):
        cid_bytes = self._to_bytes32(model_cid)
        self._contract.functions.setTokens(cid_bytes, num_tokens).call()
        tx = self._contract.functions.setTokens(
            cid_bytes, num_tokens).transact()
        self.txs.append(tx)
        return tx

    def getmaxNum(self):
        return self._contract.functions.getmaxNum().call()

    def getGenesis(self):
        cid_bytes = self._contract.functions.getGenesis().call()
        model_cid = self._from_bytes32(cid_bytes)
        return model_cid

    def getModelArchitecture(self):
        cid_bytes = self._contract.functions.getModelArchitecture().call()
        arch_cid = self._from_bytes32(cid_bytes)
        return arch_cid


