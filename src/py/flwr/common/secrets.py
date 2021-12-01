from dissononce.dh.x25519.x25519 import X25519DH
from dissononce.dh.keypair import KeyPair
from dissononce.dh.public import PublicKey
from dissononce.dh.private import PrivateKey

SERVER_KEYPAIR_HEX = {"public": "06ff39946687263ad213117ae6877b852a9a6e2fa5aae0341c9634471f31d757",
                      "private": "486e5821ef8ae3029cef740ed467f0231ecdd0610a26bcdd730f69bf370cb669"}

CLIENT_KEYPAIR_HEX_LIST = [{"public": "c8f43a6ab345d78fee662a69bf726311105f538910d021dca1fe53cd5215f047", "private": "208c717bdc07119992887b5899e81b2431004151f20d179fdd9a430fa129a76d"}, {"public": "9d4526d5a1dcfd400045045ce5bc69c2bd9041f99e67d191ec79c468b8ab0666", "private": "506e6cf84882fb2cd91e286eec0454d93a3994cebbef41d4f395a107ff981363"}, {"public": "aebcc0f39ad8844fcbd8f71e6fb16950a7404022fe792a9bbf5f2d5ce92fa761", "private": "a0cadaff8a3e71d31476ae4baa41ef31f9054ab3a3400e3d3a0674321db9596e"}, {"public": "1a0fe1007956734e67846e1795674aafa1499cbc3c3d11f4a111f5b928766220", "private": "1046b83b2ebc92434a2268a7a42d53504cf3101d6357ed73116510cb2835f27a"}, {"public": "4a8727bbe1b9d1cb36aacde88708ce016091d08e47a500cd42f0936d3429d656", "private": "d8c503122c8610dcb30fccec6e4122066de400bf5049465dc75dbc3c3aa50243"}, {"public": "442219665f535e8f86573641098eedaa9523fd1e91e938a018046287412cb16a", "private": "00a781709fb5f4474a3e9ae95d2c63b81c787d3edf0ae0612adf259a31933d4b"}, {"public": "5495f0b0eeafd0b9809aa4643174fc579ddda50b365d4697a12217dc41d0c57e", "private": "c808a8e28d904e80deaa75d5ae3675f7ed8f4ea4c53512a327d05420543ea241"}, {"public": "af49afdc362305a000fa5b458c435f183902b15be6995ad5df582aa9851ea44e", "private": "e08aa9a2a42d6e7498bdad8d7f3b502acf54a84cf1025f3b91bb8c32331b1f43"}, {
    "public": "0beccce1956bd000802a1ca2a765615a0577c81202d5b040befd270eb44e163d", "private": "507ead05478c54d40752a98cd45bff63c95ee2d0f5506228ce0d14c9be575c48"}, {"public": "32f097f0ed6f43a90bd92209f322e2b50ed496df0e95b45eb7743a6190a30614", "private": "303ea7393346769b0f86268224e3d6fc320c34b5bedc8393500395b14362f26d"}, {"public": "0afae0bcab80d7be2bedc533c494a6c3f6f2aa32a6895eff2333bdaabc7f214e", "private": "c0bae817620268ddc569e90c444559c6b4b232a883f21f96ec36f45a3f8c444b"}, {"public": "6220fc26c1a9ea216d20bf0a629b8a7b1739c3d9fe46897cddc047ca19bd201b", "private": "80ecea5817095f2cbe91c95da61bc3506e6a41f0dc841f9a271c34e5ea9d8d7d"}, {"public": "cb02f29d41e8f450922d9695ddab6af18161e782230c29f0d18c3ec4385d7f1c", "private": "d0684720a2d6f30b6dda7f7b8c235ac2a81fa9857b4265a181bcba8ac490c16f"}, {"public": "48f1900badb3bb3880b39676349921f116ca686c206ba623066ca97068707524", "private": "48b733741d034afbc69f23f710cdbf66da6437f5bf9f931faac24ee4e0d82e4a"}, {"public": "9b5ed9ca99da2a063afc822c5b0406a4d731bc0232d2c10f7268f4a06718481f", "private": "5805183e083e2454877fd4b5d6e80a12fbe20bb6f1731169cb6fb262b345bb45"}, {"public": "a99b6555bfdf2c1a0536f9c174c49ac1c284a7953f5a17efc746f139fd026e6a", "private": "4819a8c499b92640d5ac8a59cd7d2ca89d90eff259f4c09f7b3033b03542035c"}]

CLIENT_PUBLIC_KEY_LIST = [kp['public'] for kp in CLIENT_KEYPAIR_HEX_LIST]

def convert_key_pair(obj):
    return KeyPair(
        PublicKey(32, bytes.fromhex(obj["public"])),
        PrivateKey(bytes.fromhex(obj["private"]))
    )


class SecretsManager:
    @staticmethod
    def server_key_pair():
        return convert_key_pair(SERVER_KEYPAIR_HEX)

    @staticmethod
    def server_public_key():
        return convert_key_pair(SERVER_KEYPAIR_HEX).public

    @staticmethod
    def client_key_pair(idx):
        assert idx in range(len(CLIENT_KEYPAIR_HEX_LIST))
        return convert_key_pair(CLIENT_KEYPAIR_HEX_LIST[idx])

    @staticmethod
    def is_valid_client(pub_key_obj): 
        return pub_key_obj.data.hex() in CLIENT_PUBLIC_KEY_LIST
