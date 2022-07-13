from flwr.client.sa_client_wrapper import SAClient
from flwr.common.light_sec_agg.client_logic import setup_config, ask_encrypted_encoded_masks, ask_masked_models, \
    ask_aggregated_encoded_masks


class LightSecAggWrapper(SAClient):
    """Wrapper which adds LightSecAgg methods."""
    def __init__(self, c: Client):
        super().__init__(c)
        self.tm = Timer()

    def sa_respond(self, ins: SAServerMessageCarrier) -> SAClientMessageCarrier:
        self.tm.tic('s' + ins.identifier)
        if ins.identifier == '0':
            new_ins = LightSecAggSetupConfigIns(ins.str2scalar)
            res = setup_config(self, new_ins)
            ret_msg = SAClientMessageCarrier(identifier='0', bytes_list=[res.pk])
        elif ins.identifier == '1':
            public_keys_dict = dict([(int(k), LightSecAggSetupConfigRes(v)) for k, v in ins.str2scalar.items()])
            new_ins = AskEncryptedEncodedMasksIns(public_keys_dict)
            res = lsa_proto.ask_encrypted_encoded_masks(self, new_ins)
            packet_dict = dict([(str(p.destination), p.ciphertext) for p in res.packet_list])
            ret_msg = SAClientMessageCarrier(identifier='1', str2scalar=packet_dict)
        elif ins.identifier == '2':
            packets = [EncryptedEncodedMasksPacket(int(k), self.id, v) for k, v in ins.str2scalar.items()]
            new_ins = AskMaskedModelsIns(packets, ins.fit_ins)
            res = lsa_proto.ask_masked_models(self, new_ins)
            ret_msg = SAClientMessageCarrier(identifier='2', parameters=res.parameters)
        elif ins.identifier == '3':
            new_ins = AskAggregatedEncodedMasksIns(ins.numpy_ndarray_list[0].tolist())
            res = lsa_proto.ask_aggregated_encoded_masks(self, new_ins)
            ret_msg = SAClientMessageCarrier(identifier='3', parameters=res.aggregated_encoded_mask)
        self.tm.toc('s' + ins.identifier)
        if self.id == 6:
            f = open("log.txt", "a")
            f.write(f"Client without communication stage {ins.identifier}:{self.tm.get('s' + ins.identifier)} \n")
            f.close()

        return ret_msg
