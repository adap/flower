from flwr.client.abc_sa_client_wrapper import SAClientWrapper
from flwr.common.typing import SAServerMessageCarrier, SAClientMessageCarrier
from flwr.common.timer import Timer
import flwr.common.light_sec_agg.client_logic as cl
from flwr.client.client import Client


class LightSecAggWrapper(SAClientWrapper):
    """Wrapper which adds LightSecAgg methods."""
    def __init__(self, c: Client):
        super().__init__(c)
        self.tm = Timer()

    def sa_respond(self, ins: SAServerMessageCarrier) -> SAClientMessageCarrier:
        self.tm.tic('s' + ins.identifier)
        if ins.identifier == '0':
            res = cl.setup_config(self, ins.str2scalar)
            ret_msg = SAClientMessageCarrier(identifier='0', bytes_list=[res])
        elif ins.identifier == '1':
            public_keys_dict = dict([(int(k), v) for k, v in ins.str2scalar.items()])
            res = cl.ask_encrypted_encoded_masks(self, public_keys_dict)
            packet_dict = dict([(str(p[1]), p[2]) for p in res])
            ret_msg = SAClientMessageCarrier(identifier='1', str2scalar=packet_dict)
        elif ins.identifier == '2':
            sec_id = self.get_sec_id()
            packets = [(int(k), sec_id, v) for k, v in ins.str2scalar.items()]
            res = cl.ask_masked_models(self, packets, ins.fit_ins)
            ret_msg = SAClientMessageCarrier(identifier='2', parameters=res)
        elif ins.identifier == '3':
            active_clients = ins.numpy_ndarray_list[0].tolist()
            res = cl.ask_aggregated_encoded_masks(self, active_clients)
            ret_msg = SAClientMessageCarrier(identifier='3', parameters=res)
        else:
            raise Exception("Invalid identifier")
        self.tm.toc('s' + ins.identifier)
        if self.get_sec_id() == 6:
            f = open("log.txt", "a")
            f.write(f"Client without communication stage {ins.identifier}:{self.tm.get('s' + ins.identifier)} \n")
            f.close()

        return ret_msg
