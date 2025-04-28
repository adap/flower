from flwr.client import ClientApp
from flwr.common import Context
from flwr.common import Message
from flwr.common.constant import ErrorCode
from flwr.common.message import Error
import random
# Flower ClientApp
app = ClientApp()

@app.query()
def query(msg: Message, ctx: Context):
    try:
        if random.choice([True, False]):
            print("Error in client. Sending the Error reply.")
            raise ValueError()
        # load the model/dataset
        print(f"In Client msg.content.configs_records: {msg.content.configs_records}")
        return msg.create_reply(msg.content)
    except:
        return msg.create_error_reply(
            error=Error(code=ErrorCode.UNKNOWN, reason="Unknown")
        )
