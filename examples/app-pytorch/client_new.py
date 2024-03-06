import flwr
from flwr.common import Message, Context


# Run via `flower-client-app client:app`
app = flwr.client.ClientApp()


@app.train()
def train(msg: Message, ctx: Context):
    print("`train` is not implemented, echoing original message")
    return msg.create_reply(msg.content, ttl=msg.metadata.ttl)


@app.evaluate()
def eval(msg: Message, ctx: Context):
    print("`evaluate` is not implemented, echoing original message")
    return msg.create_reply(msg.content, ttl=msg.metadata.ttl)


@app.query()
def query(msg: Message, ctx: Context):
    print("`query` is not implemented, echoing original message")
    return msg.create_reply(msg.content, ttl=msg.metadata.ttl)
