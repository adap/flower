from flwr.client import ClientApp
from flwr.common import Message, Context


def hello_world_mod(msg, ctx, call_next) -> Message:
    print("Hello, ...[pause for dramatic effect]...")
    out = call_next(msg, ctx)
    print("...[pause was long enough]... World!")
    return out


# Flower ClientApp
app = ClientApp(
    mods=[
        hello_world_mod,
    ],
)


@app.train()
def train(msg: Message, ctx: Context):
    print("`train` is not implemented, echoing original message")
    return msg.create_reply(msg.content)


@app.evaluate()
def eval(msg: Message, ctx: Context):
    print("`evaluate` is not implemented, echoing original message")
    return msg.create_reply(msg.content)


@app.query()
def query(msg: Message, ctx: Context):
    print("`query` is not implemented, echoing original message")
    return msg.create_reply(msg.content)
