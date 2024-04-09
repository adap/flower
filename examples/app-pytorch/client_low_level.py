from flwr.client import ClientApp
from flwr.common import Message, Context
from flwr.common import Message
from flwr.common.record import RecordSet, MetricsRecord

from task import Net, load_data
from task import train as train_fn
from low_level_utils import parameters_to_pytorch_state_dict, pytorch_to_parameter_record


# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, ctx: Context):

    # log local context
    if 'prev' in ctx.state.metrics_records:
        print(f"last round metrics were: {ctx.state.metrics_records['prev']}")
    else:
        print("no context info")

    # instantiate model
    model = Net()

    # load local data
    train_loader, val_loader  = load_data()

    # Get sent model
    fancy_parameters = msg.content.parameters_records['fancy_model']
    state_dict = parameters_to_pytorch_state_dict(fancy_parameters)
    model.load_state_dict(state_dict=state_dict, strict=True)

    # Get sent config
    fancy_config = msg.content.configs_records['fancy_config']

    # local training
    train_metrics = train_fn(model, train_loader, val_loader, epochs=fancy_config['epochs'], device='cpu')

    # Construct reply message carrying updated model paramters and generated metrics
    reply_content = RecordSet()
    reply_content.parameters_records['fancy_model_returned'] = pytorch_to_parameter_record(model)
    reply_content.metrics_records['train_metrics'] = MetricsRecord(train_metrics)

    # store metrics in context also
    ctx.state.metrics_records['prev'] = MetricsRecord(train_metrics)

    return msg.create_reply(reply_content)


@app.evaluate()
def eval(msg: Message, ctx: Context):
    print("`evaluate` is not implemented, echoing original message")
    return msg.create_reply(msg.content)


@app.query()
def query(msg: Message, ctx: Context):
    print("`query` is not implemented, echoing original message")
    return msg.create_reply(msg.content)
