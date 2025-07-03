"""quickstart-catboost: A Flower / CatBoost app."""

import json

from catboost import CatBoostClassifier, Pool
from quickstart_catboost.task import load_data, model_temp_file

from flwr.client import ClientApp
from flwr.common import Context, Message, ConfigRecord, RecordDict

# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context):

    # Load partition
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    (X_train, y_train), (X_test, y_test), cat_features = load_data(partition_id, num_partitions)

    # Instantiate local model
    iterations = context.run_config["local-epochs"]
    learning_rate = context.run_config["learning-rate"]
    depth = context.run_config["depth"]
    cbc = CatBoostClassifier(
        iterations=iterations,
        one_hot_max_size=len(cat_features),
        learning_rate=learning_rate,
        model_size_reg=100,
        max_ctr_complexity=2,
        ctr_target_border_count=1,
        depth=depth,
        random_seed=42,
        verbose=False,
        cat_features=cat_features)

    # Load global model
    global_model_dict = msg.content["gmodel"]["model"]
    if global_model_dict:
        tmp_path = model_temp_file(global_model_dict, dump=True)
        cbc_init = CatBoostClassifier()
        cbc_init.load_model(tmp_path, "json")
    else:
        cbc_init = None

    # Local training
    cbc.fit(X_train, y_train, init_model=cbc_init)

    # Evaluation
    eval_pool = Pool(data=X_test, label=y_test, cat_features=cat_features)
    metrics = cbc.eval_metrics(eval_pool, metrics=["AUC"])
    auc = metrics["AUC"][-1]

    # Extract boosted trees and construct reply message
    tmp_path = model_temp_file(cbc, dump=False)
    model_dict = json.load(open(tmp_path, "r"))
    num_trees = len(model_dict["oblivious_trees"])
    model_dict["oblivious_trees"] = model_dict["oblivious_trees"][num_trees - iterations : num_trees]
    model_dict_b = json.dumps(model_dict).encode('utf-8')

    metric_and_model_record = ConfigRecord({"AUC": auc, "model_dict": model_dict_b})
    content = RecordDict({"metric_and_model": metric_and_model_record})
    return Message(content=content, reply_to=msg)
