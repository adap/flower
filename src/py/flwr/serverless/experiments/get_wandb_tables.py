import os
import numpy as np
from dotenv import load_dotenv
import wandb


load_dotenv()


def get_run_group(
    runs,
    is_async: bool = False,
    skew_factor: int = 0,
    num_nodes: int = 2,
    strategy: str = "fedavg",
):
    run_group = []
    for run in runs:
        if (
            run.config["use_async"] == is_async
            and run.config["skew_factor"] == skew_factor
            and run.config["num_nodes"] == num_nodes
            and run.config["strategy"] == strategy
        ):
            if "test_accuracy" in run.summary and run.summary["test_accuracy"] > 0:
                run_group.append(run)
    return run_group


def get_run_group_for_wikitext(
    runs,
    is_async: bool = False,
    num_nodes: int = 2,
    strategy: str = "fedavg",
    model_name="EleutherAI/pythia-14M",
):
    run_group = []
    for run in runs:
        # print(
        #     f"run config: {run.config['use_async']}, {run.config['num_nodes']}, {run.config.get('strategy', 'fedavg')}"
        # )
        # print(f"accuracy: {run.summary.get('eval_accuracy')}")
        if (
            run.config["use_async"] == is_async
            and run.config["num_nodes"] == num_nodes
            and run.config.get("strategy", "fedavg") == strategy
            and run.config["model_name"] == model_name
        ):
            if "eval_accuracy" in run.summary and run.summary["eval_accuracy"] > 0:
                run_group.append(run)
    return run_group


def get_mean_std(run_group, metric="test_accuracy"):
    metric_values = []
    for run in run_group:
        if metric in run.summary:
            metric_values.append(run.summary[metric])
    if len(metric_values) == 0:
        return 0, 0
    return round(sum(metric_values) / len(metric_values), 3), round(
        np.std(metric_values), 3
    )


def get_exp1_2_tables(wandb_project: str = "mnist"):
    # list runs
    api = wandb.Api()
    wandb_entity = os.getenv("WANDB_ENTITY")
    runs = api.runs(
        path=f"{wandb_entity}/{wandb_project}",
        # successful only
        filters={
            "state": "finished",
        },
    )

    def f(is_async, skew_factor, num_nodes, strategy):
        run_group = get_run_group(
            runs,
            is_async=is_async,
            skew_factor=skew_factor,
            num_nodes=num_nodes,
            strategy=strategy,
        )
        print(
            f"found {len(run_group)} runs for {strategy}, {num_nodes} nodes, skew {skew_factor}, async {is_async}"
        )
        mean, std = get_mean_std(run_group, metric="test_accuracy")
        ci95 = 1.96 * std / np.sqrt(len(run_group))
        ci95 = round(ci95, 3)
        return f"{mean} $\\pm$ {ci95}".replace("0.", ".")

    # run_group = get_run_group(runs, is_async=False, skew_factor=0, num_nodes=2, strategy="fedavg")
    # print(get_mean_std(run_group, metric="test_accuracy"))
    for d in [0, 0.9, 0.99]:
        latex_table = (
            """
        \\toprule
        & \\multicolumn{3}{c}{Number of Nodes} \\\\
        Strategy & 2 & 3 & 5  \\\\
        \\midrule
        """
            + f"""
        FedAvg & {f(False, d, 2, "fedavg")} & {f(False, d, 3, "fedavg")} & {f(False, d, 5, "fedavg")} \\\\
        FedAvgM & {f(False, d, 2, "fedavgm")} & {f(False, d, 3, "fedavgm")} & {f(False, d, 5, "fedavgm")} \\\\
        FedAdam & {f(False, d, 2, "fedadam")} & {f(False, d, 3, "fedadam")} & {f(False, d, 5, "fedadam")} \\\\
        \\midrule

        FedAvg (async) & {f(True, d, 2, "fedavg")} & {f(True, d, 3, "fedavg")} & {f(True, d, 5, "fedavg")} \\\\
        FedAvgM (async) & {f(True, d, 2, "fedavgm")} & {f(True, d, 3, "fedavgm")} & {f(True, d, 5, "fedavgm")} \\\\
        FedAdam (async) & {f(True, d, 2, "fedadam")} & {f(True, d, 3, "fedadam")} & {f(True, d, 5, "fedadam")} \\\\
        
        \\bottomrule
        """
        )
        print(latex_table)
        print(f"Above is for disparity {d}")

    # another table with sync, async as rows, skew, accuracy as columns
    latex_table = (
        """
    \\toprule
    & \\multicolumn{3}{c}{Disparity} \\\\
    Strategy & 0 & 0.9 & 1  \\\\
    \\midrule
    """
        + f"""
    sync & {f(False, 0.0, 2, "fedavg")} & {f(False, 0.9, 2, "fedavg")} & {f(False, 1, 2, "fedavg")} \\\\
    async & {f(True, 0.0, 2, "fedavg")} & {f(True, 0.9, 2, "fedavg")} & {f(True, 1, 2, "fedavg")} \\\\
    
    \\bottomrule
    """
    )

    print(latex_table)


def get_exp3_tables(wandb_project="wikitext"):
    # list runs
    api = wandb.Api()
    wandb_entity = os.getenv("WANDB_ENTITY")
    runs = api.runs(
        path=f"{wandb_entity}/{wandb_project}",
        # successful only
        filters={
            "state": "finished",
        },
    )

    def f(is_async, skew_factor, num_nodes, strategy):
        run_group = get_run_group_for_wikitext(
            runs,
            is_async=is_async,
            num_nodes=num_nodes,
            strategy=strategy,
        )
        print(
            f"found {len(run_group)} runs for {strategy}, {num_nodes} nodes, async {is_async}"
        )
        mean, std = get_mean_std(run_group, metric="eval_accuracy")
        ci95 = 1.96 * std / np.sqrt(len(run_group))
        ci95 = round(ci95, 3)
        return f"{mean} $\\pm$ {ci95}".replace("0.", ".")

    d = 0
    # compare num_nodes, sync vs async
    latex_table = (
        """
    \\toprule
    & \\multicolumn{2}{c}{Number of Nodes} \\\\
    Strategy & 2 & 3 & 5  \\\\
    \\midrule
    """
        + f"""
    FedAvg & {f(False, d, 2, "fedavg")} & {f(False, d, 3, "fedavg")} & {f(False, d, 5, "fedavg")} \\\\
    FedAvg (async) & {f(True, d, 2, "fedavg")} & {f(True, d, 3, "fedavg")} & {f(True, d, 5, "fedavg")} \\\\
    \\bottomrule
    """
    )
    print(latex_table)


if __name__ == "__main__":
    # get_exp1_2_tables("mnist")
    get_exp1_2_tables("cifar10")
    # get_exp3_tables("wikitext")
