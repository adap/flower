"""Server aggregates guardrail metrics across domains."""
from flwr.server import ServerApp, ServerConfig
from flwr.server.strategy import FedAvg
from fedguardrails.datasets import DOMAIN_NAMES


def fit_metrics_aggregation_fn(metrics):
    print(f"\n{'='*65}")
    print(f"  FEDERATED GUARDRAIL BENCHMARKING REPORT")
    print(f"{'='*65}")

    total_tp = total_tn = total_fp = total_fn = 0
    total_examples = 0

    for num_examples, m in metrics:
        idx = int(m.get("domain", 0))
        domain = DOMAIN_NAMES[idx % len(DOMAIN_NAMES)]
        tp = int(m["tp"])
        tn = int(m["tn"])
        fp = int(m["fp"])
        fn = int(m["fn"])

        total_tp += tp
        total_tn += tn
        total_fp += fp
        total_fn += fn
        total_examples += num_examples

        print(f"\n  [{domain.upper()}]")
        print(f"    Accuracy  : {m['accuracy']:.0%}")
        print(f"    Precision : {m['precision']:.0%}")
        print(f"    Recall    : {m['recall']:.0%}")
        print(f"    F1 Score  : {m['f1']:.0%}")
        print(f"    Latency   : {m['avg_latency_ms']:.3f} ms")
        print(f"    TP={tp}  TN={tn}  FP={fp}  FN={fn}")
        print(f"    Injection : {int(m['injection_flags'])} | "
              f"PII : {int(m['pii_flags'])} | "
              f"Topic : {int(m['topic_flags'])}")

    # Aggregate
    acc = (total_tp + total_tn) / total_examples if total_examples > 0 else 0
    prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

    print(f"\n  {'─'*55}")
    print(f"  AGGREGATE ({len(metrics)} domains, {total_examples} prompts)")
    print(f"  {'─'*55}")
    print(f"    Accuracy  : {acc:.0%}")
    print(f"    Precision : {prec:.0%}")
    print(f"    Recall    : {rec:.0%}")
    print(f"    F1 Score  : {f1:.0%}")
    print(f"{'='*65}\n")

    return {"accuracy": acc, "f1": f1}


def fit_config_fn(server_round):
    return {"num_checks": 20}


strategy = FedAvg(
    fraction_fit=1.0,
    fraction_evaluate=0.0,
    min_available_clients=3,
    fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
    on_fit_config_fn=fit_config_fn,
)

app = ServerApp(
    config=ServerConfig(num_rounds=1),
    strategy=strategy,
)