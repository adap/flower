"""Each client runs guardrail checks on local domain data. Prompts never leave."""
from flwr.client import ClientApp, NumPyClient
from fedguardrails.datasets import get_domain_dataset, get_domain_name
from fedguardrails.guardrails import run_guardrails


class GuardrailClient(NumPyClient):
    def __init__(self, partition_id):
        self.partition_id = partition_id
        self.domain = get_domain_name(partition_id)

    def fit(self, parameters, config):
        num_checks = int(config.get("num_checks", 20))
        dataset = get_domain_dataset(self.partition_id, num_checks)

        tp = tn = fp = fn = 0
        total_latency = 0.0
        inj_count = pii_count = top_count = 0

        for item in dataset:
            result = run_guardrails(item["prompt"])
            total_latency += result["latency_ms"]
            inj_count += int(result["injection"])
            pii_count += int(result["pii"])
            top_count += int(result["topic"])

            if item["expected_safe"] and result["is_safe"]:
                tn += 1
            elif item["expected_safe"] and not result["is_safe"]:
                fp += 1
            elif not item["expected_safe"] and not result["is_safe"]:
                tp += 1
            else:
                fn += 1

        total = len(dataset)
        accuracy = (tp + tn) / total if total > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        avg_latency = total_latency / total if total > 0 else 0.0

        print(f"\n  [{self.domain.upper()}] acc={accuracy:.0%} prec={precision:.0%} "
              f"rec={recall:.0%} f1={f1:.0%} latency={avg_latency:.3f}ms "
              f"(TP={tp} TN={tn} FP={fp} FN={fn})")

        metrics = {
            "domain": float(self.partition_id),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "avg_latency_ms": avg_latency,
            "tp": float(tp),
            "tn": float(tn),
            "fp": float(fp),
            "fn": float(fn),
            "injection_flags": float(inj_count),
            "pii_flags": float(pii_count),
            "topic_flags": float(top_count),
        }
        return parameters, total, metrics

    def evaluate(self, parameters, config):
        return 0.0, 1, {}


def client_fn(context):
    partition_id = context.node_config["partition-id"]
    return GuardrailClient(partition_id).to_client()


app = ClientApp(client_fn=client_fn)