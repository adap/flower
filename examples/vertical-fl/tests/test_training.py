"""Offline regression checks for the split-learning protocol.

Run from this example: python -m unittest discover -s tests -v
"""

import copy
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from datasets import Dataset, DatasetDict
from flwr.app import (
    Array,
    ArrayRecord,
    ConfigRecord,
    Context,
    Error,
    Message,
    RecordDict,
)
from flwr.supercore.task_identity import TaskIdentity

from vertical_fl import client_app, server_app, task

CONFIG = {
    "feature-splits": "2,3,2",
    "out-feature-dim-clientapp": 2,
    "learning-rate": 0.01,
    "reply-timeout": 120.0,
}


class VerticalTrainingTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        identity = patch.multiple(TaskIdentity, _run_id=42, _node_id=1, _task_id=5)
        identity.start()
        self.addCleanup(identity.stop)

    def request(self, node, kind, content=None):
        message = Message(
            content=content or RecordDict(), dst_node_id=node, message_type=kind
        )
        message.metadata.__dict__["_message_id"] = f"request-{node}"
        return message

    def context(self, node, party):
        return Context(
            run_id=42,
            node_id=node,
            node_config={"partition-id": party},
            state=RecordDict(),
            run_config=CONFIG,
        )

    def test_labels_follow_feature_row_order(self):
        size = 12
        table = Dataset.from_dict(
            {
                "Age": list(range(size)),
                "Sex": ["male"] * size,
                "Fare": list(range(size)),
                "Siblings/Spouses Aboard": [0] * size,
                "Name": ["Mr. Example"] * size,
                "Parents/Children Aboard": [0] * size,
                "Pclass": [1] * size,
                "Survived": [i % 2 for i in range(size)],
            }
        )
        with patch.object(task, "fds", None), patch(
            "flwr_datasets.federated_dataset.datasets.load_dataset",
            return_value=DatasetDict({"train": table}),
        ), patch.object(
            task,
            "load_and_preprocess",
            side_effect=lambda dataframe: dataframe["Age"].to_numpy(),
        ):
            features = task.load_data(0, [7])
            task.fds = None  # Labels load in a separate ServerApp process.
            labels = task.load_labels([7])
        self.assertFalse(np.array_equal(features, np.arange(size)))
        np.testing.assert_array_equal(labels, features % 2)

    def test_first_forward_saves_the_model_that_produced_embeddings(self):
        data = np.arange(18, dtype=np.float32).reshape(6, 3) / 18
        context = self.context(101, 0)
        with patch.object(client_app, "load_data", return_value=data):
            reply = client_app.get_emb(
                self.request(101, "query.generate_embeddings"), context
            )
        self.assertIn("model", context.state)
        model = task.ClientModel(3, 2)
        model.load_state_dict(context.state["model"].to_torch_state_dict())
        np.testing.assert_allclose(
            reply.content["arrays"]["embedding"].numpy(),
            model(torch.from_numpy(data)).detach().numpy(),
        )

    def test_split_updates_match_a_joint_network_with_persistent_adam(self):
        data = [
            np.random.default_rng(k).normal(size=(6, k + 2)).astype(np.float32)
            for k in range(3)
        ]
        labels = torch.tensor([[0.0], [1.0], [0.0], [1.0], [1.0], [0.0]])
        contexts = [self.context(101 + k, k) for k in range(3)]
        reference_models = [task.ClientModel(x.shape[1], 2) for x in data]
        for context, model in zip(contexts, reference_models):
            context.state["model"] = ArrayRecord(model.state_dict())
        head = task.ServerModel(6)
        reference_head = copy.deepcopy(head)
        reference_optimizers = [
            torch.optim.Adam(m.parameters(), lr=0.01) for m in reference_models
        ]
        head_optimizer = torch.optim.Adam(head.parameters(), lr=0.01)
        reference_head_optimizer = torch.optim.Adam(
            reference_head.parameters(), lr=0.01
        )

        for _ in range(3):
            replies = []
            for k, context in enumerate(contexts):
                with patch.object(client_app, "load_data", return_value=data[k]):
                    replies.append(
                        client_app.get_emb(
                            self.request(101 + k, "query.generate_embeddings"), context
                        )
                    )
            grid = Mock()
            grid.send_and_receive.return_value = replies[::-1]
            embeddings, mapping = server_app.get_remote_embeddings(
                grid, [101, 102, 103], 6, 2, timeout=1.0
            )
            reference_embeddings = torch.cat(
                [m(torch.from_numpy(x)) for m, x in zip(reference_models, data)], dim=1
            )
            torch.testing.assert_close(embeddings, reference_embeddings)
            for optimizer in [
                head_optimizer,
                reference_head_optimizer,
                *reference_optimizers,
            ]:
                optimizer.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy(head(embeddings), labels)
            reference_loss = torch.nn.functional.binary_cross_entropy(
                reference_head(reference_embeddings), labels
            )
            loss.backward()
            reference_loss.backward()
            head_optimizer.step()
            reference_head_optimizer.step()
            for optimizer in reference_optimizers:
                optimizer.step()

            def apply(messages, timeout):
                self.assertEqual(timeout, 1.0)
                acknowledgements = []
                for message in messages:
                    node = message.metadata.dst_node_id
                    message.metadata.__dict__["_message_id"] = f"gradient-{node}"
                    with patch.object(
                        client_app, "load_data", return_value=data[node - 101]
                    ):
                        acknowledgements.append(
                            client_app.apply_grad(message, contexts[node - 101])
                        )
                return acknowledgements

            grid.send_and_receive.side_effect = apply
            server_app.send_gradients_to_clients(
                grid, mapping, embeddings.grad.split(2, dim=1), timeout=1.0
            )
            for context, model in zip(contexts, reference_models):
                actual = context.state["model"].to_torch_state_dict()
                for name, expected in model.state_dict().items():
                    torch.testing.assert_close(actual[name], expected)
            for name, expected in reference_head.state_dict().items():
                torch.testing.assert_close(head.state_dict()[name], expected)

    def embedding_reply(self, node, party, values=None, error=False):
        request = self.request(node, "query.generate_embeddings")
        if error:
            return Message(
                error=Error(code=1, reason="Client failed"), reply_to=request
            )
        values = np.ones((6, 2), dtype=np.float32) if values is None else values
        return Message(
            RecordDict(
                {
                    "arrays": ArrayRecord({"embedding": Array(values)}),
                    "config": ConfigRecord({"pos": party}),
                }
            ),
            reply_to=request,
        )

    def test_incomplete_or_invalid_embeddings_fail(self):
        good = [self.embedding_reply(101 + k, k) for k in range(3)]
        cases = {
            "missing": good[:2],
            "duplicate-node": [good[0], good[0], good[2]],
            "unexpected-node": [good[0], good[1], self.embedding_reply(999, 2)],
            "duplicate-role": [good[0], good[1], self.embedding_reply(103, 1)],
            "invalid-role": [good[0], good[1], self.embedding_reply(103, 3)],
            "invalid-shape": [
                good[0],
                good[1],
                self.embedding_reply(103, 2, np.ones((5, 2))),
            ],
            "nonfinite": [
                good[0],
                good[1],
                self.embedding_reply(103, 2, np.full((6, 2), np.nan)),
            ],
            "client-error": [
                good[0],
                good[1],
                self.embedding_reply(103, 2, error=True),
            ],
        }
        for name, replies in cases.items():
            with self.subTest(name=name), self.assertRaises(RuntimeError):
                grid = Mock()
                grid.send_and_receive.return_value = replies
                server_app.get_remote_embeddings(
                    grid, [101, 102, 103], 6, 2, timeout=1.0
                )
            self.assertEqual(grid.send_and_receive.call_args.kwargs["timeout"], 1.0)

    def test_gradient_updates_require_all_acknowledgements(self):
        requests = [
            self.request(node, "train.apply_gradients") for node in [101, 102, 103, 999]
        ]
        replies = [Message(RecordDict(), reply_to=request) for request in requests]
        error = Message(
            error=Error(code=1, reason="Update failed"), reply_to=requests[2]
        )
        grid = Mock()
        mapping = {101: 0, 102: 1, 103: 2}
        gradients = tuple(torch.ones(6, 2) for _ in mapping)
        for acknowledgements in [
            replies[:2],
            [replies[0], replies[0], replies[2]],
            replies[:2] + [replies[3]],
            replies[:2] + [error],
        ]:
            with self.subTest(
                nodes=[r.metadata.src_node_id for r in acknowledgements]
            ), self.assertRaises(RuntimeError):
                grid.send_and_receive.return_value = acknowledgements
                server_app.send_gradients_to_clients(
                    grid, mapping, gradients, timeout=1.0
                )
        grid.send_and_receive.return_value = replies[:3][::-1]
        server_app.send_gradients_to_clients(grid, mapping, gradients, timeout=1.0)
        self.assertEqual(grid.send_and_receive.call_args.kwargs["timeout"], 1.0)
        grid.push_messages.assert_not_called()

    def test_reporting_does_not_switch_head_to_eval_or_repeat_forward(self):
        head = task.ServerModel(6)
        embeddings = torch.randn(6, 6)
        grid = Mock()
        grid.get_node_ids.return_value = [101, 102, 103]
        context = self.context(1, 0)
        context.run_config["num-server-rounds"] = 26
        with patch.object(server_app, "ServerModel", return_value=head), patch.object(
            server_app, "load_labels", return_value=np.array([0, 1, 0, 1, 1, 0])
        ), patch.object(
            server_app,
            "get_remote_embeddings",
            side_effect=lambda **kwargs: (
                embeddings.clone().requires_grad_(),
                {101: 0, 102: 1, 103: 2},
            ),
        ), patch.object(
            server_app, "send_gradients_to_clients"
        ) as send, patch.object(
            server_app, "log"
        ):
            server_app.main(grid, context)
        self.assertTrue(head.training)
        self.assertEqual(head.bn.num_batches_tracked.item(), 26)
        self.assertEqual(send.call_count, 26)


if __name__ == "__main__":
    unittest.main()
