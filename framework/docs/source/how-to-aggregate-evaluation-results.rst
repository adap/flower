:og:description: Aggregate custom evaluation results from federated clients in Flower using a strategy that applies weighted averaging for metrics like accuracy.
.. meta::
    :description: Aggregate custom evaluation results from federated clients in Flower using a strategy that applies weighted averaging for metrics like accuracy.

.. |fedavg_link| replace:: ``FedAvg``

.. _fedavg_link: ref-api/flwr.serverapp.strategy.FedAvg.html

.. |metricrecord_link| replace:: ``MetricRecord``

.. _metricrecord_link: ref-api/flwr.app.MetricRecord.html

.. |message_link| replace:: ``Message``

.. _message_link: ref-api/flwr.app.Message.html

.. |recorddict_link| replace:: ``RecordDict``

.. _recorddict_link: ref-api/flwr.app.RecordDict.html

Aggregate evaluation results
============================

In the |fedavg_link|_ strategy, there is a default ``evaluate_metrics_aggr_fn`` that
performs a weighted aggregation of all |metricrecord_link|_ using a specified key. By
default, the weighting key is ``num-examples``, but you can modify it according to your
requirements.

Aggregate Custom Evaluation Results
-----------------------------------

The ``evaluate_metrics_aggr_fn`` can be customized to support any evaluation results
aggregation logic you need. Its definition is:

.. code-block:: python

    Callable[[list[RecordDict], str], MetricRecord]

It takes a list of |recorddict_link|_ and a weighting key as inputs and returns a
|metricrecord_link|_. For example, the function below extracts and returns the minimum
value for each metric key across all |message_link|_:

.. code-block:: python

    from flwr.app import MetricRecord, RecordDict


    def aggregate_metricrecords(
        records: list[RecordDict], weighting_metric_name: str
    ) -> MetricRecord:
        """Extract the minimum value for each metric key."""
        aggregated_metrics = MetricRecord()

        # Track current minimum per key in a plain dict,
        # then copy into MetricRecord at the end
        mins = {}

        for record in records:
            for record_item in record.metric_records.values():
                for key, value in record_item.items():
                    if key == weighting_metric_name:
                        # We exclude the weighting key from the aggregated MetricRecord
                        continue

                    if key in mins:
                        if value < mins[key]:
                            mins[key] = value
                    else:
                        mins[key] = value

        for key, value in mins.items():
            aggregated_metrics[key] = value

        return aggregated_metrics
