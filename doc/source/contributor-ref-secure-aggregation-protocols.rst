Secure Aggregation Protocols
============================

Include SecAgg, SecAgg+, and LightSecAgg protocol. The LightSecAgg protocol has not been
implemented yet, so its diagram and abstraction may not be accurate in practice. The
SecAgg protocol can be considered as a special case of the SecAgg+ protocol.

The ``SecAgg+`` abstraction
---------------------------

In this implementation, each client will be assigned with a unique index (int) for
secure aggregation, and thus many python dictionaries used have keys of int type rather
than ClientProxy type.

The Flower server will execute and process received results in the following order:

.. mermaid::

    sequenceDiagram
        participant ServerApp as ServerApp (in SuperLink)
        participant SecAggPlusWorkflow
        participant ClientApp as secaggplus_mod
        participant RealClientApp as ClientApp (in SuperNode)

        ServerApp->>SecAggPlusWorkflow: invoke

        rect rgb(235, 235, 235)
        note over SecAggPlusWorkflow,ClientApp: Stage 0: Setup
        SecAggPlusWorkflow-->>ClientApp: Send SecAgg+ configuration
        ClientApp-->>SecAggPlusWorkflow: Send public keys
        end

        rect rgb(220, 220, 220)
        note over SecAggPlusWorkflow,ClientApp: Stage 1: Share Keys
        SecAggPlusWorkflow-->>ClientApp: Broadcast public keys
        ClientApp-->>SecAggPlusWorkflow: Send encrypted private key shares
        end

        rect rgb(235, 235, 235)
        note over SecAggPlusWorkflow,RealClientApp: Stage 2: Collect Masked Vectors
        SecAggPlusWorkflow-->>ClientApp: Forward the received shares
        ClientApp->>RealClientApp: fit instruction
        activate RealClientApp
        RealClientApp->>ClientApp: updated model
        deactivate RealClientApp
        ClientApp-->>SecAggPlusWorkflow: Send masked model parameters
        end

        rect rgb(220, 220, 220)
        note over SecAggPlusWorkflow,ClientApp: Stage 3: Unmask
        SecAggPlusWorkflow-->>ClientApp: Request private key shares
        ClientApp-->>SecAggPlusWorkflow: Send private key shares
        end
        SecAggPlusWorkflow->>SecAggPlusWorkflow: Unmask Aggregated Model
        SecAggPlusWorkflow->>ServerApp: Aggregated Model

