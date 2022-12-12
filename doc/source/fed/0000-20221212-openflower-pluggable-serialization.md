---
fed-number: 0000
title: OpenFlower Pluggable Serialization
authors: ["@danieljanes"]
creation-date: 2022-12-12
last-updated: 2022-12-12
status: provisional
---

# OpenFlower Pluggable Serialization

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Summary](#summary)
- [Motivation](#motivation)
  - [Goals](#goals)
  - [Non-Goals](#non-goals)
- [Proposal](#proposal)
- [Drawbacks](#drawbacks)
- [Alternatives Considered](#alternatives-considered)
- [Appendix](#appendix)

## Summary

[TODO - sentence 1: summary of the problem]

This doc outlines a proposal for pluggable serialization that combines ease-of-use with 

## Motivation

[TODO]

### Goals

[TODO]

### Non-Goals

[TODO]

## Proposal

[TODO]

### Diagram

```{mermaid}

    sequenceDiagram
        participant Strategy
        participant S as Flower Server<br/>start_server
        participant C1 as Flower Client
        participant C2 as Flower Client
        Note left of S: Get initial <br/>model parameters
        S->>Strategy: initialize_parameters
        activate Strategy
        Strategy-->>S: Parameters
        deactivate Strategy

        Note left of S: Federated<br/>Training
        rect rgb(249, 219, 130)

        S->>Strategy: configure_fit
        activate Strategy
        Strategy-->>S: List[Tuple[ClientProxy, FitIns]]
        deactivate Strategy

        S->>C1: FitIns
        activate C1
        S->>C2: FitIns
        activate C2

        C1-->>S: FitRes
        deactivate C1
        C2-->>S: FitRes
        deactivate C2

        S->>Strategy: aggregate_fit<br/>List[FitRes]
        activate Strategy
        Strategy-->>S: Aggregated model parameters
        deactivate Strategy

        end

        Note left of S: Centralized<br/>Evaluation
        rect rgb(249, 219, 130)

        S->>Strategy: evaluate
        activate Strategy
        Strategy-->>S: Centralized evaluation result
        deactivate Strategy

        end

        Note left of S: Federated<br/>Evaluation
        rect rgb(249, 219, 130)

        S->>Strategy: configure_evaluate
        activate Strategy
        Strategy-->>S: List[Tuple[ClientProxy, EvaluateIns]]
        deactivate Strategy

        S->>C1: EvaluateIns
        activate C1
        S->>C2: EvaluateIns
        activate C2

        C1-->>S: EvaluateRes
        deactivate C1
        C2-->>S: EvaluateRes
        deactivate C2

        S->>Strategy: aggregate_evaluate<br/>List[EvaluateRes]
        activate Strategy
        Strategy-->>S: Aggregated evaluation results
        deactivate Strategy

        end

        Note left of S: Next round, continue<br/>with federated training
```

## Drawbacks

[TODO]

## Alternatives Considered

### Serialize to/from NumPy

[TODO]

### [Alternative 2]

[TODO]

## Appendix
