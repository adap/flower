---
fed-number: 0002
title: Early stopping
authors: ["@adap"]
creation-date: 2023-03-28
last-updated: 2023-03-28
status: provisional
---

# FED Template

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Summary](#summary)
- [Motivation](#motivation)
  - [Goals](#goals)
  - [Non-Goals](#non-goals)
- [Proposal](#proposal)
<!-- - [Drawbacks](#drawbacks)
- [Alternatives Considered](#alternatives-considered)
- [Appendix](#appendix) -->

## Summary

Currently the training for a model will continue until the number of rounds set at the beginning is atteined.
This means that even if a model as achived his highest performance it can continue to train without any improvements for many rounds. 

It would be good to have a way to stop the training after a specified number of rounds where the performance has not imporved.

## Motivation

The main motivation behind this FED is this Slack [discussion](https://friendly-flower.slack.com/archives/C01RM6LMKQA/p1678706923101609). Early stopping is something that is already present in many frameworks, it would therefore be important for Flower to seemlessly integrate with it.

### Goals

* Provide a way to specify a number of rounds after which we stop the training if the model has not improved.

* Write an code example with early stoppage implemented. 

### Non-Goals

* Make breaking changes

* Go into the specific of early stoppage (the most important is to provide a way to do it, the specific of the early stoppage should only appear in the example).

## Proposal

Provide a hook which could receive the latest model, centralized and federated evaluation results and then could return a boolean to indicate to stop early. The strategy/hook author would then be responsible for handling and storing the best recent model.

<!-- ## Drawbacks

[TODO]

## Alternatives Considered

### [Alternative 1]

[TODO]

### [Alternative 2]

[TODO]

## Appendix -->
