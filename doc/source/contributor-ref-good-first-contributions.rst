Good first contributions
========================

We welcome contributions to Flower! However, it is not always easy to know
where to start. We therefore put together a few recommendations on where to
start to increase your chances of getting your PR accepted into the Flower
codebase.


Where to start
--------------

Until the Flower core library matures it will be easier to get PR's accepted if
they only touch non-core areas of the codebase. Good candidates to get started
are:

- Documentation: What's missing? What could be expressed more clearly? Are there any typos? You can also check-out our `contributing guide for translations <https://flower.ai/docs/baselines/how-to-contribute-translations.html>`_.
- Baselines: See below.
- Examples: See below.

Another part of the codebase that can make for good first contributions would be strategies.
You can already take a look at existing strategies like `FedAvg <https://github.com/adap/flower/blob/main/src/py/flwr/server/strategy/fedavg.py>`_ to better understand how they work and check-out our `strategy implementation guide <https://flower.ai/docs/framework/how-to-implement-strategies.html>`_ for further information.


Request for Flower Baselines
----------------------------

If you are not familiar with Flower Baselines, you should probably check-out our `contributing guide for baselines <https://flower.ai/docs/baselines/how-to-contribute-baselines.html>`_.

You should then check out the open
`issues <https://github.com/adap/flower/issues?q=is%3Aopen+is%3Aissue+label%3A%22new+baseline%22>`_ for baseline requests.
If you find a baseline that you'd like to work on and that has no assignees, feel free to assign it to yourself and start working on it!

Otherwise, if you don't find a baseline you'd like to work on, be sure to open a new issue with the baseline request template!

Request for examples
--------------------

We wish we had more time to write usage examples because we believe they help
users to get started with building what they want to build. Here are a few
ideas where we'd be happy to accept a PR:

- Android ONNX on-device training
