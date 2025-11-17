:og:description: Learn how to use built-in modifiers to enhance the behaviour of a ClientApp in Flower for federated learning.
.. meta::
    :description: Learn how to use built-in modifiers to enhance the behaviour of a ClientApp in Flower for federated learning.

###################
 Use Built-in Mods
###################

.. note::

    This tutorial covers preview features. The functionality and interfaces may change
    in future versions.

In this tutorial, we will learn how to utilize built-in mods to augment the behavior of
a ``ClientApp``. Mods (sometimes also called Modifiers) allow us to perform operations
before and after a task is processed in the ``ClientApp``.

****************
 What are Mods?
****************

A Mod is a callable that wraps around a ``ClientApp``. It can manipulate or inspect the
incoming ``Message`` and the resulting outgoing ``Message``. The signature for a ``Mod``
is as follows:

.. code-block:: python

    ClientAppCallable = Callable[[Message, Context], Message]
    Mod = Callable[[Message, Context, ClientAppCallable], Message]

A typical mod function might look something like this:

.. code-block:: python

    from flwr.app import Context, Message
    from flwr.clientapp.typing import ClientAppCallable


    def example_mod(msg: Message, ctx: Context, call_next: ClientAppCallable) -> Message:
        # Do something with incoming Message (or Context)
        # before passing it to the next layer in the chain.
        # This could be another Mod or, if this is the last Mod, the ClientApp itself.
        msg = call_next(msg, ctx)
        # Do something with outgoing Message (or Context)
        # before returning
        return msg

************
 Using Mods
************

Mods can be registered in two ways: **Application-wide mods** and **Function-specific
mods**.

1. **Application-wide mods**: These mods apply to all functions within the
   ``ClientApp``.
2. **Function-specific mods**: These mods apply only to a specific function (e.g, the
   function decorated by ``@app.train()``)

1. Registering Application-wide Mods
====================================

To use application-wide mods in your ``ClientApp``, follow these steps:

Import the required mods
------------------------

.. code-block:: python

    import flwr as fl
    from flwr.clientapp.mod import example_mod_1, example_mod_2

Create the ``ClientApp`` with application-wide mods
---------------------------------------------------

Create your ``ClientApp`` and pass the mods as a list to the ``mods`` argument. The
order in which you provide the mods matters:

.. code-block:: python

    app = fl.clientapp.ClientApp(
        mods=[
            example_mod_1,  # Application-wide Mod 1
            example_mod_2,  # Application-wide Mod 2
        ],
    )

2. Registering Function-specific Mods
=====================================

Instead of applying mods to the entire ``ClientApp``, you can specify them for a
particular function:

.. code-block:: python

    import flwr as fl
    from flwr.clientapp.mod import example_mod_3, example_mod_4

    app = fl.clientapp.ClientApp()


    @app.train(mods=[example_mod_3, example_mod_4])
    def train(msg, ctx):
        # Training logic here
        return reply_msg


    @app.evaluate()
    def evaluate(msg, ctx):
        # Evaluation logic here
        return reply_msg

In this case, ``example_mod_3`` and ``example_mod_4`` are only applied to the ``train``
function.

********************
 Order of Execution
********************

When the ``ClientApp`` runs, the mods execute in the following order:

1. **Application-wide mods** (executed first, in the order they are provided)
2. **Function-specific mods** (executed after application-wide mods, in the order they
   are provided)
3. **ClientApp** (core function that handles the incoming ``Message`` and returns the
   outgoing ``Message``)
4. **Function-specific mods** (on the way back, in reverse order)
5. **Application-wide mods** (on the way back, in reverse order)

Each mod has a chance to inspect and modify the incoming ``Message`` before passing it
to the next mod, and likewise with the outgoing ``Message`` before returning it up the
stack.

Example Execution Flow
======================

Assuming the following registration:

.. code-block:: python

    app = fl.clientapp.ClientApp(mods=[example_mod_1, example_mod_2])


    @app.train(mods=[example_mod_3, example_mod_4])
    def train(msg, ctx):
        return Message(fl.app.RecordDict(), reply_to=msg)


    @app.evaluate()
    def evaluate(msg, ctx):
        return Message(fl.app.RecordDict(), reply_to=msg)

The execution order for an incoming **train** message is as follows:

1. ``example_mod_1`` (before handling)
2. ``example_mod_2`` (before handling)
3. ``example_mod_3`` (before handling)
4. ``example_mod_4`` (before handling)
5. ``train`` (handling message)
6. ``example_mod_4`` (after handling)
7. ``example_mod_3`` (after handling)
8. ``example_mod_2`` (after handling)
9. ``example_mod_1`` (after handling)

The execution order for an incoming **evaluate** message is as follows:

1. ``example_mod_1`` (before handling)
2. ``example_mod_2`` (before handling)
3. ``evaluate`` (handling message)
4. ``example_mod_2`` (after handling)
5. ``example_mod_1`` (after handling)

************
 Conclusion
************

By following this guide, you have learned how to effectively use mods to enhance your
``ClientApp``'s functionality. Remember that the order of mods is crucial and affects
how the input and output are processed.

Enjoy building a more robust and flexible ``ClientApp`` with mods!
