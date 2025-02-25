:og:description: Learn how to use built-in modifiers to enhance the behaviour of a ClientApp in Flower for federated learning.
.. meta::
    :description: Learn how to use built-in modifiers to enhance the behaviour of a ClientApp in Flower for federated learning.

Use Built-in Mods
=================

.. note::

    This tutorial covers experimental features. The functionality and interfaces may
    change in future versions.

In this tutorial, we will learn how to utilize built-in mods to augment the behavior of
a ``ClientApp``. Mods (sometimes also called Modifiers) allow us to perform operations
before and after a task is processed in the ``ClientApp``.

What are Mods?
--------------

A Mod is a callable that wraps around a ``ClientApp``. It can manipulate or inspect the
incoming ``Message`` and the resulting outgoing ``Message``. The signature for a ``Mod``
is as follows:

.. code-block:: python

    ClientAppCallable = Callable[[Message, Context], Message]
    Mod = Callable[[Message, Context, ClientAppCallable], Message]

A typical mod function might look something like this:

.. code-block:: python

    from flwr.client.typing import ClientAppCallable
    from flwr.common import Context, Message
    
    def example_mod(msg: Message, ctx: Context, call_next: ClientAppCallable) -> Message:
        # Do something with incoming Message (or Context)
        # before passing it to the next layer in the chain.
        # This could be another Mod or, if this is the last Mod, the ClientApp itself.
        msg = call_next(msg, ctx)
        # Do something with outgoing Message (or Context)
        # before returning
        return msg

Using Mods
----------

Mods can be registered in two ways: **Global mods** and **Handler-specific mods**.

1. **Global mods**: These mods apply to all message handlers within the ``ClientApp``.
2. **Handler-specific mods**: These mods apply only to a specific message handler.

1. Registering Global Mods
~~~~~~~~~~~~~~~~~~~~~~~~~~

To use global mods in your ``ClientApp``, follow these steps:

Import the required mods
++++++++++++++++++++++++

.. code-block:: python

    import flwr as fl
    from flwr.client.mod import example_mod_1, example_mod_2

Create the ``ClientApp`` with global mods
+++++++++++++++++++++++++++++++++++++++++

Create your ``ClientApp`` and pass the mods as a list to the ``mods`` argument. The
order in which you provide the mods matters:

.. code-block:: python

    app = fl.client.ClientApp(
        client_fn=client_fn,  # Not needed if using decorators
        mods=[
            example_mod_1,  # Global Mod 1
            example_mod_2,  # Global Mod 2
        ],
    )

If you define message handlers using decorators instead of ``client_fn``, e.g.,
``@app.train()``, you do not need to pass the ``client_fn`` argument.

2. Registering Handler-specific Mods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Instead of applying mods globally, you can specify them for a particular handler:

.. code-block:: python

    import flwr as fl
    from flwr.client.mod import example_mod_3, example_mod_4

    app = fl.client.ClientApp()


    @app.train(mods=[example_mod_3, example_mod_4])
    def train(msg, ctx):
        # Training logic here
        return reply_msg
 
    @app.evaluate()
    def evalutate(msg, ctx):
        # Evaluate logic here
        return reply_msg

In this case, ``example_mod_3`` and ``example_mod_4`` are only applied to the ``train``
handler.

Order of execution
------------------

When the ``ClientApp`` runs, the mods execute in the following order:

1. **Global mods** (executed first, in the order they are provided)
2. **Handler-specific mods** (executed after global mods, in the order they are
   provided)
3. **Message handler** (core function that handles the incoming ``Message`` and returns
   the outgoing ``Message``)
4. **Handler-specific mods** (on the way back, in reverse order)
5. **Global mods** (on the way back, in reverse order)

Each mod has a chance to inspect and modify the incoming ``Message`` before passing it
to the next mod, and likewise with the outgoing ``Message`` before returning it up the
stack.

Example Execution Flow
~~~~~~~~~~~~~~~~~~~~~~

Assuming the following registration:

.. code-block:: python

    app = fl.client.ClientApp(mods=[global_mod_1, global_mod_2])


    @app.train(mods=[handler_mod_1, handler_mod_2])
    def train(msg, ctx):
        return msg.create_reply(fl.common.RecordSet())

The execution order would be:

1. ``global_mod_1`` (before handling)
2. ``global_mod_2`` (before handling)
3. ``handler_mod_1`` (before handling)
4. ``handler_mod_2`` (before handling)
5. ``train`` (message handler execution)
6. ``handler_mod_2`` (after handling)
7. ``handler_mod_1`` (after handling)
8. ``global_mod_2`` (after handling)
9. ``global_mod_1`` (after handling)

Conclusion
----------

By following this guide, you have learned how to effectively use mods to enhance your
``ClientApp``'s functionality. Remember that the order of mods is crucial and affects
how the input and output are processed.

Enjoy building a more robust and flexible ``ClientApp`` with mods!
