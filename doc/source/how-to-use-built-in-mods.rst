Use Built-in Mods
=================

**Note: This tutorial covers experimental features. The functionality and interfaces may change in future versions.**

In this tutorial, we will learn how to utilize built-in mods to augment the behavior of a ``ClientApp``. Mods (sometimes also called Modifiers) allow us to perform operations before and after a task is processed in the ``ClientApp``.

What are Mods?
--------------

A Mod is a callable that wraps around a ``ClientApp``. It can manipulate or inspect the incoming ``Message`` and the resulting outgoing ``Message``. The signature for a ``Mod`` is as follows:

.. code-block:: python

    ClientApp = Callable[[Message, Context], Message]
    Mod = Callable[[Message, Context, ClientApp], Message]

A typical mod function might look something like this:

.. code-block:: python

    def example_mod(msg: Message, ctx: Context, nxt: ClientApp) -> Message:
        # Do something with incoming Message (or Context)
        # before passing to the inner ``ClientApp``
        msg = nxt(msg, ctx)
        # Do something with outgoing Message (or Context)
        # before returning
        return msg

Using Mods
----------

To use mods in your ``ClientApp``, you can follow these steps:

1. Import the required mods
~~~~~~~~~~~~~~~~~~~~~~~~~~~

First, import the built-in mod you intend to use:

.. code-block:: python

    import flwr as fl
    from flwr.client.mod import example_mod_1, example_mod_2

2. Define your client function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Define your client function (``client_fn``) that will be wrapped by the mod(s):

.. code-block:: python

    def client_fn(cid):
        # Your client code goes here.
        return # your client

3. Create the ``ClientApp`` with mods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create your ``ClientApp`` and pass the mods as a list to the ``mods`` argument. The order in which you provide the mods matters:

.. code-block:: python

    app = fl.client.ClientApp(
        client_fn=client_fn,
        mods=[
            example_mod_1,  # Mod 1
            example_mod_2,  # Mod 2
        ]
    )

Order of execution
------------------

When the ``ClientApp`` runs, the mods are executed in the order they are provided in the list:

1. ``example_mod_1`` (outermost mod)
2. ``example_mod_2`` (next mod)
3. Message handler (core function that handles the incoming ``Message`` and returns the outgoing ``Message``)
4. ``example_mod_2`` (on the way back)
5. ``example_mod_1`` (outermost mod on the way back)

Each mod has a chance to inspect and modify the incoming ``Message`` before passing it to the next mod, and likewise with the outgoing ``Message`` before returning it up the stack.

Conclusion
----------

By following this guide, you have learned how to effectively use mods to enhance your ``ClientApp``'s functionality. Remember that the order of mods is crucial and affects how the input and output are processed.

Enjoy building a more robust and flexible ``ClientApp`` with mods!
