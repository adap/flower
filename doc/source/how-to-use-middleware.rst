Use Built-in Middleware Layers
==============================

In this tutorial, we will learn how to utilize built-in middleware layers to augment the behavior of a flower callable. Middleware allows us to perform operations before and after a task is processed in the flower callable.

What is Middleware?
-------------------

Middleware is a callable that wraps around a Flower Callable. It can manipulate or inspect incoming tasks (``TaskIns``) in the ``Fwd`` and the resulting tasks (``TaskRes``) in the ``Bwd``. The signature for a middleware layer (``Layer``) is as follows:

.. code-block:: python

    FlowerCallable = Callable[[Fwd], Bwd]
    Layer = Callable[[Fwd, FlowerCallable], Bwd]

A typical middleware function might look something like this:

.. code-block:: python

    def example_middleware(fwd: Fwd, fc: FlowerCallable) -> Bwd:
        # Do something with Fwd before passing to the inner flower callable.
        bwd = fc(fwd)
        # Do something with Bwd before returning.
        return bwd

Using Middleware Layers
-----------------------

To use middleware layers in your flower callable, you can follow these steps:

1. Import the Required Middleware
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First, import the built-in middleware layers you intend to use:

.. code-block:: python

    import flwr as fl
    from flwr.client.middleware import example_middleware1, example_middleware2

2. Define Your Client Function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Define your client function (``client_fn``) that will be wrapped by the middleware:

.. code-block:: python

    def client_fn(cid):
        # Your client code goes here.
        return # your client

3. Create the FlowerCallable with Middleware
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create your flower callable and pass the middleware layers as a list to the ``middleware`` argument. The order in which you provide the middleware layers matters:

.. code-block:: python

    flower = fl.app.Flower(
        client_fn=client_fn,
        middleware=[
            example_middleware1,  # Middleware layer 1
            example_middleware2,  # Middleware layer 2
        ]
    )

Order of Execution
------------------

When the flower callable runs, the middleware layers are executed in the order they are provided in the list:

1. ``example_middleware1`` (outermost layer)
2. ``example_middleware2`` (next layer)
3. Message handler (core app functionality)
4. ``example_middleware2`` (on the way back)
5. ``example_middleware1`` (outermost layer on the way back)

Each middleware has a chance to inspect and modify the ``TaskIns`` in the ``Fwd`` before passing it to the next layer, and likewise with the ``TaskRes`` in the ``Bwd`` before returning it up the stack.

Conclusion
----------

By following this guide, you have learned how to effectively use middleware layers to enhance your flower callable's functionality. Remember that the order of middleware is crucial and affects how the input and output are processed.

Enjoy building more robust and flexible flower callables with middleware layers!
