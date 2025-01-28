Public and private APIs
=======================

In Python, everything is public. To enable developers to understand which components can
be relied upon, Flower declares a public API. Components that are part of the public API
can be relied upon. Changes to the public API are announced in the release notes and are
subject to deprecation policies.

Everything that is not part of the public API is part of the private API. Even though
Python allows accessing them, user code should never use those components. Private APIs
can change at any time, even in patch releases.

How can you determine whether a component is part of the public API or not? Easy:

- `Use the Flower API reference documentation <ref-api/flwr.html>`_
- `Use the Flower CLI reference documentation <ref-api-cli.html>`_

Everything listed in the reference documentation is part of the public API. This
document explains how Flower maintainers define the public API and how you can determine
whether a component is part of the public API or not by reading the Flower source code.

Flower public API
-----------------

Flower has a well-defined public API. Let's look at this in more detail.

.. important::

    Every component that is reachable by recursively following ``__init__.__all__``
    starting from the root package (``flwr``) is part of the public API.

If you want to determine whether a component (class/function/generator/...) is part of
the public API or not, you need to start at the root of the ``flwr`` package. Let's use
``tree -L 1 -d src/py/flwr`` to look at the Python sub-packages contained ``flwr``:

.. code-block:: bash

    flwr
    ├── cli
    ├── client
    ├── common
    ├── proto
    ├── server
    └── simulation

Contrast this with the definition of ``__all__`` in the root
``src/py/flwr/__init__.py``:

.. code-block:: python

    # From `flwr/__init__.py`
    __all__ = [
        "client",
        "common",
        "server",
        "simulation",
    ]

You can see that ``flwr`` has six subpackages (``cli``, ``client``, ``common``,
``proto``, ``server``, ``simulation``), but only four of them are "exported" via
``__all__`` (``client``, ``common``, ``server``, ``simulation``).

What does this mean? It means that ``client``, ``common``, ``server`` and ``simulation``
are part of the public API, but ``cli`` and ``proto`` are not. The ``flwr`` subpackages
``cli`` and ``proto`` are private APIs. A private API can change completely from one
release to the next (even in patch releases). It can change in a breaking way, it can be
renamed (for example, ``flwr.cli`` could be renamed to ``flwr.command``) and it can even
be removed completely.

Therefore, as a Flower user:

- ``from flwr import client`` ✅ Ok, you're importing a public API.
- ``from flwr import proto`` ❌ Not recommended, you're importing a private API.

What about components that are nested deeper in the hierarchy? Let's look at Flower
strategies to see another typical pattern. Flower strategies like ``FedAvg`` are often
imported using ``from flwr.server.strategy import FedAvg``. Let's look at
``src/py/flwr/server/strategy/__init__.py``:

.. code-block:: python

    from .fedavg import FedAvg as FedAvg

    # ... more imports

    __all__ = [
        "FedAvg",
        # ... more exports
    ]

What's notable here is that all strategies are implemented in dedicated modules (e.g.,
``fedavg.py``). In ``__init__.py``, we *import* the components we want to make part of
the public API and then *export* them via ``__all__``. Note that we export the component
itself (for example, the ``FedAvg`` class), but not the module it is defined in (for
example, ``fedavg.py``). This allows us to move the definition of ``FedAvg`` into a
different module (or even a module in a subpackage) without breaking the public API (as
long as we update the import path in ``__init__.py``).

Therefore:

- ``from flwr.server.strategy import FedAvg`` ✅ Ok, you're importing a class that is
  part of the public API.
- ``from flwr.server.strategy import fedavg`` ❌ Not recommended, you're importing a
  private module.

This approach is also implemented in the tooling that automatically builds API reference
docs.

Flower public API of private packages
-------------------------------------

We also use this to define the public API of private subpackages. Public, in this
context, means the API that other ``flwr`` subpackages should use. For example,
``flwr.server.driver`` is a private subpackage (it's not exported via
``src/py/flwr/server/__init__.py``'s ``__all__``).

Still, the private sub-package ``flwr.server.driver`` defines a "public" API using
``__all__`` in ``src/py/flwr/server/driver/__init__.py``:

.. code-block:: python

    from .driver import Driver
    from .grpc_driver import GrpcDriver
    from .inmemory_driver import InMemoryDriver

    __all__ = [
        "Driver",
        "GrpcDriver",
        "InMemoryDriver",
    ]

The interesting part is that both ``GrpcDriver`` and ``InMemoryDriver`` are never used
by Flower framework users, only by other parts of the Flower framework codebase. Those
other parts of the codebase import, for example, ``InMemoryDriver`` using ``from
flwr.server.driver import InMemoryDriver`` (i.e., the ``InMemoryDriver`` exported via
``__all__``), not ``from flwr.server.driver.in_memory_driver import InMemoryDriver``
(``in_memory_driver.py`` is the module containing the actual ``InMemoryDriver`` class
definition).

This is because ``flwr.server.driver`` defines a public interface for other ``flwr``
subpackages. This allows codeowners of ``flwr.server.driver`` to refactor the package
without breaking other ``flwr``-internal users.
