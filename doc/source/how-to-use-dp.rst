Use Differential Privacy
------------------------------
Below, we explain how users can utilize differential privacy in the Flower framework. If you are not familiar with differential privacy, you can refer to :doc:`explanation-differential-privacy`.

.. warning::

   Differential Privacy in Flower is at the experimental phase. If you plan to use these features in a production environment or with sensitive data, please contact us to discuss your needs and to receive guidance on how to best use these features.


Central Differential Privacy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This approach consists of two seprate phases: clipping of the updates and adding noise to the aggregated model.
For the clipping phase, Flower framework has made it possible to decide whether to perform clipping on the server side or the client side.

- **Server-side Clipping**: This approach has the advantage of the server enforcing uniform clipping across all clients' updates and reducing the communication overhead for clipping values. However, it also has the disadvantage of increasing the computational load on the server due to the need to perform the clipping operation for all clients.
- **Client-side Clipping**: This approach has the advantage of reducing the computational overhead on the server. However, it also has the disadvantage of lacking centralized control, as the server has less control over the clipping process.



Server-side Clipping
^^^^^^^^^^^^^^^^^^^^
To utilize the central DP with server side clipping, there are two wrapper classes :code:`DifferentialPrivacyServerSideFixedClipping` and :code:`DifferentialPrivacyServerSideAdaptiveClipping` to be used for fixed or adaptive clipping.

.. image:: ./_static/DP/serversideCDP.png
  :align: center
  :width: 700
  :alt: server side clipping


Below is a sample code that enables a strategy using :code:`DifferentialPrivacyServerSideFixedClipping` wrapper class. The same approach can be used with :code:`DifferentialPrivacyServerSideAdaptiveClipping` by adjusting the corresponding input parameters.

.. code-block:: python

  # Server-side:
  from flwr.server.strategy import DifferentialPrivacyClientSideFixedClipping

  # Configure the strategy
  strategy = fl.server.strategy.FedAvg(...)
  # Wrap the strategy with the DifferentialPrivacyServerSideFixedClipping wrapper
  dp_strategy = DifferentialPrivacyServerSideFixedClipping(
    strategy, cfg.noise_multiplier, cfg.clipping_norm, cfg.num_sampled_clients
  )



Client-side Clipping
^^^^^^^^^^^^^^^^^^^^
For client-side clipping, the server sends the clipping value to selected clients on each round. Clients can use existing Flower :code:`Mods` [5] to perform the clipping.
Two mods are available for fixed and adaptive client-side clipping: :code:`fixedclipping_mod` and :code:`adaptiveclipping_mod` with corresponding server-side wrappers :code:`DifferentialPrivacyClientSideFixedClipping` and :code:`DifferentialPrivacyClientSideAdaptiveClipping`.

.. image:: ./_static/DP/clientsideCDP.png
  :align: center
  :width: 800
  :alt: client side clipping


Below is a sample code that enables a strategy using :code:`DifferentialPrivacyClientSideFixedClipping` wrapper class. On the client, `fixedclipping_mod` can be added to the client-side mods:

.. code-block:: python

  # Server-side:
  from flwr.server.strategy import DifferentialPrivacyClientSideFixedClipping

  # Configure the strategy
  strategy = fl.server.strategy.FedAvg(...)
  # Wrap the strategy with the DifferentialPrivacyClientSideFixedClipping wrapper
  dp_strategy = DifferentialPrivacyClientSideFixedClipping(
    strategy, cfg.noise_multiplier, cfg.clipping_norm, cfg.num_sampled_clients
  )


.. code-block:: python

  # Client-side:
  from flwr.client.mod.centraldp_mods import fixedclipping_mod

  # Add fixedclipping_mod to the client-side mods
  app = fl.client.ClientApp(client_fn=client_fn, mods=[fixedclipping_mod])


Please note that the order of mods, especially those that modify parameters, is important when using multiple modifiers. Typically, differential privacy (DP) modifiers should be the last to operate on parameters.