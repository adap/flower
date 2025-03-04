Flower Intelligence
===================

Flower Intelligence is a cross-platform inference library that let's user
seamlessly interact with Large-Language Models both locally and remotely in a
secure and private way. The library was created by the ``Flower Labs`` team that also created `Flower: A Friendly Federated AI Framework <https://flower.ai>`_.

We currently only provide a SDK for TypeScript/JavaScript.


Install
-------

.. tab-set::
    :sync-group: category

    .. tab-item:: TypeScript
        :sync: ts

        .. code-block:: bash

          npm i "@flwr/flwr"

    .. tab-item:: JavaScript
        :sync: js

        .. code-block:: bash

          npm i "@flwr/flwr"


Basic usage
-----------

This guide will help you get started quickly with Flower Intelligence. The library is designed to be easy to set up and use – even if you’re just starting out.

Flower Intelligence uses the Singleton design pattern. This means that you only have to create and configure one instance, which you can then use anywhere in your project. There's no need to worry about managing multiple copies of the library.

.. tab-set::
    :sync-group: category

    .. tab-item:: TypeScript
        :sync: ts

        .. code-block:: ts

            // Access the singleton instance
            const fi: FlowerIntelligence = FlowerIntelligence.instance;

    .. tab-item:: JavaScript
        :sync: js

        .. code-block:: js

            // Access the singleton instance
            const fi = FlowerIntelligence.instance;


Chatting with Flower Intelligence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once your instance is set up, you can start chatting! Here’s a basic example:

.. tab-set::
    :sync-group: category

    .. tab-item:: TypeScript
        :sync: ts

        .. code-block:: ts

            const reply: ChatResponseResult = await fi.chat("Why is the sky blue?");
            if (reply.ok){
                console.log(reply.message.content);
            }

    .. tab-item:: JavaScript
        :sync: js

        .. code-block:: js

            const reply = await fi.chat("Why is the sky blue?");
            console.log(reply.message.content);


That’s it – your message is sent, and the AI responds!

Examples
~~~~~~~~

Below are a few more examples to illustrate different ways to interact with the chat:

1. **Streaming Responses**

   Watch the response as it is being generated.

    .. tab-set::
        :sync-group: category

        .. tab-item:: TypeScript
            :sync: ts

            .. code-block:: ts

                const reply: ChatResponseResult = await fi.chat("Why is the sky blue?", {
                  stream: true,
                  onStreamEvent: (event: StreamEvent) => console.log(event.chunk)
                });
                if (reply.ok){
                  console.log("Full response:", reply.message.content);
                }

        .. tab-item:: JavaScript
            :sync: js

            .. code-block:: js

                const reply = await fi.chat("Why is the sky blue?", {
                  stream: true,
                  onStreamEvent: (event) => console.log(event.chunk)
                });
                console.log("Full response:", reply.message.content);

2. **Using Roles**

   Provide an array of messages to maintain conversation context. Instead of
   just passing a string to the ``chat`` method, you can use ``Message`` or an array
   of ``Message``. This lets you use different roles like system messages:

    .. tab-set::
        :sync-group: category

        .. tab-item:: TypeScript
            :sync: ts

            .. code-block:: ts

                const reply: ChatResponseResult = await fi.chat({
                  messages: [
                    { role: "system", content: "You are a friendly assistant that loves using emojies." }
                    { role: "user", content: "Why is the sky blue?" }
                  ]
                });
                if (reply.ok){
                  console.log(reply.message.content);
                }

        .. tab-item:: JavaScript
            :sync: js

            .. code-block:: js

                const reply = await fi.chat({
                  messages: [
                    { role: "system", content: "You are a friendly assistant that loves using emojies." }
                    { role: "user", content: "Why is the sky blue?" }
                  ]
                });
                console.log(reply.content);

.. note::
   Checkout out full examples over on GitHub for more information!

Flower Confidential Remote Compute
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. warning::
   Flower Confidential Remote Compute is available in private beta. If you are interested in using Confidential Remote Compute, please apply for Early Access via the `Flower Intelligence Pilot Program <https://forms.gle/J8pFpMrsmek2VFKq8>`_.

Flower Intelligence prioritizes local inference, but also allows to privately handoff
the compute to the Flower Confidential Remote Compute service when local resources are scarce. You can find more
information on `flower.ai/intelligence <https://flower.ai/intelligence>`_.

This feature is turned off by default, and can be enabled by using the ``remoteHandoff``
attribute of the ``FlowerIntelligence`` object.

You will also need to provide a valid API key via the ``apiKey`` attribute.

.. tab-set::
    :sync-group: category

    .. tab-item:: TypeScript
        :sync: ts

        .. code-block:: ts

            // Access the singleton instance
            const fi: FlowerIntelligence = FlowerIntelligence.instance;

            // Enable remote processing and provide your API key
            fi.remoteHandoff = true;
            fi.apiKey = "YOUR_API_KEY";

    .. tab-item:: JavaScript
        :sync: js

        .. code-block:: js

            // Access the singleton instance
            const fi = FlowerIntelligence.instance;

            // Enable remote processing and provide your API key
            fi.remoteHandoff = true;
            fi.apiKey = "YOUR_API_KEY";

References
----------

Information-oriented API reference and other reference material.

.. toctree::
   :maxdepth: 1
   :glob:
   :caption: Reference docs

   ref-models
   ts-api-ref/index

Join the Flower Community
-------------------------

The Flower Community is growing quickly - we're a friendly group of researchers, engineers, students, professionals, academics, and other enthusiasts.

.. button-link:: https://flower.ai/join-slack
    :color: primary
    :shadow:

    Join us on Slack

