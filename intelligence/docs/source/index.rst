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


Hello, Flower Intelligence!
---------------------------

In this introductory example, you learn how to quickly get started with the library. Flower Intelligence is built around the Singleton design pattern, meaning you only need to configure a single instance that can be reused throughout your project. This simple setup helps you integrate powerful AI capabilities with minimal overhead.

.. tab-set::
    :sync-group: category

    .. tab-item:: TypeScript
        :sync: ts

        .. code-block:: ts

            import { ChatResponseResult, FlowerIntelligence } from '@flwr/flwr';

            // Access the singleton instance
            const fi: FlowerIntelligence = FlowerIntelligence.instance;

            async function main() {
              const response: ChatResponseResult = await fi.chat("Why is the sky blue?");
            }

            await main().then().catch();

    .. tab-item:: JavaScript
        :sync: js

        .. code-block:: js

            import { FlowerIntelligence } from '@flwr/flwr';

            // Access the singleton instance
            const fi = FlowerIntelligence.instance;

            async function main() {
              const response = await fi.chat("Why is the sky blue?");
            }

            await main().then().catch();


Specify the model
-----------------

Flower Intelligence gives you the flexibility to choose the language model that best suits your application. By specifying a model in the chat options, you can easily switch between different AI models available in the ecosystem. For a full list of supported models, please refer to the :doc:`available models list <ref-models>`.

.. tab-set::
    :sync-group: category

    .. tab-item:: TypeScript
        :sync: ts

        .. code-block:: ts

            import { ChatResponseResult, FlowerIntelligence } from '@flwr/flwr';

            // Access the singleton instance
            const fi: FlowerIntelligence = FlowerIntelligence.instance;

            async function main() {
              const response: ChatResponseResult = await fi.chat('Why is the sky blue?', {
                model: 'meta/llama3.2-1b/instruct-fp16',
              });
            }

            await main().then().catch();

    .. tab-item:: JavaScript
        :sync: js

        .. code-block:: js

            import { FlowerIntelligence } from '@flwr/flwr';

            // Access the singleton instance
            const fi = FlowerIntelligence.instance;

            async function main() {
              const response = await fi.chat('Why is the sky blue?', {
                model: 'meta/llama3.2-1b/instruct-fp16',
              });
            }

            await main().then().catch();

Check for errors
----------------

Robust error handling is a key feature of Flower Intelligence. Instead of throwing exceptions that might crash your application, the library returns a response object that includes a dedicated failure property. This design allows you to inspect any issues via a structured failure object (as defined in :doc:`Failure <ts-api-ref/interfaces/Failure>`), enabling graceful error handling and improved application stability.

.. tab-set::
    :sync-group: category

    .. tab-item:: TypeScript
        :sync: ts

        .. code-block:: ts

            import { ChatResponseResult, FlowerIntelligence } from '@flwr/flwr';

            // Access the singleton instance
            const fi: FlowerIntelligence = FlowerIntelligence.instance;

            async function main() {
              const response: ChatResponseResult = await fi.chat('Why is the sky blue?', {
                model: 'meta/llama3.2-1b/instruct-fp16',
              });

              if (!response.ok) {
                console.error(`${response.failure.code}: ${response.failure.description}`);
              } else {
                console.log(response.message.content);
              }
            }

            await main().then().catch();

    .. tab-item:: JavaScript
        :sync: js

        .. code-block:: js

            import { FlowerIntelligence } from '@flwr/flwr';

            // Access the singleton instance
            const fi = FlowerIntelligence.instance;

            async function main() {
              const response = await fi.chat('Why is the sky blue?', {
                model: 'meta/llama3.2-1b/instruct-fp16',
              });

              if (!response.ok) {
                console.error(`${response.failure.code}: ${response.failure.description}`);
              } else {
                console.log(response.message.content);
              }
            }

            await main().then().catch();

Stream Responses
----------------

For applications that benefit from real-time feedback, Flower Intelligence supports response streaming. By enabling the stream option and providing a callback function, you can watch the AI’s response as it is being generated. This approach is ideal for interactive applications, as it lets you process partial responses immediately before the full answer is available.

.. tab-set::
    :sync-group: category

    .. tab-item:: TypeScript
        :sync: ts

        .. code-block:: ts

            import { ChatResponseResult, FlowerIntelligence, type StreamEvent } from '@flwr/flwr';

            // Access the singleton instance
            const fi: FlowerIntelligence = FlowerIntelligence.instance;

            async function main() {
              const response: ChatResponseResult = await fi.chat('Why is the sky blue?', {
                model: 'meta/llama3.2-1b/instruct-fp16',
                stream: true,
                onStreamEvent: (event: StreamEvent) => console.log(event.chunk)
              });

              if (!response.ok) {
                console.error(`${response.failure.code}: ${response.failure.description}`);
              } else {
                console.log('Full response:', response.message.content);
              }
            }

            await main().then().catch();

    .. tab-item:: JavaScript
        :sync: js

        .. code-block:: js

            import { FlowerIntelligence } from '@flwr/flwr';

            // Access the singleton instance
            const fi = FlowerIntelligence.instance;

            async function main() {
              const response = await fi.chat('Why is the sky blue?', {
                model: 'meta/llama3.2-1b/instruct-fp16',
                stream: true,
                onStreamEvent: (event) => console.log(event.chunk)
              });

              if (!response.ok) {
                console.error(`${response.failure.code}: ${response.failure.description}`);
              } else {
                console.log(response.message.content);
              }
            }

            await main().then().catch();

Use Roles
---------

To enhance conversation context, Flower Intelligence supports the use of different message roles. Instead of simply sending a single string, you can provide an array of :doc:`messages <ts-api-ref/interfaces/Message>` with designated roles such as “system” and “user.” This allows you to define the behavior and context of the conversation more clearly, ensuring that the assistant responds in a way that’s tailored to the scenario.

.. tab-set::
    :sync-group: category

    .. tab-item:: TypeScript
        :sync: ts

        .. code-block:: ts

            import { ChatResponseResult, FlowerIntelligence, type StreamEvent } from '@flwr/flwr';

            // Access the singleton instance
            const fi: FlowerIntelligence = FlowerIntelligence.instance;

            async function main() {
              const response: ChatResponseResult = await fi.chat({
                messages: [
                  { role: "system", content: "You are a friendly assistant that loves using emojies." }
                  { role: "user", content: "Why is the sky blue?" }
                ],
                model: 'meta/llama3.2-1b/instruct-fp16',
                stream: true,
                onStreamEvent: (event: StreamEvent) => console.log(event.chunk)
              });

              if (!response.ok) {
                console.error(`${response.failure.code}: ${response.failure.description}`);
              } else {
                console.log('Full response:', response.message.content);
              }
            }

            await main().then().catch();

    .. tab-item:: JavaScript
        :sync: js

        .. code-block:: js

            import { FlowerIntelligence } from '@flwr/flwr';

            // Access the singleton instance
            const fi = FlowerIntelligence.instance;

            async function main() {
              const response = await fi.chat({
                messages: [
                  { role: "system", content: "You are a friendly assistant that loves using emojies." }
                  { role: "user", content: "Why is the sky blue?" }
                ],
                model: 'meta/llama3.2-1b/instruct-fp16',
                stream: true,
                onStreamEvent: (event) => console.log(event.chunk)
              });

              if (!response.ok) {
                console.error(`${response.failure.code}: ${response.failure.description}`);
              } else {
                console.log(response.message.content);
              }
            }

            await main().then().catch();

Handle history
--------------

For context-aware conversations, managing a history of messages is essential. In this example, the conversation history is maintained in an array that includes both system and user messages. Each time a new message is sent, it is appended to the history, ensuring that the assistant has access to the full dialogue context. This method allows Flower Intelligence to generate responses that are informed by previous interactions, resulting in a more coherent and dynamic conversation.

.. tab-set::
    :sync-group: category

    .. tab-item:: TypeScript
        :sync: ts

        .. code-block:: ts

            import { ChatResponseResult, FlowerIntelligence, type StreamEvent } from '@flwr/flwr';

            // Access the singleton instance
            const fi: FlowerIntelligence = FlowerIntelligence.instance;

            // Initialize history with a system message.
            const history: Message[] = [
              { role: "system", content: "You are a friendly assistant that loves using emojis." }
            ];

            // Function to chat while preserving conversation history.
            async function chatWithHistory(userInput: string): Promise<void> {
              // Append user input to the history.
              history.push({ role: "user", content: userInput });

              // Send the entire history to the chat method.
              const response: ChatResponseResult = await fi.chat({
                messages: history,
                model: 'meta/llama3.2-1b/instruct-fp16',
                stream: true,
                onStreamEvent: (event: StreamEvent) => console.log(event.chunk)
              });

              if (response.ok) {
                // Append the assistant's response to the history.
                history.push(response.message);
                console.log("Assistant's full response:", response.message.content);
              } else {
                console.error("Chat error:", response.failure.description);
              }
            }

            async function main() {
              chatWithHistory("Why is the sky blue?");
            }

            await main().then().catch();

    .. tab-item:: JavaScript
        :sync: js

        .. code-block:: js

            import { FlowerIntelligence } from '@flwr/flwr';

            // Access the singleton instance
            const fi = FlowerIntelligence.instance;

            // Initialize history with a system message.
            const history = [
              { role: "system", content: "You are a friendly assistant that loves using emojis." }
            ];

            // Function to chat while preserving conversation history.
            async function chatWithHistory(userInput) {
              // Append user input to the history.
              history.push({ role: "user", content: userInput });

              // Send the entire history to the chat method.
              const response = await fi.chat({
                messages: history,
                model: 'meta/llama3.2-1b/instruct-fp16',
                stream: true,
                onStreamEvent: (event) => console.log(event.chunk)
              });

              if (response.ok) {
                // Append the assistant's response to the history.
                history.push(response.message);
                console.log("Assistant's full response:", response.message.content);
              } else {
                console.error("Chat error:", response.failure.description);
              }
            }

            async function main() {
              chatWithHistory("Why is the sky blue?");
            }

            await main().then().catch();

.. note::
   Checkout out full examples over on `GitHub <https://github.com/adap/flower/tree/main/intelligence/ts/examples>`_ for more information!

Flower Confidential Remote Compute
----------------------------------

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

            import { ChatResponseResult, FlowerIntelligence, type StreamEvent } from '@flwr/flwr';

            // Access the singleton instance
            const fi: FlowerIntelligence = FlowerIntelligence.instance;

            // Enable remote processing and provide your API key
            fi.remoteHandoff = true;
            fi.apiKey = "YOUR_API_KEY";

            async function main() {
              const response: ChatResponseResult = await fi.chat({
                messages: [
                  { role: "system", content: "You are a friendly assistant that loves using emojies." }
                  { role: "user", content: "Why is the sky blue?" }
                ],
                model: 'meta/llama3.2-1b/instruct-fp16',
                stream: true,
                onStreamEvent: (event: StreamEvent) => console.log(event.chunk)
              });

              if (!response.ok) {
                console.error(`${response.failure.code}: ${response.failure.description}`);
              } else {
                console.log('Full response:', response.message.content);
              }
            }

            await main().then().catch();

    .. tab-item:: JavaScript
        :sync: js

        .. code-block:: js

            import { FlowerIntelligence } from '@flwr/flwr';

            // Access the singleton instance
            const fi = FlowerIntelligence.instance;

            // Enable remote processing and provide your API key
            fi.remoteHandoff = true;
            fi.apiKey = "YOUR_API_KEY";

            async function main() {
              const response = await fi.chat({
                messages: [
                  { role: "system", content: "You are a friendly assistant that loves using emojies." }
                  { role: "user", content: "Why is the sky blue?" }
                ],
                model: 'meta/llama3.2-1b/instruct-fp16',
                stream: true,
                onStreamEvent: (event) => console.log(event.chunk)
              });

              if (!response.ok) {
                console.error(`${response.failure.code}: ${response.failure.description}`);
              } else {
                console.log(response.message.content);
              }
            }

            await main().then().catch();

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

