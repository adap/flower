Flower Intelligence
===================

Flower Intelligence is a cross-platform inference library that let's user seamlessly
interact with Large-Language Models both locally and remotely in a secure and private
way. The library was created by the ``Flower Labs`` team that also created `Flower: A
Friendly Federated AI Framework <https://flower.ai>`_.

We currently provide SDKs for TypeScript/JavaScript and Swift.

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

    .. tab-item:: Swift
        :sync: swift

        .. code-block:: bash

          # You can add package dependency to your Xcode project via its UI.
          # Please refer to https://developer.apple.com/documentation/xcode/adding-package-dependencies-to-your-app.
          #
          # To add dependency to your Swift package, you can run the following command:
          swift package add-dependency "https://github.com/adap/flower.git"

Hello, Flower Intelligence!
---------------------------

Flower Intelligence is built around the Singleton design pattern, meaning you only need
to configure a single instance that can be reused throughout your project. This simple
setup helps you integrate powerful AI capabilities with minimal overhead.

.. tab-set::
    :sync-group: category

    .. tab-item:: TypeScript
        :sync: ts

        .. code-block:: ts

            import { ChatResponseResult, FlowerIntelligence } from '@flwr/flwr';

            // Access the singleton instance
            const fi: FlowerIntelligence = FlowerIntelligence.instance;

            async function main() {
              // Perform the inference
              const response: ChatResponseResult = await fi.chat("Why is the sky blue?");

              if (response.ok) {
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
              // Perform the inference
              const response = await fi.chat("Why is the sky blue?");

              console.log(response.message.content);
            }

            await main().then().catch();

    .. tab-item:: Swift
        :sync: swift

        .. code-block:: swift

            import FlowerIntelligence

            let fi = FlowerIntelligence.instance

            let result = await fi.chat("Why is the sky blue?")
            switch result {
            case .success(let message):
              print(message.content)
            case .failure(let error):
              print(error.localizedDescription)
            }

Specify the model
-----------------

By specifying a model in the chat options, you can easily switch between different AI
models available in the ecosystem. For a full list of supported models, please refer to
the :doc:`available models list <ref-models>`.

.. tab-set::
    :sync-group: category

    .. tab-item:: TypeScript
        :sync: ts

        .. code-block:: ts

            import { ChatResponseResult, FlowerIntelligence } from '@flwr/flwr';

            // Access the singleton instance
            const fi: FlowerIntelligence = FlowerIntelligence.instance;

            async function main() {
              // Perform the inference
              const response: ChatResponseResult = await fi.chat('Why is the sky blue?', {
                model: 'meta/llama3.2-1b/instruct-fp16',
              });

              if (response.ok) {
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
              // Perform the inference
              const response = await fi.chat('Why is the sky blue?', {
                model: 'meta/llama3.2-1b/instruct-fp16',
              });

              console.log(response.message.content);
            }

            await main().then().catch();

    .. tab-item:: Swift
        :sync: swift

        .. code-block:: swift

            import FlowerIntelligence

            let fi = FlowerIntelligence.instance

            let options = ChatOptions(model: "meta/llama3.2-1b/instruct-fp16")
            let result = await fi.chat("Why is the sky blue?", maybeOptions: options)

            switch result {
            case .success(let message):
                print(message.content)
            case .failure(let error):
                print(error.localizedDescription)
            }

Check for errors
----------------

Instead of throwing exceptions that might crash your application, Flower Intelligence
returns a response object that includes a dedicated :doc:`Failure
<ts-api-ref/interfaces/Failure>` property, enabling graceful error handling and improved
application stability.

.. tab-set::
    :sync-group: category

    .. tab-item:: TypeScript
        :sync: ts

        .. code-block:: ts

            import { ChatResponseResult, FlowerIntelligence } from '@flwr/flwr';

            // Access the singleton instance
            const fi: FlowerIntelligence = FlowerIntelligence.instance;

            async function main() {
              // Perform the inference
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
              // Perform the inference
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

    .. tab-item:: Swift
        :sync: swift

        .. code-block:: swift

            import FlowerIntelligence

            let fi = FlowerIntelligence.instance

            let options = ChatOptions(model: "meta/llama3.2-1b/instruct-fp16")
            let result = await fi.chat("Why is the sky blue?", maybeOptions: options)

            switch result {
            case .success(let message):
                print(message.content)
            case .failure(let error):
                print(error.localizedDescription)
            }

Stream Responses
----------------

By enabling the stream option and providing a callback function, you can watch the AI’s
response as it is being generated. This approach is ideal for interactive applications,
as it lets you process partial responses immediately before the full answer is
available. The callback function must accept an argument of type :doc:`StreamEvent
<ts-api-ref/interfaces/StreamEvent>`.

.. tab-set::
    :sync-group: category

    .. tab-item:: TypeScript
        :sync: ts

        .. code-block:: ts

            import { ChatResponseResult, FlowerIntelligence, type StreamEvent } from '@flwr/flwr';

            // Access the singleton instance
            const fi: FlowerIntelligence = FlowerIntelligence.instance;

            async function main() {
              // Perform the inference
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
              // Perform the inference
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

    .. tab-item:: Swift
        :sync: swift

        .. code-block:: swift

            import FlowerIntelligence

            let fi = FlowerIntelligence.instance

            let options = ChatOptions(
              model: "meta/llama3.2-1b/instruct-fp16",
              stream: true,
              onStreamEvent: { event in
                  print(event.chunk)
              }
            )

            let result = await fi.chat("Why is the sky blue?", maybeOptions: options)

            if case .failure(let error) = result {
                print(error.localizedDescription)
            }

Use Roles
---------

Instead of simply sending a single string, you can provide an array of :doc:`messages
<ts-api-ref/interfaces/Message>` with designated roles such as ``system`` and ``user``.
This allows you to define the behavior and context of the conversation more clearly,
ensuring that the assistant responds in a way that’s tailored to the scenario.

.. tab-set::
    :sync-group: category

    .. tab-item:: TypeScript
        :sync: ts

        .. code-block:: ts

            import { ChatResponseResult, FlowerIntelligence, type StreamEvent } from '@flwr/flwr';

            // Access the singleton instance
            const fi: FlowerIntelligence = FlowerIntelligence.instance;

            async function main() {
              // Perform the inference
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
              // Perform the inference
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

    .. tab-item:: Swift
        :sync: swift

        .. code-block:: swift

            import FlowerIntelligence

            let fi = FlowerIntelligence.instance

            let messages = [
              Message(role: "system", content: "You are a helpful assistant."),
              Message(role: "user", content: "Why is the sky blue?")
            ]

            let options = ChatOptions(model: "meta/llama3.2-1b/instruct-fp16")
            let result = await fi.chat(options: (messages, options))
            switch result {
            case .success(let message):
              print(message.content)
            case .failure(let error):
              print(error.localizedDescription)
            }

Handle history
--------------

In this example, the conversation history is maintained in an array that includes both
system and user messages. Each time a new message is sent, it is appended to the
history, ensuring that the assistant has access to the full dialogue context. This
method allows Flower Intelligence to generate responses that are informed by previous
interactions, resulting in a more coherent and dynamic conversation.

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

    .. tab-item:: Swift
        :sync: swift

        .. code-block:: swift

            import FlowerIntelligence

            let fi = FlowerIntelligence.instance

            // Initialize history with a system message
            var history: [Message] = [
                Message(role: "system", content: "You are a friendly assistant that loves using emojis.")
            ]

            // Function to chat while preserving conversation history
            func chatWithHistory(userInput: String) async {
                // Append user input to the history
                history.append(Message(role: "user", content: userInput))

                // Define chat options with streaming
                let options = ChatOptions(
                    model: "meta/llama3.2-1b/instruct-fp16",
                    stream: true,
                    onStreamEvent: { event in
                        print(event.chunk)
                    }
                )

                // Perform chat with full history
                let result = await fi.chat(options: (history, options))

                switch result {
                case .success(let response):
                    // Append assistant's response to history
                    history.append(response)
                    print("Assistant's full response:", response.content)
                case .failure(let error):
                    print(error.localizedDescription)
                }
            }

            // Start the conversation
            await chatWithHistory("Why is the sky blue?")

Pre-loading the model
---------------------

You might have noticed that the first time you run inference on a given model, you'll
have to wait longer for it to complete compared to the second time you call the model.
This is because the model first needs to be downloaded. This might be undesirable if you
have an app where users can click a button and expect a quick response from the model.
In this case, you might want to first let the user download the model (or download it on
the first start-up), so once they click on the inference button, the results are
consistently fast. This can be done using the :doc:`fetchModel
<ts-api-ref/classes/FlowerIntelligence>` method.

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
              // Download the model first
              await fi.fetchModel('meta/llama3.2-1b/instruct-fp16');
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
              await fi.fetchModel('meta/llama3.2-1b/instruct-fp16');
              chatWithHistory("Why is the sky blue?");
            }

            await main().then().catch();

If you want to follow the progress of the download, you can pass a callback function
that takes a :doc:`Progress <ts-api-ref/interfaces/Progress>` object as input:

.. tab-set::
    :sync-group: category

    .. tab-item:: TypeScript
        :sync: ts

        .. code-block:: ts

            import { FlowerIntelligence, Progress } from '@flwr/flwr';

            await fi.fetchModel('meta/llama3.2-1b/instruct-fp16', (progress: Progress) =>
              console.log(progress.percentage ?? '')
            );

    .. tab-item:: JavaScript
        :sync: js

        .. code-block:: js

            import { FlowerIntelligence } from '@flwr/flwr';

            await fi.fetchModel('meta/llama3.2-1b/instruct-fp16', (progress) =>
              console.log(progress.percentage ?? '')
            );

.. note::

    Checkout out full examples over on `GitHub
    <https://github.com/adap/flower/tree/main/intelligence/ts/examples>`_ for more
    information!

Flower Confidential Remote Compute
----------------------------------

.. warning::

    Flower Confidential Remote Compute is available in private beta. If you are
    interested in using Confidential Remote Compute, please apply for Early Access via
    the `Flower Intelligence Pilot Program <https://forms.gle/J8pFpMrsmek2VFKq8>`_.

Flower Intelligence prioritizes local inference, but also allows to privately handoff
the compute to the Flower Confidential Remote Compute service when local resources are
scarce. You can find more information on `flower.ai/intelligence
<https://flower.ai/intelligence>`_.

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

    .. tab-item:: Swift
        :sync: swift

        .. code-block:: swift

            import FlowerIntelligence

            let fi = FlowerIntelligence.instance
            fi.remoteHandoff = true
            fi.apiKey = "YOUR_API_KEY"

            let messages = [
              Message(role: "system", content: "You are a helpful assistant."),
              Message(role: "user", content: "Why is the sky blue?")
            ]

            let options = ChatOptions(
              model: "meta/llama3.2-1b/instruct-fp16",
              stream: true,
              onStreamEvent: { event in
                  print(event.chunk)
              }
            )

            let result = await fi.chat(options: (messages, options))

            if case .failure(let error) = result {
              print(error.localizedDescription)
            }

References
----------

Information-oriented API reference and other reference material.

.. toctree::
    :maxdepth: 1
    :glob:
    :caption: Reference docs

    ref-models
    examples
    ts-api-ref/index
    swift-api-ref/index

Contributor guides
------------------

If you are interested in contributing or playing around with the source code.

.. toctree::
    :maxdepth: 1
    :glob:
    :caption: Contributor docs

    contributor-how-to-build-from-source.rst

Join the Flower Community
-------------------------

The Flower Community is growing quickly - we're a friendly group of researchers,
engineers, students, professionals, academics, and other enthusiasts.

.. button-link:: https://flower.ai/join-slack
    :color: primary
    :shadow:

    Join us on Slack
