Creating New Messages
=====================

This is a simple guide for creating a new type of message between the server and clients in Flower.

Let's suppose we have the following example functions in :code:`server.py` and :code:`numpy_client.py`...

Server's side:

.. code-block:: python

    def example_request(self, client: ClientProxy) -> Tuple[str, int]:
        question = "Could you find the sum of the list, Bob?"
        l = [1, 2, 3]
        return client.request(question, l)

Client's side:

.. code-block:: python

    def example_response(self, question: str, l: List[int]) -> Tuple[str, int]:
        response = "Here you go Alice!"
        answer = sum(question)
        return response, answer

Let's now see what we need to implement in order to get this simple function between the server and client to work!


Message Types for Protocol Buffers
----------------------------------

The first thing we need to do is to define a message type for the RPC system in :code:`transport.proto`.
Note that we have to do it for both the request and response messages. For more details on the syntax of proto3, please see the  `official documentation <https://protobuf.dev/programming-guides/proto3/>`_.

Within the :code:`ServerMessage` block:

.. code-block:: proto

    message ExampleIns{
        string question=1;
        repeated int64 l=2;
    }
    oneof msg {
        ReconnectIns reconnect_ins = 1;
        GetPropertiesIns get_properties_ins = 2;
        GetParametersIns get_parameters_ins = 3;
        FitIns fit_ins = 4;
        EvaluateIns evaluate_ins = 5;
        ExampleIns example_ins = 6;
    }

Within the ClientMessage block:

.. code-block:: proto

    message ExampleRes{
        string response = 1;
        int64 answer = 2;
    }

    oneof msg {
        DisconnectRes disconnect_res = 1;
        GetPropertiesRes get_properties_res = 2;
        GetParametersRes get_parameters_res = 3;
        FitRes fit_res = 4;
        EvaluateRes evaluate_res = 5;
        ExampleRes examples_res = 6;
    }

Make sure to also add a field of the newly created message type in :code:`oneof msg`.

Once that is done, we will compile the file with:

.. code-block:: shell

  $ python -m flwr_tool.protoc

If it compiles successfully, you should see the following message:

.. code-block:: shell

  Writing mypy to flwr/proto/transport_pb2.pyi
  Writing mypy to flwr/proto/transport_pb2_grpc.pyi


Serialization and Deserialization Functions
--------------------------------------------

Our next step is to add functions to serialize and deserialize Python datatypes to or from our defined RPC message types. You should add these functions in :code:`serde.py`.

The four functions:

.. code-block:: python

    def example_msg_to_proto(question: str, l: List[int]) -> ServerMessage.ExampleIns:
        return ServerMessage.ExampleIns(question=question, l=l)


    def example_msg_from_proto(msg: ServerMessage.ExampleIns) -> Tuple[str, List[int]]:
        return msg.question, msg.l


    def example_res_to_proto(response: str, answer: int) -> ClientMessage.ExampleRes:
        return ClientMessage.ExampleRes(response=response, answer=answer)


    def example_res_from_proto(res: ClientMessage.ExampleRes) -> Tuple[str, int]:
        return res.response, res.answer


Sending the Message from the Server
-----------------------------------

Now write the request function in your Client Proxy class (e.g., :code:`grpc_client_proxy.py`) using the serde functions you just created:

.. code-block:: python

    def request(self, question: str, l: List[int]) -> Tuple[str, int]:
        request_msg = serde.example_msg_to_proto(question, l)
        client_msg: ClientMessage = self.bridge.request(
            ServerMessage(example_ins=request_msg)
        )
        response, answer = serde.example_res_from_proto(client_msg.examples_res)
        return response, answer


Receiving the Message by the Client
-----------------------------------

Last step! Modify the code in :code:`message_handler.py` to check the field of your message and call the :code:`example_response` function. Remember to use the serde functions!

Within the handle function:

.. code-block:: python

    if server_msg.HasField("example_ins"):
        return _example_response(client, server_msg.example_ins), 0, True

And add a new function:

.. code-block:: python

    def _example_response(client: Client, msg: ServerMessage.ExampleIns) -> ClientMessage:
        question,l = serde.evaluate_ins_from_proto(msg)
        response, answer = client.example_response(question,l)
        example_res = serde.example_res_to_proto(response,answer)
        return ClientMessage(examples_res=example_res)

Hopefully, when you run your program you will get the intended result!

.. code-block:: shell

  ('Here you go Alice!', 6)
