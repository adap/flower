# ``flwr``

Seamlessly integrate Flower federated learning framework into your existing machine learning project.

## Overview

Flower Swift client SDK provides tools and functionalities for federating your machine learning project easily using Flower. The framework provides protocol to create a Flower ``Client``. You can create your own custom Flower clients by conforming to the provided protocol.

You can connect to a Flower server using GRPC by instantiating ``FlwrGRPC`` and providing a correct hostname and port. To start a gRPC connection, you can call the function ``FlwrGRPC/startFlwrGRPC(client:)`` and provide a ``Client`` as its argument.


## Topics

### API reference

- ``Client``
- ``FlwrGRPC``

### Tutorial

- <doc:How-to-use-Flower>
- <doc:Flower-with-CoreML>
- <doc:Configure-Serialization>
