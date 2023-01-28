# flwr iOS SDK

This package implements the iOS SDK for Flower - a friendly federated learning framework https://flower.dev/

## Installation

You can either download the Flower project and integrate the package manually, or by adding the following to the package dependencies list.

```
.package(url: "TBD", .branch("main"))
```

## Usage

A comprehensive example is available in: ```examples/iOS/``` To give information about the usage structurally: 

```
import flwr

let mlFlwrClient = MLFlwrClient(layerWrappers: layerWrappers, dataLoader: dataLoader, compiledModelUrl: compiledModelUrl)
let flwrGRPC = FlwrGRPC(serverHost: hostname, serverPort: port)
startFlwrGRPC(client: mlFlwrClient) {
    // completion handler
    print("Federated learning completed")
}
...
```

## License

Apache-2.0 license
