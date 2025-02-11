# Federate CoreML with Flower Tutorial

In this comprehensive tutorial we will learn to federate CoreML with Flower and enable federated learning on iOS devices.

## Overview

This tutorial is based on https://github.com/adap/flower/tree/main/examples/ios, but we break it down piece by piece to give more understanding about running federated learning on iOS devices with Flower.

### Layer wrapper

Notice that due to the CoreML black-box approach in running machine learning training, we can't easily federate our training using CoreML. One particular problem is that CoreML needs to give us the ability to get initial weights and their shape. We need to peek inside the CoreML model specification before federating our training to fix this.

```swift
private func getLayerWrapper(layer: CoreML_Specification_NeuralNetworkLayer) -> MLLayerWrapper? {
    if let layerType = layer.layer {
        switch layerType {
        case .convolution:
            let convolution = layer.convolution
            //shape definition = [outputChannels, kernelChannels, kernelHeight, kernelWidth]
            let shape = [Int16(convolution.outputChannels), Int16(convolution.kernelChannels), Int16(convolution.kernelSize[0]), Int16(convolution.kernelSize[1])]
            return MLLayerWrapper(shape: shape,
                                  name: layer.name,
                                  weights: convolution.weights.floatValue,
                                  isUpdatable: layer.isUpdatable)
        case .innerProduct:
            let innerProduct = layer.innerProduct
            //shape definition = [C_out, C_in].
            let shape = [Int16(innerProduct.outputChannels), Int16(innerProduct.inputChannels)]
            return MLLayerWrapper(shape: shape,
                                  name: layer.name,
                                  weights: innerProduct.weights.floatValue,
                                  isUpdatable: layer.isUpdatable)
        default:
            return nil
        }
    }
    return nil
}
```

The `getLayerWrapper()` function maps the CoreML model specification into LayerWrapper, we will use this throughout our federated learning process using Flower.

### Parameters to weights

`weightsToParameters()` and `parametersToWeights()` are responsible for serializing our weights to a format that the server can communicate and understand during federated learning. Check out our <doc:Configure-Serialization> tutorial to learn more about this.


### Run local machine learning pipeline

To update a CoreML model, you need to instantiate `MLUpdateTask`, this is basically what `runMLTask` does. In a nutshell, `runMLTask` differentiates test (evaluate) and train (fit). For the train, we update our `LayerWrapper` weights and save the new model by overwriting the old model with the new one, and for the test, we just get the loss value.

```swift
let completionHandler: (MLUpdateContext) -> Void = { finalContext in
    if task == .train {
        self.parameters.updateLayerWrappers(context: finalContext)
        self.saveModel(finalContext)
    }
    
    let loss = String(format: "%.20f", finalContext.metrics[.lossValue] as! Double)
    let result = MLResult(loss: Double(loss)!, numSamples: dataset.count, accuracy: (1.0 - Double(loss)!) * 100)
    promise?.succeed(result)
}
```

To instantiate MLUpdateTask you need four arguments: URL to the CoreML, training data, configuration, and progress handler. We do not cover how to provide training data here, so feel free to check out CoreML official documentation and our code example.

After creating our progress handlers to differentiate, test, and train, we can instantiate MLUpdateTask. To execute the CoreML update task, call the resume() function.

```swift
let updateTask = try MLUpdateTask(forModelAt: compiledModelUrl,
                                  trainingData: dataset,
                                  configuration: configuration,
                                  progressHandlers: progressHandlers)
updateTask.resume()
```

### Flower client

Let's build our Flower client after we get our local training pipeline sorted. To conform to Flower client, we need to implement four functions: `getParameters()`, `getProperties()`, `fit()`, and `evaluate()`. In the following sections, we provide a detailed description of each function.

```swift
public protocol Client {
    
    /// Return the current local model parameters.
    func getParameters() -> GetParametersRes
    
    /// Return set of client properties.
    func getProperties(ins: GetPropertiesIns) -> GetPropertiesRes
    
    /// Refine the provided parameters using the locally held dataset.
    func fit(ins: FitIns) -> FitRes
    
    /// Evaluate the provided parameters using the locally held dataset.
    func evaluate(ins: EvaluateIns) -> EvaluateRes
}
```

### Get Parameters

`getParameters()` is a function that sends the weights to the server. It expects to return `GetParametersRes` object.

```swift
public func getParameters() -> GetParametersRes {
    parameters.initializeParameters()
    let parameters = parameters.weightsToParameters()
    let status = Status(code: .ok, message: String())
    
    return GetParametersRes(parameters: parameters, status: status)
}
```

### Fit

`fit()` is a function that runs the local machine learning training. It expects to return `FitRes` object. In this case, we call our `runMLTask()` function and add `.train` as its arguments and convert the result with the weights as `FitRes`.

```swift
public func fit(ins: FitIns) -> FitRes {
    let status = Status(code: .ok, message: String())
    let result = runMLTask(configuration: parameters.parametersToWeights(parameters: ins.parameters), task: .train)
    let parameters = parameters.weightsToParameters()
    
    return FitRes(parameters: parameters, numExamples: result.numSamples, status: status)
}
```

### Evaluate

`evaluate()` is a function that runs the local machine learning evaluation. It expects to return the `EvaluateRes` object. In this case, we call our `runMLTask()` function, add `.test` as its argument, and convert the result to `EvaluateRes`.

```swift
public func evaluate(ins: EvaluateIns) -> EvaluateRes {
    let status = Status(code: .ok, message: String())
    let result = runMLTask(configuration: parameters.parametersToWeights(parameters: ins.parameters), task: .test)
    
    return EvaluateRes(loss: Float(result.loss), numExamples: result.numSamples, status: status)
}
```

### Run Flower client

Great, now you have your own Flower client. To run your Flower client and enable communicating with the server, follow the steps below:

```swift
let flwrGRPC = FlwrGRPC(serverHost: hostname, serverPort: port)
flwrGRPC.startFlwrGRPC(client: mlFlwrClient)
```

Congratulations, now your Flower client will run and communicate with the server to do federated learning.


### Related documentation
- iOS code example: [https://github.com/adap/flower/tree/main/examples/ios](https://github.com/adap/flower/tree/main/examples/ios)
- Tutorial video: [Federating iOS](https://www.youtube.com/watch?v=5v8hJhKDv20&pp=ygUOZmVkZXJhdGluZyBpb3M%3D)

### API reference
- Client: <doc:Client>
- FlwrGRPC: <doc:FlwrGRPC>
