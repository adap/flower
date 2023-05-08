//
//  File.swift
//  
//
//  Created by Daniel Nugraha on 20.01.23.
//

import Foundation
import CoreML
import os

public struct MLDataLoader {
    public let trainBatchProvider: MLBatchProvider
    public let testBatchProvider: MLBatchProvider
    
    public init(trainBatchProvider: MLBatchProvider, testBatchProvider: MLBatchProvider) {
        self.trainBatchProvider = trainBatchProvider
        self.testBatchProvider = testBatchProvider
    }
}

public struct MLLayerWrapper {
    let shape: [Int16]
    let name: String
    var weights: [Float]
    let isUpdatable: Bool
    
    public init(shape: [Int16], name: String, weights: [Float], isUpdatable: Bool) {
        self.shape = shape
        self.name = name
        self.weights = weights
        self.isUpdatable = isUpdatable
    }
}

struct MLResult {
    let loss: Double
    let numSamples: Int
    let accuracy: Double
}

@available(iOS 14.0, *)
public class MLParameter {
    private var parameterConverter = ParameterConverter.shared
    
    var layerWrappers: [MLLayerWrapper]
    private let log = Logger(subsystem: Bundle.main.bundleIdentifier ?? "flwr.Flower",
                                    category: String(describing: MLParameter.self))
    
    /// Inits MLParameter class that contains information about the model parameters and implements routines for their update and transformation.
    ///
    /// - Parameters:
    ///   - layerWrappers: Information about the layer provided with primitive data types.
    public init(layerWrappers: [MLLayerWrapper]) {
        self.layerWrappers = layerWrappers
    }
    
    /// Converts the Parameters structure to MLModelConfiguration to interface with CoreML.
    ///
    /// - Parameters:
    ///   - parameters: The parameters of the model passed as Parameters struct.
    /// - Returns: Specification of the machine learning model configuration in the CoreML structure.
    public func parametersToWeights(parameters: Parameters) -> MLModelConfiguration {
        let config = MLModelConfiguration()
        
        guard parameters.tensors.count == self.layerWrappers.count else {
            log.info("parameters received is not valid")
            return config
        }
        
        for (index, data) in parameters.tensors.enumerated() {
            let expectedNumberOfElements = layerWrappers[index].shape.map({Int($0)}).reduce(1, *)
            if let weightsArray = parameterConverter.dataToArray(data: data) {
                guard weightsArray.count == expectedNumberOfElements else {
                    log.info("array received has wrong number of elements")
                    continue
                }
                
                layerWrappers[index].weights = weightsArray
                if layerWrappers[index].isUpdatable {
                    if let weightsMultiArray = parameterConverter.dataToMultiArray(data: data) {
                        let weightsShape = weightsMultiArray.shape.map { Int16(truncating: $0) }
                        guard weightsShape == layerWrappers[index].shape else {
                            log.info("shape not the same")
                            continue
                        }
                        let paramKey = MLParameterKey.weights.scoped(to: layerWrappers[index].name)
                        config.parameters?[paramKey] = weightsMultiArray
                    }
                }
            }
        }
        
        return config
    }
    
    /// Returns the weights of the current layer wrapper in parameter format
    ///
    /// - Returns: The weights of the current layer wrapper in parameter format
    public func weightsToParameters() -> Parameters {
        let dataArray = layerWrappers.compactMap { parameterConverter.arrayToData(array: $0.weights, shape: $0.shape) }
        if dataArray.count != layerWrappers.count {
            log.info("dataArray size != layerWrappers size")
        }
        return Parameters(tensors: dataArray, tensorType: "ndarray")
    }
    
    /// Updates the layers given the CoreML update context
    ///
    /// - Parameters:
    ///   - context: The context of the update procedure of the CoreML model.
    public func updateLayerWrappers(context: MLUpdateContext) {
        for (index, layer) in self.layerWrappers.enumerated() {
            if layer.isUpdatable {
                let paramKey = MLParameterKey.weights.scoped(to: layer.name)
                if let weightsMultiArray = try? context.model.parameterValue(for: paramKey) as? MLMultiArray {
                    let weightsShape = Array(weightsMultiArray.shape.map({ Int16(truncating: $0) }).drop(while: { $0 < 2 }))
                    guard weightsShape == layer.shape else {
                        log.info("shape \(weightsShape) is not the same as \(layer.shape)")
                        continue
                    }
                    
                    if let pointer = try? UnsafeBufferPointer<Float>(weightsMultiArray) {
                        let array = pointer.compactMap{$0}
                        self.layerWrappers[index].weights = array
                    }
                }
            }
        }
    }
}
