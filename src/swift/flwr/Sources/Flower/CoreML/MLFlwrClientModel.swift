//
//  File.swift
//  
//
//  Created by Daniel Nugraha on 20.01.23.
//

import Foundation
import CoreML

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
    
    public init(layerWrappers: [MLLayerWrapper]) {
        self.layerWrappers = layerWrappers
    }
    
    public func parametersToWeights(parameters: Parameters) -> MLModelConfiguration {
        let config = MLModelConfiguration()
        
        guard parameters.tensors.count == self.layerWrappers.count else {
            print("parameters received is not valid")
            return config
        }
        
        for (index, data) in parameters.tensors.enumerated() {
            let expectedNumberOfElements = layerWrappers[index].shape.map({Int($0)}).reduce(1, *)
            if let weightsArray = parameterConverter.dataToArray(data: data) {
                guard weightsArray.count == expectedNumberOfElements else {
                    print("array received has wrong number of elements")
                    continue
                }
                
                layerWrappers[index].weights = weightsArray
                if layerWrappers[index].isUpdatable {
                    if let weightsMultiArray = parameterConverter.dataToMultiArray(data: data) {
                        let weightsShape = weightsMultiArray.shape.map { Int16(truncating: $0) }
                        guard weightsShape == layerWrappers[index].shape else {
                            print("shape not the same")
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
    
    public func weightsToParameters() -> Parameters {
        let dataArray = layerWrappers.compactMap { parameterConverter.arrayToData(array: $0.weights, shape: $0.shape) }
        if dataArray.count != layerWrappers.count {
            print("dataArray size != layerWrappers size")
        }
        return Parameters(tensors: dataArray, tensorType: "ndarray")
    }
    
    public func updateLayerWrappers(context: MLUpdateContext) {
        for (index, layer) in self.layerWrappers.enumerated() {
            //print("checking if layer \(layer.name) updatable")
            if layer.isUpdatable {
                let paramKey = MLParameterKey.weights.scoped(to: layer.name)
                //print("retrieving parameters for layer \(layer.name)")
                if let weightsMultiArray = try? context.model.parameterValue(for: paramKey) as? MLMultiArray {
                    let weightsShape = Array(weightsMultiArray.shape.map({ Int16(truncating: $0) }).drop(while: { $0 < 2 }))
                    guard weightsShape == layer.shape else {
                        print("shape \(weightsShape) is not the same as \(layer.shape)")
                        continue
                    }
                    
                    if let pointer = try? UnsafeBufferPointer<Float>(weightsMultiArray) {
                        //print("updating weights for layer \(layer.name)")
                        let array = pointer.compactMap{$0}
                        //print("layer weight array size = \(layer.weights.count)")
                        //print("array size = \(array.count)")
                        self.layerWrappers[index].weights = array
                    }
                }
            }
        }
    }
}
