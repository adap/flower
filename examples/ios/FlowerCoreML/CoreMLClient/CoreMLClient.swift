//
//  CoreMLClient.swift
//  FlowerSDK
//
//  Created by Daniel Nugraha on 04.03.22.
//

import Foundation
import Vision
import CoreML
import SwiftUI
import Flower


class CoreMLClient: Client {
    
    var model : MLModel?
    
    private var parameterConverter = ParameterConverter()
    private let dataLoader: DataLoader
    
    var layerWrappers: [MLLayerWrapper] = []
    
    /// The location of the app's Application Support directory for the user.
    private static let appDirectory = FileManager.default.urls(for: .applicationSupportDirectory,
                                                               in: .userDomainMask).first!
    /// The default model's file URL.
    // need cleanup
    private let defaultModelURL: URL
    /// The permanent location of the updated model.
    private var updatedModelURL = appDirectory.appendingPathComponent("MNIST_Model.mlmodelc")
    /// The temporary location of the updated model.
    private var tempUpdatedModelURL = appDirectory.appendingPathComponent("MNIST_Model_tmp.mlmodelc")
    
    init(modelUrl url: URL, dataLoader: DataLoader) {
        self.dataLoader = dataLoader
        let compiledModelUrl = try! MLModel.compileModel(at: url)
        let modelFileName = url.deletingPathExtension().lastPathComponent
        updatedModelURL = CoreMLClient.appDirectory.appendingPathComponent("\(modelFileName).mlmodelc")
        self.defaultModelURL = compiledModelUrl
        self.updatedModelURL = compiledModelUrl
        self.model = loadModel(url: compiledModelUrl)
        if let modelInspect = try? MLModelInspect(serializedData: Data(contentsOf: url)) {
            self.layerWrappers = modelInspect.getLayerWrappers()
        }
        
    }
    
    private func convertToDataTensor(data: Any?) -> Data? {
        try? JSONSerialization.data(withJSONObject: data as Any)
    }
    
    func getParameters() -> ParametersRes {
//        print(layerWrappers.map{ $0.shape })
        let parameters = weightsToParameters()
        return ParametersRes(parameters: parameters)
    }
    
    func fit(ins: FitIns) -> FitRes {
        parametersToWeights(parameters: ins.parameters)
        var trainingResult: MLResult? = nil
        print("Starting local training")
        train(modelConfig: parametersToWeights(parameters: ins.parameters)) { result in
            trainingResult = result
        }
        while true {
            if let result = trainingResult {
                let parameters = weightsToParameters()
                return FitRes(parameters: parameters, numExamples: result.numSamples)
            }
        }
    }
    
    func evaluate(ins: EvaluateIns) -> EvaluateRes {
        var evaluateResult: MLResult? = nil
        test(modelConfig: parametersToWeights(parameters: ins.parameters)) { result in
            evaluateResult = result
        }
        while true {
            if let result = evaluateResult {
                return EvaluateRes(loss: Float(result.loss), numExamples: result.numSamples)
            }
        }
    }
    
    func parametersToWeights(parameters: Parameters) -> MLModelConfiguration {
        parameterConverter = ParameterConverter()
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
    
    func weightsToParameters() -> Parameters {
        parameterConverter = ParameterConverter()
        let dataArray = layerWrappers.compactMap { parameterConverter.arrayToData(array: $0.weights, shape: $0.shape) }
        if dataArray.count != layerWrappers.count {
            print("dataArray size != layerWrappers size")
        }
        return Parameters(tensors: dataArray, tensorType: "ndarray")
    }
    
    func train(modelConfig: MLModelConfiguration, result: @escaping(MLResult) -> Void) {
        /// The URL of the currently active Model
        let usingUpdatedModel = model != nil
        let currentModelURL = usingUpdatedModel ? updatedModelURL : defaultModelURL
        
        let fileManager = FileManager.default
//        if fileManager.fileExists(atPath: currentModelURL.path) {
//            print(currentModelURL)
//        }
        
        var loss = 0.0
        let trainingData = dataLoader.trainBatchProvider
        let progressHandlers = MLUpdateProgressHandlers(
            forEvents: [.epochEnd],
            /// check progress after epoch
            progressHandler: { contextProgress in
                loss = contextProgress.metrics[.lossValue] as! Double
                print("Epoch \(contextProgress.metrics[.epochIndex] as! Int + 1) finished with loss \(loss)")
            
         }) { (finalContext) in
             for (index, layer) in self.layerWrappers.enumerated() {
                 //print("checking if layer \(layer.name) updatable")
                 if layer.isUpdatable {
                     let paramKey = MLParameterKey.weights.scoped(to: layer.name)
                     //print("retrieving parameters for layer \(layer.name)")
                     if let weightsMultiArray = try? finalContext.model.parameterValue(for: paramKey) as? MLMultiArray {
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
             
             let trainingResult = MLResult(loss: finalContext.metrics[.lossValue] as! Double, numSamples: trainingData.count, accuracy: (1.0 - loss) * 100)
             
             result(trainingResult)
             
             self.saveModel(finalContext)
             
             self.model = self.loadModel(url: self.updatedModelURL)
         }
        
        do {
            let updateTask = try MLUpdateTask(forModelAt: currentModelURL,
                                              trainingData: trainingData,
                                              configuration: modelConfig,
                                              progressHandlers: progressHandlers)
            updateTask.resume()
        } catch {
            print(error)
        }
    }
    
    public func localTrain(configuration: MLModelConfiguration, progressHandler: @escaping (MLUpdateContext) -> Void, completionHandler: @escaping (MLUpdateContext) -> Void) {
        let usingUpdatedModel = model != nil
        let currentModelURL = usingUpdatedModel ? updatedModelURL : defaultModelURL
        let trainingData = dataLoader.trainBatchProvider
        
        let completionHandlerWithSave = { (finalContext: MLUpdateContext) in
            completionHandler(finalContext)
            self.saveModel(finalContext)
            self.model = self.loadModel(url: self.updatedModelURL)
        }
        
        let progressHandlers =  MLUpdateProgressHandlers(
            forEvents: [.trainingBegin, .epochEnd],
            progressHandler: progressHandler,
            completionHandler: completionHandlerWithSave)
        
        do {
            let updateTask = try MLUpdateTask(forModelAt: currentModelURL,
                                              trainingData: trainingData,
                                              configuration: configuration,
                                              progressHandlers: progressHandlers)
            updateTask.resume()
        } catch {
            print(error)
        }
    }
    
    public func localTest(configuration: MLModelConfiguration, progressHandlers: MLUpdateProgressHandlers) {
        let usingUpdatedModel = model != nil
        let currentModelURL = usingUpdatedModel ? updatedModelURL : defaultModelURL
        let trainingData = dataLoader.testBatchProvider
        do {
            let updateTask = try MLUpdateTask(forModelAt: currentModelURL,
                                              trainingData: trainingData,
                                              configuration: configuration,
                                              progressHandlers: progressHandlers)
            updateTask.resume()
        } catch {
            print(error)
        }
    }
    
    func test(modelConfig: MLModelConfiguration, result: @escaping(MLResult) -> Void) {
        /// The URL of the currently active Model
        let usingUpdatedModel = model != nil
        let currentModelURL = usingUpdatedModel ? updatedModelURL : defaultModelURL
        let epochs = MLParameterKey.epochs
        modelConfig.parameters = [epochs:1]
        var loss = 0.0
        let trainingData = dataLoader.testBatchProvider
        let progressHandlers = MLUpdateProgressHandlers(
            forEvents: [.epochEnd],
            /// check progress after epoch
            progressHandler: { (contextProgress) in
                loss = contextProgress.metrics[.lossValue] as! Double
                print("Evaluate finished with loss \(loss)")
                
            }
        ) { finalContext in
            let evaluateResult = MLResult(loss: finalContext.metrics[.lossValue] as! Double, numSamples: trainingData.count, accuracy: (1.0 - loss) * 100)
            result(evaluateResult)
        }
        
        do {
            let updateTask = try MLUpdateTask(forModelAt: currentModelURL,
                                          trainingData: trainingData,
                                          configuration: modelConfig,
                                          progressHandlers: progressHandlers)
            updateTask.resume()
        } catch {
            print(error)
        }
    }
    
    // MARK: - Private Type Helper Methods
    /// Saves the model in the given Update Context provided by an MLUpdateTask.
    /// - Parameter updateContext: The context from the Update Task that contains the updated model.
    /// - Tag: SaveModel
    private func saveModel(_ updateContext: MLUpdateContext) {
        let updatedModel = updateContext.model
        let fileManager = FileManager.default
        do {
            // Create a directory for the updated model.
            try fileManager.createDirectory(at: tempUpdatedModelURL,
                                            withIntermediateDirectories: true,
                                            attributes: nil)
            
            // Save the updated model to temporary filename.
            try updatedModel.write(to: tempUpdatedModelURL)
            
            // Replace any previously updated model with this one.
            _ = try fileManager.replaceItemAt(updatedModelURL,
                                              withItemAt: tempUpdatedModelURL)
            
            //print("Updated model saved to:\n\t\(updatedModelURL)")
        } catch let error {
            print("Could not save updated model to the file system: \(error.localizedDescription)")
            return
        }
    }
    
    /// Load Model if available.
    /// - Tag: LoadModel
    private func loadModel(url: URL) -> MLModel? {
        guard FileManager.default.fileExists(atPath: url.path) else {
            // The updated model is not present at its designated path.
            return nil
        }
        do {
            // Create an instance of the model.
            return try MLModel(contentsOf: url)
        } catch {
            return nil
        }
    }
}
