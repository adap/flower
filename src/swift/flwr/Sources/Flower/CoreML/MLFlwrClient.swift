//
//  MLFlwrClient.swift
//  
//
//  Created by Daniel Nugraha on 18.01.23.
//

import Foundation
import NIOCore
import NIOPosix
import CoreML
import os

public enum MLTask {
    case train
    case test
}

@available(iOS 14.0, *)
public class MLFlwrClient: Client {
    private var eventLoopGroup: EventLoopGroup?
    
    private var parameters: MLParameter
    private var dataLoader: MLDataLoader
    
    private var compiledModelUrl: URL
    private var tempModelUrl: URL
    
    private let log = Logger(subsystem: Bundle.main.bundleIdentifier ?? "flwr.Flower",
                                    category: String(describing: MLFlwrClient.self))
    
    /// Inits the implementation of the Client protocol.
    ///
    /// - Parameters:
    ///   - layerWrappers: A MLLayerWrapper struct that contains layer information.
    ///   - dataLoader: A MLDataLoader struct that contains train- and testdata batches.
    ///   - compiledModelUrl: An URL specifying the location or path of the compiled model.
    public init(layerWrappers: [MLLayerWrapper], dataLoader: MLDataLoader, compiledModelUrl: URL) {
        self.parameters = MLParameter(layerWrappers: layerWrappers)
        self.eventLoopGroup = MultiThreadedEventLoopGroup(numberOfThreads: 1)
        self.dataLoader = dataLoader
        self.compiledModelUrl = compiledModelUrl
        
        let modelFileName = compiledModelUrl.deletingPathExtension().lastPathComponent
        self.tempModelUrl = appDirectory.appendingPathComponent("temp\(modelFileName).mlmodelc")
    }
    
    private func initGroup() {
        if eventLoopGroup == nil {
            self.eventLoopGroup = MultiThreadedEventLoopGroup(numberOfThreads: 1)
        }
    }
    
    /// Parses the parameters from the local model and returns them as GetParametersRes struct
    ///
    /// - Returns: Parameters from the local model
    public func getParameters() -> GetParametersRes {
        let parameters = parameters.weightsToParameters()
        let status = Status(code: .ok, message: String())
        
        return GetParametersRes(parameters: parameters, status: status)
    }
    
    /// Calls the routine to fit the local model
    ///
    /// - Returns: The result from the local training, e.g., updated parameters
    public func fit(ins: FitIns) -> FitRes {
        let status = Status(code: .ok, message: String())
        let result = runMLTask(configuration: parameters.parametersToWeights(parameters: ins.parameters), task: .train)
        let parameters = parameters.weightsToParameters()
        
        return FitRes(parameters: parameters, numExamples: result.numSamples, status: status)
    }
    
    /// Calls the routine to evaluate the local model
    ///
    /// - Returns: The result from the evaluation, e.g., loss
    public func evaluate(ins: EvaluateIns) -> EvaluateRes {
        let status = Status(code: .ok, message: String())
        let result = runMLTask(configuration: parameters.parametersToWeights(parameters: ins.parameters), task: .test)
        
        return EvaluateRes(loss: Float(result.loss), numExamples: result.numSamples, status: status)
    }
    
    private func runMLTask(configuration: MLModelConfiguration, task: MLTask) -> MLResult {
        initGroup()
        let result: MLResult?
        let promise = self.eventLoopGroup?.next().makePromise(of: MLResult.self)
        let dataset: MLBatchProvider
        
        switch task {
        case .train:
            dataset = self.dataLoader.trainBatchProvider
        case .test:
            let epochs = MLParameterKey.epochs
            configuration.parameters = [epochs:1]
            dataset = self.dataLoader.testBatchProvider
        }
        
        let progressHandler: (MLUpdateContext) -> Void = { contextProgress in
            let loss = String(format: "%.4f", contextProgress.metrics[.lossValue] as! Double)
            switch task {
            case .train:
                self.log.info("Epoch \(contextProgress.metrics[.epochIndex] as! Int + 1) finished with loss \(loss)")
            case .test:
                self.log.info("Evaluate finished with loss \(loss)")
            }
        }
        
        let completionHandler: (MLUpdateContext) -> Void = { finalContext in
            if task == .train {
                self.parameters.updateLayerWrappers(context: finalContext)
                self.saveModel(finalContext)
            }
            
            let loss = finalContext.metrics[.lossValue] as! Double
            let result = MLResult(loss: loss, numSamples: dataset.count, accuracy: (1.0 - loss) * 100)
            promise?.succeed(result)
        }
        
        let progressHandlers = MLUpdateProgressHandlers(
            forEvents: [.epochEnd],
            progressHandler: progressHandler,
            completionHandler: completionHandler
        )
        
        do {
            let updateTask = try MLUpdateTask(forModelAt: compiledModelUrl,
                                              trainingData: dataset,
                                              configuration: configuration,
                                              progressHandlers: progressHandlers)
            updateTask.resume()
            
            result = try promise?.futureResult.wait()
        } catch {
            result = nil
            log.error("\(error)")
        }
        
        return result ?? MLResult(loss: 1, numSamples: 0, accuracy: 0)
    }
    
    /// Closes the initiated group of event-loop
    public func closeEventLoopGroup() {
        do {
            try self.eventLoopGroup?.syncShutdownGracefully()
            self.eventLoopGroup = nil
        } catch {
            log.error("\(error)")
        }
    }
    
    private func saveModel(_ updateContext: MLUpdateContext) {
        let updatedModel = updateContext.model
        let fileManager = FileManager.default
        do {
            try fileManager.createDirectory(at: tempModelUrl, withIntermediateDirectories: true, attributes: nil)
            try updatedModel.write(to: tempModelUrl)
            _ = try fileManager.replaceItemAt(compiledModelUrl, withItemAt: tempModelUrl)
        } catch let error {
            log.error("Could not save updated model to the file system: \(error.localizedDescription)")
        }
    }
}
