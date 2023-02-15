//
//  FLiOSModel.swift
//  FLiOS
//
//  Created by Daniel Nugraha on 18.01.23.
//

import Foundation
import CoreML
import flwr

public class FLiOSModel: ObservableObject {
    @Published public var scenarioSelection = Constants.ScenarioTypes.MNIST
    private var trainingBatchProvider: MLBatchProvider?
    @Published public var trainingBatchStatus = PreparationStatus.notPrepared
    let scenarios = Constants.ScenarioTypes.allCases
    
    private var testBatchProvider: MLBatchProvider?
    @Published public var testBatchStatus = PreparationStatus.notPrepared
    
    private var localClient: LocalClient?
    @Published public var localClientStatus = PreparationStatus.notPrepared
    @Published public var epoch: Int = 5
    @Published public var localTrainingStatus = TaskStatus.idle
    @Published public var localTestStatus = TaskStatus.idle
    
    private var mlFlwrClient: MLFlwrClient?
    @Published public var mlFlwrClientStatus = PreparationStatus.notPrepared
    
    private var flwrGRPC: FlwrGRPC?
    @Published public var hostname: String = "localhost"
    @Published public var port: Int = 8080
    @Published public var federatedServerStatus = TaskStatus.idle
    
    public var benchmarkSuite = BenchmarkSuite.shared
    
    public func prepareTrainDataset() {
        trainingBatchStatus = .preparing(count: 0)
        self.benchmarkSuite.takeActionSnapshot(snapshot: ActionSnaptshot(action: "preparing train dataset"))
        DispatchQueue.global(qos: .userInitiated).async {
            let batchProvider = MNISTDataLoader.trainBatchProvider { count in
                DispatchQueue.main.async {
                    self.trainingBatchStatus = .preparing(count: count)
                }
            }
            DispatchQueue.main.async {
                self.trainingBatchProvider = batchProvider
                self.trainingBatchStatus = .ready
                self.benchmarkSuite.takeActionSnapshot(snapshot: ActionSnaptshot(action: "finished preparing train dataset"))
            }
        }
    }
    
    public func prepareTestDataset() {
        testBatchStatus = .preparing(count: 0)
        self.benchmarkSuite.takeActionSnapshot(snapshot: ActionSnaptshot(action: "preparing test dataset"))
        DispatchQueue.global(qos: .userInitiated).async {
            let batchProvider = MNISTDataLoader.testBatchProvider { count in
                DispatchQueue.main.async {
                    self.testBatchStatus = .preparing(count: count)
                }
            }
            DispatchQueue.main.async {
                self.testBatchProvider = batchProvider
                self.testBatchStatus = .ready
                self.benchmarkSuite.takeActionSnapshot(snapshot: ActionSnaptshot(action: "finished test dataset"))
            }
        }
    }
    
    public func initLocalClient() {
        self.benchmarkSuite.takeActionSnapshot(snapshot: ActionSnaptshot(action: "init local client"))
        self.localClientStatus = .preparing(count: 0)
        if self.localClient == nil {
            DispatchQueue.global(qos: .userInitiated).async {
                let dataLoader = MLDataLoader(trainBatchProvider: self.trainingBatchProvider!, testBatchProvider: self.testBatchProvider!)
                if let modelUrl = Bundle.main.url(forResource: "MNIST_Model", withExtension: "mlmodel") {
                    self.initClient(modelUrl: modelUrl, dataLoader: dataLoader, clientType: .local)
                    DispatchQueue.main.async {
                        self.localClientStatus = .ready
                        self.benchmarkSuite.takeActionSnapshot(snapshot: ActionSnaptshot(action: "local client ready"))
                    }
                }
            }
        } else {
            self.localClientStatus = .ready
            self.benchmarkSuite.takeActionSnapshot(snapshot: ActionSnaptshot(action: "local client ready"))
        }
    }
    
    public func initMLFlwrClient() {
        self.mlFlwrClientStatus = .preparing(count: 0)
        self.benchmarkSuite.takeActionSnapshot(snapshot: ActionSnaptshot(action: "init ML Flwr Client"))
        if self.mlFlwrClient == nil {
            DispatchQueue.global(qos: .userInitiated).async {
                let dataLoader = MLDataLoader(trainBatchProvider: self.trainingBatchProvider!, testBatchProvider: self.testBatchProvider!)
                if let modelUrl = Bundle.main.url(forResource: "MNIST_Model", withExtension: "mlmodel") {
                    self.initClient(modelUrl: modelUrl, dataLoader: dataLoader, clientType: .federated)
                    DispatchQueue.main.async {
                        self.mlFlwrClientStatus = .ready
                        self.benchmarkSuite.takeActionSnapshot(snapshot: ActionSnaptshot(action: "ML Flwr Client ready"))
                    }
                }
            }
        } else {
            self.mlFlwrClientStatus = .ready
            self.benchmarkSuite.takeActionSnapshot(snapshot: ActionSnaptshot(action: "ML Flwr Client ready"))
        }
    }
    
    private func initClient(modelUrl url: URL, dataLoader: MLDataLoader, clientType: ClientType) {
        do {
            let compiledModelUrl = try MLModel.compileModel(at: url)
            switch clientType {
            case .federated:
                let modelInspect = try MLModelInspect(serializedData: Data(contentsOf: url))
                let layerWrappers = modelInspect.getLayerWrappers()
                self.mlFlwrClient = MLFlwrClient(layerWrappers: layerWrappers,
                                                 dataLoader: dataLoader,
                                                 compiledModelUrl: compiledModelUrl)
            case .local:
                self.localClient = LocalClient(dataLoader: dataLoader, compiledModelUrl: compiledModelUrl)
            }
            
        } catch let error {
            print(error)
        }
    }
    
    public func startLocalTrain() {
        let statusHandler: (TaskStatus) -> Void = { status in
            DispatchQueue.main.async {
                self.localTrainingStatus = status
                self.benchmarkSuite.takeActionSnapshot(snapshot: ActionSnaptshot(action: "local train status: \(status)"))
            }
        }
        localClient!.runMLTask(statusHandler: statusHandler, numEpochs: self.epoch, task: .train)
    }
    
    public func startLocalTest() {
        let statusHandler: (TaskStatus) -> Void = { status in
            DispatchQueue.main.async {
                self.localTestStatus = status
                self.benchmarkSuite.takeActionSnapshot(snapshot: ActionSnaptshot(action: "local test status: \(status)"))
            }
        }
        localClient!.runMLTask(statusHandler: statusHandler, numEpochs: 1, task: .test)
    }
    
    public func startFederatedLearning() {
        print("starting federated learning")
        self.federatedServerStatus = .ongoing(info: "Starting federated learning")
        self.benchmarkSuite.takeActionSnapshot(snapshot: ActionSnaptshot(action: "starting federated learning"))
        if self.flwrGRPC == nil {
            self.flwrGRPC = FlwrGRPC(serverHost: hostname, serverPort: port, extendedInterceptor: BenchmarkInterceptor())
        }
        
        self.flwrGRPC?.startFlwrGRPC(client: self.mlFlwrClient!) {
            DispatchQueue.main.async {
                self.federatedServerStatus = .completed(info: "Federated learning completed")
                self.flwrGRPC = nil
                self.benchmarkSuite.takeActionSnapshot(snapshot: ActionSnaptshot(action: "Federated learning completed"))
            }
        }
    }
    
    public func abortFederatedLearning() {
        print("aborting federated learning")
        self.flwrGRPC?.abortGRPCConnection(reasonDisconnect: .powerDisconnected) {
            DispatchQueue.main.async {
                self.federatedServerStatus = .completed(info: "Federated learning aborted")
                self.flwrGRPC = nil
                self.benchmarkSuite.takeActionSnapshot(snapshot: ActionSnaptshot(action: "Federated learning aborted"))
            }
        }
    }
}

public enum PreparationStatus: Comparable {
    case notPrepared
    case preparing(count: Int)
    case ready
    
    var description: String {
        switch self {
        case .notPrepared:
            return "Not Prepared"
        case .preparing(let count):
            return "Preparing \(count)"
        case .ready:
            return "Ready"
        }
    }
}

public enum TaskStatus: Equatable {
    case idle
    case ongoing(info: String)
    case completed(info: String)
    
    var description: String {
        switch self {
        case .idle:
            return "Not Yet Started"
        case .ongoing(let info):
            return info
        case .completed(let info):
            return info
        }
    }
    
    public static func ==(lhs: TaskStatus, rhs: TaskStatus) -> Bool {
        switch(lhs, rhs) {
        case (.idle, .idle):
            return true
        case (.ongoing(_), .ongoing(_)):
            return true
        case (.completed(_), .completed(_)):
            return true
        default:
            return false
        }
    }
}

public enum ClientType {
    case federated
    case local
}

class LocalClient {
    private var dataLoader: MLDataLoader
    private var compiledModelUrl: URL
    
    init(dataLoader: MLDataLoader, compiledModelUrl: URL) {
        self.dataLoader = dataLoader
        self.compiledModelUrl = compiledModelUrl
    }
    
    func runMLTask(statusHandler: @escaping (TaskStatus) -> Void,
                   numEpochs: Int,
                   task: flwr.MLTask
    ) {
        let dataset: MLBatchProvider
        let configuration = MLModelConfiguration()
        let epochs = MLParameterKey.epochs
        configuration.parameters = [epochs:numEpochs]
        
        switch task {
        case .train:
            dataset = self.dataLoader.trainBatchProvider
        case .test:
            dataset = self.dataLoader.testBatchProvider
        }
        
        var startTime = Date()
        let progressHandler = { (contextProgress: MLUpdateContext) in
            switch contextProgress.event {
            case .trainingBegin:
                let taskStatus: TaskStatus = .ongoing(info: "Started to \(task) locally")
                statusHandler(taskStatus)
            case .epochEnd:
                let taskStatus: TaskStatus
                let loss = String(format: "%.4f", contextProgress.metrics[.lossValue] as! Double)
                switch task {
                case .train:
                    let epochIndex = contextProgress.metrics[.epochIndex] as! Int
                    taskStatus = .ongoing(info: "Epoch \(epochIndex + 1) end with loss \(loss)")
                case .test:
                    taskStatus = .ongoing(info: "Local test end with loss \(loss)")
                }
                statusHandler(taskStatus)
            default:
                print("Unknown event")
            }
        }
        
        let completionHandler = { (finalContext: MLUpdateContext) in
            let loss = String(format: "%.4f", finalContext.metrics[.lossValue] as! Double)
            let taskStatus: TaskStatus = .completed(info: "Local \(task) completed with loss: \(loss) in \(Int(Date().timeIntervalSince(startTime))) secs")
            statusHandler(taskStatus)
        }
        
        
        let progressHandlers = MLUpdateProgressHandlers(
            forEvents: [.trainingBegin, .epochEnd],
            progressHandler: progressHandler,
            completionHandler: completionHandler
        )
        
        startTime = Date()
        do {
            let updateTask = try MLUpdateTask(forModelAt: compiledModelUrl,
                                              trainingData: dataset,
                                              configuration: configuration,
                                              progressHandlers: progressHandlers)
            updateTask.resume()
            
        } catch let error {
            print(error)
        }
    }
}
