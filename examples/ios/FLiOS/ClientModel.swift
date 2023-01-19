//
//  ClientModel.swift
//  FlowerCoreML
//
//  Created by Daniel Nugraha on 24.06.22.
//

import Foundation
import flwr
import CoreML
import Compression
import Combine

public class ClientModel: ObservableObject {
    private var coreMLClient: CoreMLClient?
    private var flwrGRPC: FlwrGRPC?
    private static let appDirectory = FileManager.default.urls(for: .applicationSupportDirectory,
                                                               in: .userDomainMask).first!
    
    public var benchmarkSuite = BenchmarkSuite.shared
    
    @Published public var trainingBatchStatus = BatchPreparationStatus.notPrepared
    @Published public var testBatchStatus = BatchPreparationStatus.notPrepared
    @Published public var modelCompilationStatus = BatchPreparationStatus.notPrepared
    
    @Published public var trainingBatchProvider: MLBatchProvider?
    @Published public var testBatchProvider: MLBatchProvider?

    
    @Published public var hostname: String = "localhost"
    @Published public var port: Int = 8080
    
    @Published public var epoch: Int = 5
    @Published public var modelStatus = BatchPreparationModelStatus.notPrepared
    
    @Published public var federatedServerStatus = ServerStatus.stop
    
    private var cancellable = Set<AnyCancellable>()
    
    public func prepareTrainDataset() {
        trainingBatchStatus = .preparing(count: 0)
        self.benchmarkSuite.takeActionSnapshot(snapshot: ActionSnaptshot(action: "startedPreparingTrainDataset"))
        DispatchQueue.global(qos: .userInitiated).async {
            let batchProvider = self.prepareTrainBatchProvider()
            DispatchQueue.main.async {
                self.trainingBatchProvider = batchProvider
                self.trainingBatchStatus = .ready
                self.benchmarkSuite.takeActionSnapshot(snapshot: ActionSnaptshot(action: "finishedPreparingTrainDataset"))
            }
        }
    }
    
    public func prepareTestDataset() {
        testBatchStatus = .preparing(count: 0)
        self.benchmarkSuite.takeActionSnapshot(snapshot: ActionSnaptshot(action: "startedPreparingTestDataset"))
        DispatchQueue.global(qos: .userInitiated).async {
            let batchProvider = self.prepareTestBatchProvider()
            DispatchQueue.main.async {
                self.testBatchProvider = batchProvider
                self.testBatchStatus = .ready
                self.benchmarkSuite.takeActionSnapshot(snapshot: ActionSnaptshot(action: "finishedPreparingTestDataset"))
            }
        }
    }
    
    private func initCoreMLClient() {
        if coreMLClient == nil {
            DispatchQueue.global(qos: .userInitiated).async {
                let url = Bundle.main.url(forResource: "MNIST_Model", withExtension: "mlmodel")
                let dataLoader = DataLoader(trainBatchProvider: self.trainingBatchProvider!, testBatchProvider: self.testBatchProvider!)
                self.coreMLClient = CoreMLClient(modelUrl: url!, dataLoader: dataLoader)
                DispatchQueue.main.async {
                    self.modelCompilationStatus = .ready
                }
            }
        }
        else {
            modelCompilationStatus = .ready
        }
    }
    
    public func compileModel() {
        self.benchmarkSuite.takeActionSnapshot(snapshot: ActionSnaptshot(action: "startedCompilingModel"))
        self.modelCompilationStatus = .preparing(count: 0)
        initCoreMLClient()
        self.benchmarkSuite.takeActionSnapshot(snapshot: ActionSnaptshot(action: "finishedCompilingModel"))
    }
    
    public func startLocalTraining() {
        self.benchmarkSuite.takeActionSnapshot(snapshot: ActionSnaptshot(action: "startedLocalTraining"))
        let configuration = MLModelConfiguration()
        let epochs = MLParameterKey.epochs
        configuration.parameters = [epochs:self.epoch]
        configuration.computeUnits = .all
        
        var trainingStartTime = Date()

        
        let progressHandler = { (contextProgress: MLUpdateContext) in
            switch contextProgress.event {
            case .trainingBegin:
                DispatchQueue.main.async {
                    self.modelStatus = .preparing(info: "Started to train")
                }
            case .epochEnd:
                let epochIndex = contextProgress.metrics[.epochIndex] as! Int
                let trainLoss = contextProgress.metrics[.lossValue] as! Double
                DispatchQueue.main.async {
                    self.modelStatus = .preparing(info: "Epoch \(epochIndex + 1) end with loss \(trainLoss)")
                    self.benchmarkSuite.takeActionSnapshot(snapshot: ActionSnaptshot(action: "LocalTraining: Epoch \(epochIndex + 1) end with loss \(trainLoss)"))
                }

            default:
                print("Unknown event")
            
            }
        }
        
        let completionHandler = { (finalContext: MLUpdateContext) in
            let trainLoss = finalContext.metrics[.lossValue] as! Double
            DispatchQueue.main.async {
                self.modelStatus = .ready(info: "Training completed with loss: \(trainLoss) in \(Int(Date().timeIntervalSince(trainingStartTime))) secs")
                self.benchmarkSuite.takeActionSnapshot(snapshot: ActionSnaptshot(action: "LocalTraining: Training completed with loss: \(trainLoss) in \(Int(Date().timeIntervalSince(trainingStartTime))) secs"))
                self.benchmarkSuite.takeActionSnapshot(snapshot: ActionSnaptshot(action: "finishedLocalTraining"))
            }
        }
        
        trainingStartTime = Date()
        coreMLClient?.localTrain(configuration: configuration,
                                 progressHandler: progressHandler,
                                 completionHandler: completionHandler)
        
    }
    
    public func startLocalTest() {
        self.benchmarkSuite.takeActionSnapshot(snapshot: ActionSnaptshot(action: "startedLocalTest"))
        //coreMLClient?.test(modelConfig: <#T##MLModelConfiguration#>, result: <#T##(MLResult) -> Void#>)
        let configuration = MLModelConfiguration()
        let epochs = MLParameterKey.epochs
        configuration.parameters = [epochs:1]
        self.benchmarkSuite.takeActionSnapshot(snapshot: ActionSnaptshot(action: "finishedLocalTest"))
    }
    
    
    public func stopFederatedLearning() {
        if self.federatedServerStatus == .run {
            DispatchQueue.main.async {
                self.federatedServerStatus = .stop
                self.benchmarkSuite.takeActionSnapshot(snapshot: ActionSnaptshot(action: "stopFederatedLearning"))
            }
            self.flwrGRPC?.closeGRPCConnection()
        }
    }
    
    public func startFederatedLearning() {
        self.federatedServerStatus = .run
        initCoreMLClient()
        self.flwrGRPC = FlwrGRPC(serverHost: hostname, serverPort: port)
        self.flwrGRPC?.startFlwrGRPC(client: coreMLClient!)
    }
    
    /// Extract file
    ///
    /// - parameter sourceURL:URL of source file
    /// - parameter destinationFilename: Choosen destination filename
    ///
    /// - returns: Temporary path of extracted file
    fileprivate func extractFile(from sourceURL: URL, to destinationURL: URL) -> String {
        let sourceFileHandle = try! FileHandle(forReadingFrom: sourceURL)
        var isDir:ObjCBool = true
        let fileManager = FileManager.default
        if !fileManager.fileExists(atPath: ClientModel.appDirectory.path, isDirectory: &isDir) {
            try! fileManager.createDirectory(at: ClientModel.appDirectory, withIntermediateDirectories: true)
        }
        if fileManager.fileExists(atPath: destinationURL.path) {
            return destinationURL.path
        }
        FileManager.default.createFile(atPath: destinationURL.path,
                                       contents: nil,
                                       attributes: nil)
        
        let destinationFileHandle = try! FileHandle(forWritingTo: destinationURL)
        let bufferSize = 65536
        
        let filter = try! OutputFilter(.decompress, using: .lzfse, bufferCapacity: 655360) { data in
            if let data = data {
                destinationFileHandle.write(data)
            }
        }
        
        while true {
            let data = sourceFileHandle.readData(ofLength: bufferSize)
            
            try! filter.write(data)
            if data.count < bufferSize {
                break
            }
        }
        
        sourceFileHandle.closeFile()
        destinationFileHandle.closeFile()

        return destinationURL.path
    }

    /// Extract train file
    ///
    /// - returns: Temporary path of extracted file
    public func extractTrainFile() -> String {
        let sourceURL = Bundle.main.url(forResource: "mnist_train", withExtension: "csv.lzfse")!
        let destinationURL = ClientModel.appDirectory.appendingPathComponent("mnist_train.csv")
        return extractFile(from: sourceURL, to: destinationURL)
    }

    /// Extract test file
    ///
    /// - returns: Temporary path of extracted file
    public func extractTestFile() -> String {
        let sourceURL = Bundle.main.url(forResource: "mnist_test", withExtension: "csv.lzfse")!
        let destinationURL = ClientModel.appDirectory.appendingPathComponent("mnist_test.csv")
        return extractFile(from: sourceURL, to: destinationURL)
    }
    
    public func prepareTrainBatchProvider() -> MLBatchProvider {
        var featureProviders = [MLFeatureProvider]()
        
        var count = 0
        errno = 0
        
        let trainFilePath = extractTrainFile()
        if freopen(trainFilePath, "r", stdin) == nil {
            print("error opening file")
        }
        while let line = readLine()?.split(separator: ",") {
            count += 1
            DispatchQueue.main.async {
                self.trainingBatchStatus = .preparing(count: count)
            }
            let imageMultiArr = try! MLMultiArray(shape: [1, 28, 28], dataType: .float32)
            let outputMultiArr = try! MLMultiArray(shape: [1], dataType: .int32)

            for r in 0..<28 {
                for c in 0..<28 {
                    let i = (r*28)+c
                        imageMultiArr[i] = NSNumber(value: Float(String(line[i + 1]))! / Float(255.0))
                }
            }

            outputMultiArr[0] = NSNumber(value: Int(String(line[0]))!)
            
            let imageValue = MLFeatureValue(multiArray: imageMultiArr)
            let outputValue = MLFeatureValue(multiArray: outputMultiArr)

            let dataPointFeatures: [String: MLFeatureValue] = ["image": imageValue,
                                                               "output_true": outputValue]
            //let image = imageMultiArr.image(min: 0, max: 1)
            
            //let trainingInput = UpdatableMNISTDigitClassifierTrainingInput(image: image!.pixelBuffer()!, classLabel: String(line[0]))
                            
            //let imageValue = MLFeatureValue(pixelBuffer: (image?.pixelBuffer())!)
            //let outputValue = MLFeatureValue(multiArray: outputMultiArr)

            //let dataPointFeatures: [String: MLFeatureValue] = ["image": imageValue,
                                                              // "classLabel": outputValue]
            
            if let provider = try? MLDictionaryFeatureProvider(dictionary: dataPointFeatures) {
                featureProviders.append(provider)
            }
            //featureProviders.append(trainingInput)
        }
        
        return MLArrayBatchProvider(array: featureProviders)
    }
    
    public func prepareTestBatchProvider() -> MLBatchProvider {
        var featureProviders = [MLFeatureProvider]()
        
        var count = 0
        errno = 0
        
        let testFilePath = extractTestFile()
        if freopen(testFilePath, "r", stdin) == nil {
            print("error opening file")
        }
        while let line = readLine()?.split(separator: ",") {
            count += 1
            DispatchQueue.main.async {
                self.testBatchStatus = .preparing(count: count)
            }
            let imageMultiArr = try! MLMultiArray(shape: [1, 28, 28], dataType: .float32)
            let outputMultiArr = try! MLMultiArray(shape: [1], dataType: .int32)

            for r in 0..<28 {
                for c in 0..<28 {
                    let i = (r*28)+c
                    imageMultiArr[i] = NSNumber(value: Float(String(line[i + 1]))! / Float(255.0))
                }
            }
            
            outputMultiArr[0] = NSNumber(value: Int(String(line[0]))!)
            
            let imageValue = MLFeatureValue(multiArray: imageMultiArr)
            let outputValue = MLFeatureValue(multiArray: outputMultiArr)

            let dataPointFeatures: [String: MLFeatureValue] = ["image": imageValue,
                                                               "output_true": outputValue]
            
            if let provider = try? MLDictionaryFeatureProvider(dictionary: dataPointFeatures) {
                featureProviders.append(provider)
            }
        }

        return MLArrayBatchProvider(array: featureProviders)
    }
    
}

public enum BatchPreparationStatus: Comparable {
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

public enum BatchPreparationModelStatus {
    case notPrepared
    case preparing(info: String)
    case ready(info: String)
    
    var description: String {
        switch self {
        case .notPrepared:
            return "Not Prepared"
        case .preparing(let info):
            return info
        case .ready(let info):
            return info
        }
    }
}

public enum ServerStatus {
    case stop
    case run
}
