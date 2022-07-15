//
//  DataLoader.swift
//  FlowerSDK
//
//  Created by Daniel Nugraha on 25.05.22.
//  Adapted from https://github.com/JacopoMangiavacchi/MNIST-CoreML-Training

import Foundation
import CoreML
import Compression
import UIKit

public class MNISTDataLoader {
    private static let appDirectory = FileManager.default.urls(for: .applicationSupportDirectory,
                                                               in: .userDomainMask).first!

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
        if !fileManager.fileExists(atPath: MNISTDataLoader.appDirectory.path, isDirectory: &isDir) {
            try! fileManager.createDirectory(at: MNISTDataLoader.appDirectory, withIntermediateDirectories: true)
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
        let destinationURL = MNISTDataLoader.appDirectory.appendingPathComponent("mnist_train.csv")
        return extractFile(from: sourceURL, to: destinationURL)
    }

    /// Extract test file
    ///
    /// - returns: Temporary path of extracted file
    public func extractTestFile() -> String {
        let sourceURL = Bundle.main.url(forResource: "mnist_test", withExtension: "csv.lzfse")!
        let destinationURL = MNISTDataLoader.appDirectory.appendingPathComponent("mnist_test.csv")
        return extractFile(from: sourceURL, to: destinationURL)
    }
    
    public func trainBatchProvider() -> MLBatchProvider {
        var featureProviders = [MLFeatureProvider]()
        
        var count = 0
        errno = 0
        
        let trainFilePath = extractTrainFile()
        if freopen(trainFilePath, "r", stdin) == nil {
            print("error opening file")
        }
        while let line = readLine()?.split(separator: ",") {
            count += 1
            
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
    
    public func testBatchProvider() -> MLBatchProvider {
        var featureProviders = [MLFeatureProvider]()
        
        var count = 0
        errno = 0
        
        let testFilePath = extractTestFile()
        if freopen(testFilePath, "r", stdin) == nil {
            print("error opening file")
        }
        while let line = readLine()?.split(separator: ",") {
            count += 1
            
            let imageMultiArr = try! MLMultiArray(shape: [1, 28, 28], dataType: .float32)

            for r in 0..<28 {
                for c in 0..<28 {
                    let i = (r*28)+c
                    imageMultiArr[i] = NSNumber(value: Float(String(line[i + 1]))! / Float(255.0))
                }
            }
            
    
        
            let imageValue = MLFeatureValue(multiArray: imageMultiArr)

            let dataPointFeatures: [String: MLFeatureValue] = ["image": imageValue]
            
            
            if let provider = try? MLDictionaryFeatureProvider(dictionary: dataPointFeatures) {
                featureProviders.append(provider)
            }
        }

        return MLArrayBatchProvider(array: featureProviders)
    }
}
