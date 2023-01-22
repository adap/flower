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

let appDirectory = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!

class MNISTDataLoader {
    static func trainBatchProvider(progressHandler: @escaping (Int) -> Void) -> MLBatchProvider {
        return prepareMLBatchProvider(filePath: extractTrainFile(), progressHandler: progressHandler)
    }
    
    static func testBatchProvider(progressHandler: @escaping (Int) -> Void) -> MLBatchProvider {
        return prepareMLBatchProvider(filePath: extractTestFile(), progressHandler: progressHandler)
    }

    /// Extract file
    ///
    /// - parameter sourceURL:URL of source file
    /// - parameter destinationFilename: Choosen destination filename
    ///
    /// - returns: Temporary path of extracted file
    fileprivate static func extractFile(from sourceURL: URL, to destinationURL: URL) -> String {
        let sourceFileHandle = try! FileHandle(forReadingFrom: sourceURL)
        var isDir:ObjCBool = true
        let fileManager = FileManager.default
        if !fileManager.fileExists(atPath: appDirectory.path, isDirectory: &isDir) {
            try! fileManager.createDirectory(at: appDirectory, withIntermediateDirectories: true)
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
    private static func extractTrainFile() -> String {
        let sourceURL = Bundle.main.url(forResource: "mnist_train", withExtension: "csv.lzfse")!
        let destinationURL = appDirectory.appendingPathComponent("mnist_train.csv")
        return extractFile(from: sourceURL, to: destinationURL)
    }

    /// Extract test file
    ///
    /// - returns: Temporary path of extracted file
    private static func extractTestFile() -> String {
        let sourceURL = Bundle.main.url(forResource: "mnist_test", withExtension: "csv.lzfse")!
        let destinationURL = appDirectory.appendingPathComponent("mnist_test.csv")
        return extractFile(from: sourceURL, to: destinationURL)
    }
    
    private static func prepareMLBatchProvider(filePath: String, progressHandler: @escaping (Int) -> Void) -> MLBatchProvider {
        var featureProviders = [MLFeatureProvider]()
        
        var count = 0
        errno = 0
        
        if freopen(filePath, "r", stdin) == nil {
            print("error opening file")
        }
        while let line = readLine()?.split(separator: ",") {
            if count == 1000 {
                break
            }
            count += 1
            progressHandler(count)
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
    
    static func predictionBatchProvider() -> MLBatchProvider {
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
