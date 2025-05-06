//
//  Parameter.swift
//  FlowerSDK
//
//  Created by Daniel Nugraha on 09.01.22.
//

import Foundation
import NumPySupport
import PythonSupport
import PythonKit
import CoreML
import NIOCore
import NIOPosix
import os

let appDirectory = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!

/// A class responsible for (de)serializing model parameters.
///
/// ## Topics
///
/// ### Usage
///
/// - ``shared``
/// - ``finalize()``
///
/// ### Serialization
///
/// - ``multiArrayToData(multiArray:)``
/// - ``arrayToData(array:shape:)``
///
/// ### Deserialization
///
/// - ``dataToMultiArray(data:)``
/// - ``dataToArray(data:)``
@available(iOS 14.0, *)
public class ParameterConverter {
    private var np: PythonObject?
    
    /// The permanent location of the numpyArray.
    private var numpyArrayUrl = appDirectory.appendingPathComponent("numpyArray.npy")
    private var group: EventLoopGroup?
    
    private let log = Logger(subsystem: Bundle.main.bundleIdentifier ?? "flwr.Flower",
                                    category: String(describing: ParameterConverter.self))
    
    /// ParameterConverter singleton object.
    public static let shared = ParameterConverter()
    
    private init() {
        initGroup()
    }
    
    private func initGroup() {
        if group == nil {
            log.log("Opening Python event loop group")
            group = MultiThreadedEventLoopGroup(numberOfThreads: 1)
            
            let future = group?.next().submit {
                PythonSupport.initialize()
                NumPySupport.sitePackagesURL.insertPythonPath()
                self.np = Python.import("numpy")
            }
            
            do {
                try future?.wait()
            } catch {
                log.error("\(error)")
            }
        }
    }
    
    private func multiArrayToNumpy(multiArray: MLMultiArray) -> PythonObject? {
        let pointer = try! UnsafeBufferPointer<Float>(multiArray)
        let array = pointer.compactMap{$0}
        let shape = multiArray.shape.map { Int16(truncating: $0) }
        let filteredShape = shape.filter { $0 > 1 }
        return array.makeNumpyArray().reshape(filteredShape)
    }
    
    private func numpyToData(numpy: PythonObject) -> Data? {
        guard np != nil else {
            return nil
        }
        
        guard Python.isinstance(numpy, np?.ndarray) == true else {
            return nil
        }
        
        let fileManager = FileManager.default
        if fileManager.fileExists(atPath: numpyArrayUrl.path) {
            try? fileManager.removeItem(at: numpyArrayUrl)
        }
        
        np?.save(numpyArrayUrl.path, numpy)
        return try? Data(contentsOf: numpyArrayUrl)
    }
    
    private func dataToNumpy(data: Data) -> PythonObject? {
        guard np != nil else {
            return nil
        }
        
        let fileManager = FileManager.default
        if fileManager.fileExists(atPath: numpyArrayUrl.path) {
            try? fileManager.removeItem(at: numpyArrayUrl)
        }
        try? data.write(to: numpyArrayUrl)
        
        return np?.load(numpyArrayUrl.path)
    }
    
    private func numpyToArray(numpy: PythonObject) -> [Float]? {
        guard np != nil else {
            return nil
        }
        
        guard Python.isinstance(numpy, np?.ndarray) == true else {
            return nil
        }
        let flattened = numpy.flatten()
        return Array<Float>(numpy: flattened)
    }
    
    private func numpyToMultiArray(numpy: PythonObject) -> MLMultiArray? {
        guard np != nil else {
            return nil
        }
        
        guard Python.isinstance(numpy, np?.ndarray) == true else {
            return nil
        }
        
        let pyShape = numpy.__array_interface__["shape"]
        guard let shape = Array<Int>(pyShape) else { return nil }
        let shapeNSNumber = shape.map { NSNumber(value: $0) }
        
        if let swiftArray = numpyToArray(numpy: numpy),
           let multiArray = try? MLMultiArray(shape: shapeNSNumber, dataType: .float) {
            for (index, element) in swiftArray.enumerated() {
                multiArray[index] = NSNumber(value: element)
            }
            return multiArray
        }
        return nil
    }
    
    /// Deserialize bytes to MLMultiArray.
    public func dataToMultiArray(data: Data) -> MLMultiArray? {
        initGroup()
        let future = group?.next().submit {
            if let numpy = self.dataToNumpy(data: data) {
                return self.numpyToMultiArray(numpy: numpy)
            }
            return nil
        }
        
        do {
            let ret = try future?.wait()
            return ret
        } catch {
            log.error("\(error)")
            return nil
        }
        
    }
    
    /// Serialize MLMultiArray to bytes.
    public func multiArrayToData(multiArray: MLMultiArray) -> Data? {
        initGroup()
        let future = group?.next().submit {
            if let numpy = self.multiArrayToNumpy(multiArray: multiArray) {
                return self.numpyToData(numpy: numpy)
            }
            return nil
        }
        
        do {
            let ret = try future?.wait()
            return ret
        } catch {
            log.error("\(error)")
            return nil
        }
        
    }
    
    /// Deserialize bytes into float array.
    public func dataToArray(data: Data) -> [Float]? {
        initGroup()
        let future = group?.next().submit {
            if let numpy = self.dataToNumpy(data: data) {
                return self.numpyToArray(numpy: numpy)
            }
            return nil
        }
        
        do {
            let ret = try future?.wait()
            return ret
        } catch {
            log.error("\(error)")
            return nil
        }
        
    }
    
    /// Serialize float array to bytes.
    public func arrayToData(array: [Float], shape: [Int16]) -> Data? {
        initGroup()
        let future = group?.next().submit {
            let numpy = array.makeNumpyArray().reshape(shape)
            return self.numpyToData(numpy: numpy)
        }
        
        do {
            let ret = try future?.wait()
            return ret
        } catch {
            log.error("\(error)")
            return nil
        }
    }
    
    /// Shutdown EventLoopGroup gracefully.
    public func finalize() {
        initGroup()
        let future = group?.next().submit {
            PythonSupport.finalize()
        }
        
        do {
            try future?.wait()
            try group?.syncShutdownGracefully()
            group = nil
        } catch {
            log.error("\(error)")
        }
    }
}

