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

@available(iOS 14.0, *)
public class ParameterConverter {
    var np: PythonObject
    var io: PythonObject
    var typing: PythonObject
    
    private static let appDirectory = FileManager.default.urls(for: .applicationSupportDirectory,
                                                               in: .userDomainMask).first!
    /// The permanent location of the numpyArray.
    private var numpyArrayUrl = appDirectory.appendingPathComponent("numpyArray.npy")
    
    public init() {
        PythonSupport.initialize()
        NumPySupport.sitePackagesURL.insertPythonPath()
        np = Python.import("numpy")
        io = Python.import("io")
        typing = Python.import("typing")
    }
    
    private func multiArrayToNumpy(multiArray: MLMultiArray) -> PythonObject? {
        let pointer = try! UnsafeBufferPointer<Float>(multiArray)
        let array = pointer.compactMap{$0}
        let shape = multiArray.shape.map { Int16(truncating: $0) }
        let filteredShape = shape.filter { $0 > 1 }
        return array.makeNumpyArray().reshape(filteredShape)
    }
    
    private func numpyToData(numpy: PythonObject) -> Data? {
        guard Python.isinstance(numpy, np.ndarray) == true else {
            return nil
        }
        
        let fileManager = FileManager.default
        if fileManager.fileExists(atPath: numpyArrayUrl.path) {
            try? fileManager.removeItem(at: numpyArrayUrl)
        }
        
        np.save(numpyArrayUrl.path, numpy)
        return try? Data(contentsOf: numpyArrayUrl)
    }
    
    private func dataToNumpy(data: Data) -> PythonObject? {
        let fileManager = FileManager.default
        if fileManager.fileExists(atPath: numpyArrayUrl.path) {
            try? fileManager.removeItem(at: numpyArrayUrl)
        }
        try? data.write(to: numpyArrayUrl)
        
        return np.load(numpyArrayUrl.path)
    }
    
    private func numpyToArray(numpy: PythonObject) -> [Float]? {
        guard Python.isinstance(numpy, np.ndarray) == true else {
            return nil
        }
        let flattened = numpy.flatten()
        return Array<Float>(numpy: flattened)
    }
    
    private func numpyToMultiArray(numpy: PythonObject) -> MLMultiArray? {
        guard Python.isinstance(numpy, np.ndarray) == true else {
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
    
    public func dataToMultiArray(data: Data) -> MLMultiArray? {
        if let numpy = dataToNumpy(data: data) {
            return numpyToMultiArray(numpy: numpy)
        }
        return nil
    }
    
    public func multiArrayToData(multiArray: MLMultiArray) -> Data? {
        if let numpy = multiArrayToNumpy(multiArray: multiArray) {
            return numpyToData(numpy: numpy)
        }
        return nil
    }
    
    public func dataToArray(data: Data) -> [Float]? {
        if let numpy = dataToNumpy(data: data) {
            return numpyToArray(numpy: numpy)
        }
        return nil
    }
    
    public func arrayToData(array: [Float], shape: [Int16]) -> Data? {
        let numpy = array.makeNumpyArray().reshape(shape)
        //print("before shape: \(shape)")
        //print("after shape: \(numpy.shape)")
        return numpyToData(numpy: numpy)
    }
}

