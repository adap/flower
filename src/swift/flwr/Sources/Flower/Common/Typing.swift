//
//  Typing.swift
//  FlowerSDK
//
//  Created by Daniel Nugraha on 09.01.22.
//

import Foundation

public enum Code {
    case ok
    case getPropertiesNotImplemented
    case getParametersNotImplemented
    case fitNotImplemented
    case evaluateNotImplemented
}

public struct Scalar {
    public var bool: Bool?
    public var bytes: Data?
    public var float: Float?
    public var int: Int?
    public var str: String?
}

public typealias Metrics = [String: Scalar]
public typealias Properties = [String: Scalar]

public struct Status {
    public var code: Code
    public var message: String
    
    public init(code: Code, message: String) {
        self.code = code
        self.message = message
    }
}

public struct Parameters {
    public var tensors: [Data]
    public var tensorType: String
    
    public init(tensors: [Data], tensorType: String) {
        self.tensors = tensors
        self.tensorType = tensorType
    }
}

public struct ParametersRes {
    public var parameters: Parameters
    public var status: Status
    
    public init(parameters: Parameters) {
        self.parameters = parameters
        self.status = Status(code: .ok, message: String())
    }
}

public struct FitIns {
    public var parameters: Parameters
    public var config: [String: Scalar]
}

public struct FitRes {
    public var parameters: Parameters
    public var numExamples: Int
    public var metrics: Metrics? = nil
    public var status: Status
    
    public init(parameters: Parameters, numExamples: Int, metrics: Metrics? = nil) {
        self.parameters = parameters
        self.numExamples = numExamples
        self.metrics = metrics
        self.status = Status(code: .ok, message: String())
    }
}

public struct EvaluateIns {
    public var parameters: Parameters
    public var config: [String: Scalar]
}

public struct EvaluateRes {
    public var loss: Float
    public var numExamples: Int
    public var metrics: Metrics? = nil
    public var status: Status
    
    public init(loss: Float, numExamples: Int, metrics: Metrics? = nil) {
        self.loss = loss
        self.numExamples = numExamples
        self.metrics = metrics
        self.status = Status(code: .ok, message: String())
    }
}

public struct PropertiesIns {
    public var config: Properties
}

public struct PropertiesRes {
    public var properties: Properties
    public var status: Status = Status(code: .ok, message: String())
}

public struct Reconnect {
    public var seconds: Int?
}

public struct Disconnect {
    public var reason: String
}
