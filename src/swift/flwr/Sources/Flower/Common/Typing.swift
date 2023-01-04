//
//  Typing.swift
//  FlowerSDK
//
//  Created by Daniel Nugraha on 09.01.22.
//

import Foundation

public enum Code {
    typealias RawValue = Int
    case ok
    case getPropertiesNotImplemented
    case getParametersNotImplemented
    case fitNotImplemented
    case evaluateNotImplemented
    case UNRECOGNIZED(Int)
    
    init(rawValue: Int) {
        switch rawValue {
            case 0: self = .ok
            case 1: self = .getPropertiesNotImplemented
            case 2: self = .getParametersNotImplemented
            case 3: self = .fitNotImplemented
            case 4: self = .evaluateNotImplemented
            default: self = .UNRECOGNIZED(rawValue)
        }
    }
    
    var rawValue: Int {
      switch self {
      case .ok: return 0
      case .getPropertiesNotImplemented: return 1
      case .getParametersNotImplemented: return 2
      case .fitNotImplemented: return 3
      case .evaluateNotImplemented: return 4
      case .UNRECOGNIZED(let i): return i
      }
    }
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

public struct GetParametersRes {
    public var parameters: Parameters
    public var status: Status
    
    public init(parameters: Parameters, status: Status) {
        self.parameters = parameters
        self.status = status
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
    
    public init(parameters: Parameters, numExamples: Int, metrics: Metrics? = nil, status: Status) {
        self.parameters = parameters
        self.numExamples = numExamples
        self.metrics = metrics
        self.status = status
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
    
    public init(loss: Float, numExamples: Int, metrics: Metrics? = nil, status: Status) {
        self.loss = loss
        self.numExamples = numExamples
        self.metrics = metrics
        self.status = status
    }
}

public struct GetPropertiesIns {
    public var config: Properties
}

public struct GetPropertiesRes {
    public var properties: Properties
    public var status: Status
    public init(properties: Properties, status: Status) {
        self.properties = properties
        self.status = status
    }
}

public struct Reconnect {
    public var seconds: Int?
}

public struct Disconnect {
    public var reason: String
}
