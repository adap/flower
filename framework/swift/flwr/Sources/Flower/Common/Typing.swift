//
//  Typing.swift
//  FlowerSDK
//
//  Created by Daniel Nugraha on 09.01.22.
//

import Foundation

/// Client status codes.
///
/// ## Topics
///
/// ### Status Codes
///
/// - ``ok``
/// - ``getParametersNotImplemented``
/// - ``getPropertiesNotImplemented``
/// - ``fitNotImplemented``
/// - ``evaluateNotImplemented``
/// - ``UNRECOGNIZED(_:)``
public enum Code: Equatable {
    typealias RawValue = Int
    
    /// Everything is okay status code.
    case ok
    
    /// No client implementation for getProperties status code.
    case getPropertiesNotImplemented
    
    /// No client implementation for getParameters status code.
    case getParametersNotImplemented
    
    /// No client implementation for fit status code.
    case fitNotImplemented
    
    /// No client implementation for evaluate status code.
    case evaluateNotImplemented
    
    /// Unrecognized client status code.
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

/// Set of disconnect reasons for client.
///
/// ## Topics
///
/// ### Disconnect Reasons
///
/// - ``unknown``
/// - ``reconnect``
/// - ``powerDisconnected``
/// - ``wifiUnavailable``
/// - ``ack``
/// - ``UNRECOGNIZED(_:)``
public enum ReasonDisconnect {
    typealias RawValue = Int
    
    /// Unknown disconnect reason.
    case unknown // = 0
    
    /// Reconnect disconnect reason.
    case reconnect // = 1
    
    /// Power disconnected disconnect reason.
    case powerDisconnected // = 2
    
    /// WiFi unavailable disconnect reason.
    case wifiUnavailable // = 3
    
    /// Acknowledge disconnect reason.
    case ack // = 4
    
    /// Unrecognized disconnect reason.
    case UNRECOGNIZED(Int)

    var rawValue: Int {
      switch self {
      case .unknown: return 0
      case .reconnect: return 1
      case .powerDisconnected: return 2
      case .wifiUnavailable: return 3
      case .ack: return 4
      case .UNRECOGNIZED(let i): return i
      }
    }
}

/// Container for a set of recognised single quantity values.
///
/// ## Topics
///
/// ### Scalar Values
///
/// - ``bool``
/// - ``bytes``
/// - ``float``
/// - ``int``
/// - ``str``
public struct Scalar: Equatable {
    
    /// Boolean scalar value.
    public var bool: Bool?
    
    /// Raw bytes scalar value.
    public var bytes: Data?
    
    /// Float scalar value.
    public var float: Float?
    
    /// Integer scalar value.
    public var int: Int?
    
    /// String scalar value.
    public var str: String?
}

/// Typealias for a dictionary containing String and Scalar key-value pairs.
public typealias Metrics = [String: Scalar]

/// Typealias for a dictionary containing String and Scalar key-value pairs.
public typealias Properties = [String: Scalar]


/// Client status.
public struct Status: Equatable {
    public static func == (lhs: Status, rhs: Status) -> Bool {
        if lhs.code == rhs.code && lhs.message == rhs.message {
            return true
        }
        return false
    }
    
    public var code: Code
    public var message: String
    
    public init(code: Code, message: String) {
        self.code = code
        self.message = message
    }
}

/// Parameters message.
public struct Parameters: Equatable {
    public var tensors: [Data]
    public var tensorType: String
    
    public init(tensors: [Data], tensorType: String) {
        self.tensors = tensors
        self.tensorType = tensorType
    }
}

/// Response when asked to return parameters.
public struct GetParametersRes: Equatable {
    public var parameters: Parameters
    public var status: Status
    
    public init(parameters: Parameters, status: Status) {
        self.parameters = parameters
        self.status = status
    }
}

/// Fit instructions for a client.
public struct FitIns: Equatable {
    public var parameters: Parameters
    public var config: [String: Scalar]
}

/// Fit response from a client.
public struct FitRes: Equatable {
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

/// Evaluate instructions for a client.
public struct EvaluateIns: Equatable {
    public var parameters: Parameters
    public var config: [String: Scalar]
}

/// Evaluate response from a client.
public struct EvaluateRes: Equatable {
    public static func == (lhs: EvaluateRes, rhs: EvaluateRes) -> Bool {
        if lhs.loss == rhs.loss && lhs.numExamples == rhs.numExamples && lhs.metrics == rhs.metrics && lhs.status == rhs.status {
            return true
        }
        return false
    }
    
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

/// Properties request for a client.
public struct GetPropertiesIns: Equatable {
    public var config: Properties
}

/// Properties response from a client.
public struct GetPropertiesRes: Equatable {
    public var properties: Properties
    public var status: Status
    public init(properties: Properties, status: Status) {
        self.properties = properties
        self.status = status
    }
}

/// Reconnect message from server to client.
public struct Reconnect: Equatable {
    public var seconds: Int?
}

/// Disconnect message from client to server.
public struct Disconnect: Equatable {
    public var reason: String
}
