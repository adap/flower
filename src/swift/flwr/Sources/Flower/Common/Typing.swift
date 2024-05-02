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

/// A structure that stores metadata associated with the current message.
///
/// - Parameters:
///   - runID: An identifier for the current run.
///   - messageID: An identifier for the current message.
///   - srcNodeID: An identifier for the node sending this message.
///   - dstNodeID: An identifier for the node receiving this message.
///   - replyToMessage: An identifier for the message that this message replies to.
///   - groupID: An identifier for grouping messages. In some settings,
///     this is used as the federated learning (FL) round.
///   - ttl: Time-to-live for this message, in seconds.
///   - messageType: A string that encodes the action to be executed
///     on the receiving end.
///   - partitionID: An optional identifier used when loading a particular
///     data partition for a client application. This is more relevant
///     when conducting simulations.
public struct Metadata: Equatable {
    public var runID: Int64
    public var messageID: String
    public var srcNodeID: Int64
    public var dstNodeID: Int64
    public var replyToMessage: String
    public var groupID: String
    public var ttl: Double
    public var messageType: String
    public var partitionID: Int64?
    public let createdAt: TimeInterval // Unix timestamp (in seconds)
    
    public init(
        runID: Int64,
        messageID: String,
        srcNodeID: Int64,
        dstNodeID: Int64,
        replyToMessage: String,
        groupID: String,
        ttl: Double,
        messageType: String,
        partitionID: Int64? = nil
    ) {
        self.runID = runID
        self.messageID = messageID
        self.srcNodeID = srcNodeID
        self.dstNodeID = dstNodeID
        self.replyToMessage = replyToMessage
        self.groupID = groupID
        self.ttl = ttl
        self.messageType = messageType
        self.partitionID = partitionID
        self.createdAt = Date().timeIntervalSince1970
    }
}

/// A structure that stores the state of your run.
///
/// - Parameters:
///   - state: An instance of `RecordSet` that holds records added by the entity
///     in a given run and remains local. The data it holds never leaves the system
///     it's running on. This can be used as intermediate storage or a scratchpad
///     when executing modifications. It can also be used as memory to access
///     at different points during the lifecycle of this entity (e.g., across
///     multiple rounds).
public struct Context: Equatable {
    public var state: RecordSet
    
    public init(state: RecordSet) {
        self.state = state
    }
}

/// A structure that contains serialized data from an array-like or tensor-like object,
/// along with metadata about it.
///
/// - Parameters:
///   - dtype: A string representing the data type of the serialized object (e.g., `np.float32`).
///   - shape: An array of integers representing the shape of the unserialized array-like object.
///     This is used to deserialize the data (depending on the serialization method) or
///     serves as a metadata field.
///   - stype: A string indicating the type of serialization mechanism used to generate the
///     bytes in `data` from an array-like or tensor-like object.
///   - data: A buffer of bytes containing the data.
public struct MultiArray: Equatable {
    public var dtype: String
    public var shape: [Int32]
    public var stype: String
    public var data: Data
    
    public init(dtype: String, shape: [Int32], stype: String, data: Data) {
        self.dtype = dtype
        self.shape = shape
        self.stype = stype
        self.data = data
    }
}

/// Possible `MetricsRecord`'s value, which can be one of these types.
///
/// - Cases:
///   - double: Represents a `Double` metric value.
///   - sint64: Represents an `Int64` metric value.
///   - doubleList: Represents an array of `Double` metric values.
///   - sint64List: Represents an array of `Int64` metric values.
public enum MetricsRecordValue: Equatable {
    case double(Double)
    case sint64(Int64)
    case doubleList(Array<Double>)
    case sint64List(Array<Int64>)
}

/// Possible `ConfigsRecord`'s value, which can be one of these types.
///
/// - Cases:
///   - double: Represents a `Double` configuration value.
///   - sint64: Represents an `Int64` configuration value.
///   - bool: Represents a `Bool` configuration value.
///   - string: Represents a `String` configuration value.
///   - bytes: Represents a `Data` configuration value.
///   - doubleList: Represents an array of `Double` configuration values.
///   - sint64List: Represents an array of `Int64` configuration values.
///   - boolList: Represents an array of `Bool` configuration values.
///   - stringList: Represents an array of `String` configuration values.
///   - bytesList: Represents an array of `Data` configuration values.
public enum ConfigsRecordValue: Equatable {
    case double(Double)
    case sint64(Int64)
    case bool(Bool)
    case string(String)
    case bytes(Data)
    case doubleList(Array<Double>)
    case sint64List(Array<Int64>)
    case boolList(Array<Bool>)
    case stringList(Array<String>)
    case bytesList(Array<Data>)
}

internal enum Request: Equatable {
    case createNode(Flwr_Proto_CreateNodeRequest)
    case deleteNode(Flwr_Proto_DeleteNodeRequest)
    case pullTaskIns(Flwr_Proto_PullTaskInsRequest)
    case pushTaskRes(Flwr_Proto_PushTaskResRequest)
    case getRun(Flwr_Proto_GetRunRequest)
}

internal enum Response: Equatable {
    case createNode(Flwr_Proto_CreateNodeResponse)
    case deleteNode(Flwr_Proto_DeleteNodeResponse)
    case pullTaskIns(Flwr_Proto_PullTaskInsResponse)
    case pushTaskRes(Flwr_Proto_PushTaskResResponse)
    case getRun(Flwr_Proto_GetRunResponse)
}

/// A structure that stores parameters informations.
///
/// - Parameters:
///   - dataKeys: An array of keys representing the parameter names.
///   - dataValues: An array of `MultiArray` representing the parameter values.
public struct ParametersRecord: Equatable {
    public var dataKeys: [String]
    public var dataValues: [MultiArray]
    
    public init(dataKeys: [String], dataValues: [MultiArray]) {
        self.dataKeys = dataKeys
        self.dataValues = dataValues
    }
}

/// A structure that stores metric informations.
///
/// - Parameters:
///   - data: A dictionary with string and (`MetricsRecordValue`) pairs.
public struct MetricsRecord: Equatable {
    public var data: [String : MetricsRecordValue]
    
    public init(data: [String : MetricsRecordValue]) {
        self.data = data
    }
}

/// A structure that stores config informations.
///
/// - Parameters:
///   - data: A dictionary with string and (`ConfigsRecordValue`) pairs.
public struct ConfigsRecord: Equatable {
    public var data: [String : ConfigsRecordValue]
    
    public init(data: [String : ConfigsRecordValue]) {
        self.data = data
    }
}
/// Represents a group of parameters, metrics, and configurations.
///
/// - Parameters:
///   - parameters: A dictionary with string and `ParametersRecord` pairs.
///   - metrics: A dictionary with string and `MetricsRecord` pairs.
///   - configs: A dictionary with string and`ConfigsRecord` pairs.
public struct RecordSet: Equatable {
    public var parameters: [String : ParametersRecord]
    public var metrics: [String : MetricsRecord]
    public var configs: [String : ConfigsRecord]
    
    public init(
        parameters: [String : ParametersRecord],
        metrics: [String : MetricsRecord],
        configs: [String : ConfigsRecord]
    ) {
        self.parameters = parameters
        self.metrics = metrics
        self.configs = configs
    }
}

/// A structure that stores information about an error that occurred.
///
/// - Parameters:
///   - code: An identifier for the error.
///   - reason: An optional string that provides a reason for why the error occurred
///     (e.g., an exception stack trace).
public struct FlwrError: Equatable {
    public var code: Int64
    public var reason: String?
    
    public init(code: Int64, reason: String? = nil) {
        self.code = code
        self.reason = reason
    }
}

/// Represents the state of your application from the viewpoint of the entity using it.
///
/// - Parameters:
///   - metadata: An instance of `Metadata` containing information
///     about the message to be executed.
///   - content: An optional instance of `RecordSet` that holds records
///     either sent by another entity (e.g., sent by server-side logic
///     to a client, or vice-versa) or that will be sent to it.
///   - error: An optional instance of `Error` that captures information
///     about an error that occurred while processing another message.
public struct Message: Equatable {
    public var metadata: Metadata
    public var content: RecordSet?
    public var error: FlwrError?
    
    public init(
        metadata: Metadata,
        content: RecordSet? = nil,
        error: FlwrError? = nil
    ) throws {
        if content != nil && error != nil {
            throw FlowerException.ValueError(
                "Either `content` or `error` must be set, but not both."
            )
        }
        self.metadata = metadata
        self.content = content
        self.error = error
    }
}
