//
//  Serde.swift
//  FlowerSDK
//
//  Created by Daniel Nugraha on 09.01.22.
//

import Foundation

func parametersToProto(parameters: Parameters) -> Flwr_Proto_Parameters {
    var ret = Flwr_Proto_Parameters()
    ret.tensors = parameters.tensors
    ret.tensorType = parameters.tensorType
    return ret
}

func parametersFromProto(msg: Flwr_Proto_Parameters) -> Parameters {
    return Parameters(tensors: msg.tensors, tensorType: msg.tensorType)
}

func statusToProto(status: Status) -> Flwr_Proto_Status {
    var ret = Flwr_Proto_Status()
    ret.code = Flwr_Proto_Code(rawValue: status.code.rawValue) ?? .UNRECOGNIZED(status.code.rawValue)
    ret.message = status.message
    return ret
}

func statusFromProto(msg: Flwr_Proto_Status) -> Status {
    return Status(code: Code(rawValue: msg.code.rawValue), message: msg.message)
}

func reconnectToProto(reconnect: Reconnect) -> Flwr_Proto_ServerMessage.ReconnectIns {
    if let seconds = reconnect.seconds {
        var ret = Flwr_Proto_ServerMessage.ReconnectIns()
        ret.seconds = Int64(seconds)
        return ret
    }
    return Flwr_Proto_ServerMessage.ReconnectIns()
}

func reconnectFromProto(msg: Flwr_Proto_ServerMessage.ReconnectIns) -> Reconnect {
    return Reconnect(seconds: Int(msg.seconds))
}

func disconnectToProto(disconnect: Disconnect) -> Flwr_Proto_ClientMessage.DisconnectRes {
    var reason: Flwr_Proto_Reason = .unknown
    switch disconnect.reason {
    case "RECONNECT":
        reason = .reconnect
    case "POWER_DISCONNECTED":
        reason = .powerDisconnected
    case "WIFI_UNAVAILABLE":
        reason = .wifiUnavailable
    default:
        reason = .unknown
    }
    var ret = Flwr_Proto_ClientMessage.DisconnectRes()
    ret.reason = reason
    return ret
}

func disconnectFromProto(msg: Flwr_Proto_ClientMessage.DisconnectRes) -> Disconnect {
    switch msg.reason {
    case .wifiUnavailable:
        return Disconnect(reason: "WIFI_UNAVAILABLE")
    case .powerDisconnected:
        return Disconnect(reason: "POWER_DISCONNECTED")
    case .reconnect:
        return Disconnect(reason: "RECONNECT")
    default:
        return Disconnect(reason: "UNKNOWN")
    }
}

func getParametersToProto() -> Flwr_Proto_ServerMessage.GetParametersIns {
    return Flwr_Proto_ServerMessage.GetParametersIns()
}

func parametersResToProto(res: GetParametersRes) -> Flwr_Proto_ClientMessage.GetParametersRes {
    let parametersProto = parametersToProto(parameters: res.parameters)
    var ret = Flwr_Proto_ClientMessage.GetParametersRes()
    ret.parameters = parametersProto
    return ret
}

func parametersResFromProto(msg: Flwr_Proto_ClientMessage.GetParametersRes) -> GetParametersRes {
    let parameters = parametersFromProto(msg: msg.parameters)
    let status = statusFromProto(msg: msg.status)
    return GetParametersRes(parameters: parameters, status: status)
}

func fitInsToProto(ins: FitIns) -> Flwr_Proto_ServerMessage.FitIns {
    let parametersProto = parametersToProto(parameters: ins.parameters)
    let configMsg = metricsToProto(metrics: ins.config)
    var ret = Flwr_Proto_ServerMessage.FitIns()
    ret.parameters = parametersProto
    ret.config = configMsg
    return ret
}

func fitInsFromProto(msg: Flwr_Proto_ServerMessage.FitIns) -> FitIns {
    let parameters = parametersFromProto(msg: msg.parameters)
    let config = metricsFromProto(proto: msg.config)
    return FitIns(parameters: parameters, config: config)
}

func fitResToProto(res: FitRes) -> Flwr_Proto_ClientMessage.FitRes {
    var ret = Flwr_Proto_ClientMessage.FitRes()
    let parameters = parametersToProto(parameters: res.parameters)
    if let metrics = res.metrics {
       ret.metrics = metricsToProto(metrics: metrics)
    }
    ret.parameters = parameters
    ret.numExamples = Int64(res.numExamples)
    return ret
}

func fitResFromProto(msg: Flwr_Proto_ClientMessage.FitRes) -> FitRes {
    let parameters = parametersFromProto(msg: msg.parameters)
    let metrics = metricsFromProto(proto: msg.metrics)
    let status = statusFromProto(msg: msg.status)
    return FitRes(parameters: parameters, numExamples: Int(msg.numExamples), metrics: metrics, status: status)
}

func propertiesInsToProto(ins: GetPropertiesIns) -> Flwr_Proto_ServerMessage.GetPropertiesIns {
    var ret = Flwr_Proto_ServerMessage.GetPropertiesIns()
    let config = propertiesToProto(properties: ins.config)
    ret.config = config
    return ret
}

func propertiesInsFromProto(msg: Flwr_Proto_ServerMessage.GetPropertiesIns) -> GetPropertiesIns {
    let config = propertiesFromProto(proto: msg.config)
    return GetPropertiesIns(config: config)
}

func propertiesResToProto(res: GetPropertiesRes) -> Flwr_Proto_ClientMessage.GetPropertiesRes {
    let properties = propertiesToProto(properties: res.properties)
    var ret = Flwr_Proto_ClientMessage.GetPropertiesRes()
    ret.properties = properties
    return ret
}

func propertiesResFromProto(msg: Flwr_Proto_ClientMessage.GetPropertiesRes) -> GetPropertiesRes {
    let properties = propertiesFromProto(proto: msg.properties)
    let status = statusFromProto(msg: msg.status)
    return GetPropertiesRes(properties: properties, status: status)
}

func evaluateInsToProto(ins: EvaluateIns) -> Flwr_Proto_ServerMessage.EvaluateIns {
    let parametersProto = parametersToProto(parameters: ins.parameters)
    let configMsg = metricsToProto(metrics: ins.config)
    var ret = Flwr_Proto_ServerMessage.EvaluateIns()
    ret.config = configMsg
    ret.parameters = parametersProto
    return ret
}

func evaluateInsFromProto(msg: Flwr_Proto_ServerMessage.EvaluateIns) -> EvaluateIns {
    let parameters = parametersFromProto(msg: msg.parameters)
    let config = metricsFromProto(proto: msg.config)
    return EvaluateIns(parameters: parameters, config: config)
}

func evaluateResToProto(res: EvaluateRes) -> Flwr_Proto_ClientMessage.EvaluateRes {
    var ret = Flwr_Proto_ClientMessage.EvaluateRes()
    if let metrics = res.metrics {
        ret.metrics = metricsToProto(metrics: metrics)
    }
    ret.loss = res.loss
    ret.numExamples = Int64(res.numExamples)
    return ret
}

func evaluateResFromProto(msg: Flwr_Proto_ClientMessage.EvaluateRes) -> EvaluateRes {
    let status = statusFromProto(msg: msg.status)
    return EvaluateRes(loss: msg.loss, numExamples: Int(msg.numExamples), metrics: metricsFromProto(proto: msg.metrics), status: status)
}

func propertiesToProto(properties: Properties) -> [String: Flwr_Proto_Scalar] {
    var proto: [String: Flwr_Proto_Scalar] = [:]
    for (key, value) in properties {
        if let scalar = try? scalarToProto(scalar: value) {
            proto[key] = scalar
        }
    }
    return proto
}

func propertiesFromProto(proto: [String: Flwr_Proto_Scalar]) -> Properties {
    var properties: Properties = [:]
    for (key, value) in proto {
        if let scalarMsg = try? scalarFromProto(scalarMsg: value) {
            properties[key] = scalarMsg
        }
    }
    return properties
}

func metricsToProto(metrics: Metrics) -> [String: Flwr_Proto_Scalar] {
    var proto: [String: Flwr_Proto_Scalar] = [:]
    for (key, value) in metrics {
        if let scalar = try? scalarToProto(scalar: value) {
            proto[key] = scalar
        }
    }
    return proto
}

func metricsFromProto(proto: [String: Flwr_Proto_Scalar]) -> Metrics {
    var metrics: Metrics = [:]
    for (key, value) in proto {
        if let scalarMsg = try? scalarFromProto(scalarMsg: value) {
            metrics[key] = scalarMsg
        }
    }
    return metrics
}

func scalarToProto(scalar: Scalar) throws -> Flwr_Proto_Scalar {
    var ret = Flwr_Proto_Scalar()
    if let bool = scalar.bool {
        ret.bool = bool
    } else if let bytes = scalar.bytes {
        ret.bytes = bytes
    } else if let int = scalar.int {
        ret.sint64 = Int64(int)
    } else if let float = scalar.float {
        ret.double = Double(float)
    } else if let str = scalar.str {
        ret.string = str
    } else {
        throw FlowerException.TypeError("Accepted Types : Bool, Data, Float, Int, Str")
    }
    return ret
}

func scalarFromProto(scalarMsg: Flwr_Proto_Scalar) throws -> Scalar {
    var ret = Scalar()
    switch scalarMsg.scalar {
    case .double:
        ret.float = Float(scalarMsg.double)
    case .bool:
        ret.bool = scalarMsg.bool
    case .bytes:
        ret.bytes = scalarMsg.bytes
    case .sint64:
        ret.int = Int(scalarMsg.sint64)
    case.string:
        ret.str = scalarMsg.string
    default:
        throw FlowerException.TypeError("Accepted Types : Bool, Data, Float, Int, Str")
    }
    return ret
}

extension Array<Double> {
    internal func arrayToProto() -> Flwr_Proto_DoubleList {
        Flwr_Proto_DoubleList.with { list in
            list.vals = self
        }
    }
}

extension Array<Int64> {
    internal func arrayToProto() -> Flwr_Proto_Sint64List {
        Flwr_Proto_Sint64List.with { list in
            list.vals = self
        }
    }
}

extension Array<Bool> {
    internal func arrayToProto() -> Flwr_Proto_BoolList {
        Flwr_Proto_BoolList.with { list in
            list.vals = self
        }
    }
}

extension Array<String> {
    internal func arrayToProto() -> Flwr_Proto_StringList {
        Flwr_Proto_StringList.with { list in
            list.vals = self
        }
    }
}

extension Array<Data> {
    internal func arrayToProto() -> Flwr_Proto_BytesList {
        Flwr_Proto_BytesList.with { list in
            list.vals = self
        }
    }
}

extension Flwr_Proto_DoubleList {
    internal func arrayFromProto() -> [Double] {
        self.vals
    }
}

extension Flwr_Proto_Sint64List {
    internal func arrayFromProto() -> [Int64] {
        self.vals
    }
}

extension Flwr_Proto_BoolList {
    internal func arrayFromProto() -> [Bool] {
        self.vals
    }
}

extension Flwr_Proto_StringList {
    internal func arrayFromProto() -> [String] {
        self.vals
    }
}

extension Flwr_Proto_BytesList {
    internal func arrayFromProto() -> [Data] {
        self.vals
    }
}

extension Flwr_Proto_ConfigsRecordValue {
    internal func configsRecordValueFromProto() throws -> ConfigsRecordValue {
        switch self.value {
        case let .double(val):
            return .double(val)
        case let .sint64(val):
            return .sint64(val)
        case let .bool(val):
            return .bool(val)
        case let .string(val):
            return .string(val)
        case let .bytes(val):
            return .bytes(val)
        case let .doubleList(list):
            return .doubleList(list.arrayFromProto())
        case let .sint64List(list):
            return .sint64List(list.arrayFromProto())
        case let .boolList(list):
            return .boolList(list.arrayFromProto())
        case let .stringList(list):
            return .stringList(list.arrayFromProto())
        case let .bytesList(list):
            return .bytesList(list.arrayFromProto())
        case .none:
            throw FlowerException.ValueError("Flwr_Proto_ConfigsRecordValue is none")
        }
    }
}

extension Flwr_Proto_MetricsRecordValue {
    internal func metricsRecordValueFromProto() throws -> MetricsRecordValue {
        switch self.value {
        case let .double(val):
            // Return the associated Double value
            return .double(val)
        case let .sint64(val):
            // Return the associated Int64 value
            return .sint64(val)
        case let .doubleList(list):
            // Return the associated Flwr_Proto_DoubleList value
            return .doubleList(list.arrayFromProto())
        case let .sint64List(list):
            // Return the associated Flwr_Proto_Sint64List value
            return .sint64List((list.arrayFromProto()))
        case .none:
            throw FlowerException.ValueError("Flwr_Proto_MetricsRecordValue is none")
        }
    }
}

extension Flwr_Proto_Array {
    internal func multiArrayFromProto() -> MultiArray {
        MultiArray(dtype: self.dtype, shape: self.shape, stype: self.stype, data: self.data)
    }
}

extension Flwr_Proto_ParametersRecord {
    internal func parametersRecordFromProto() -> ParametersRecord {
        ParametersRecord(dataKeys: self.dataKeys, dataValues: self.dataValues.map { $0.multiArrayFromProto() })
    }
}

extension Flwr_Proto_MetricsRecord {
    internal func metricsRecordFromProto() -> MetricsRecord {
        MetricsRecord(data: self.data.compactMapValues { try? $0.metricsRecordValueFromProto() })
    }
}

extension Flwr_Proto_ConfigsRecord {
    internal func configsRecordFromProto() -> ConfigsRecord {
        ConfigsRecord(data: self.data.compactMapValues { try? $0.configsRecordValueFromProto() })
    }
}

extension Flwr_Proto_RecordSet {
    internal func recordSetFromProto() -> RecordSet {
        RecordSet(
            parameters: self.parameters.compactMapValues { $0.parametersRecordFromProto() },
            metrics: self.metrics.compactMapValues { $0.metricsRecordFromProto() },
            configs: self.configs.compactMapValues { $0.configsRecordFromProto() }
        )
    }
}

extension Flwr_Proto_Error {
    internal func errorFromProto() -> FlwrError {
        FlwrError(code: self.code, reason: self.reason)
    }
}

extension Flwr_Proto_TaskIns {
    internal func messageFromTaskInsProto() throws -> Message {
        let metadata = Metadata(
            runID: self.runID,
            messageID: self.taskID,
            srcNodeID: self.task.producer.nodeID,
            dstNodeID: self.task.consumer.nodeID,
            replyToMessage: self.task.ancestry.first ?? "",
            groupID: self.groupID,
            ttl: self.task.ttl,
            messageType: self.task.taskType
        )

        return try Message(
            metadata: metadata,
            content: self.task.recordset.recordSetFromProto(),
            error: self.task.error.errorFromProto()
        )
    }
}

extension ConfigsRecordValue {
    internal func configsRecordValueToProto() -> Flwr_Proto_ConfigsRecordValue {
        Flwr_Proto_ConfigsRecordValue.with {
            switch self {
            case let .double(val):
                $0.double = val
            case let .sint64(val):
                $0.sint64 = val
            case let .bool(val):
                $0.bool = val
            case let .string(val):
                $0.string = val
            case let .bytes(val):
                $0.bytes = val
            case let .doubleList(list):
                $0.doubleList = list.arrayToProto()
            case let .sint64List(list):
                $0.sint64List = list.arrayToProto()
            case let .boolList(list):
                $0.boolList = list.arrayToProto()
            case let .stringList(list):
                $0.stringList = list.arrayToProto()
            case let .bytesList(list):
                $0.bytesList = list.arrayToProto()
            }
        }
    }
}

extension MetricsRecordValue {
    internal func metricsRecordValueToProto() -> Flwr_Proto_MetricsRecordValue {
        Flwr_Proto_MetricsRecordValue.with {
            switch self {
            case let .double(val):
                $0.double = val
            case let .sint64(val):
                $0.sint64 = val
            case let .doubleList(list):
                $0.doubleList = list.arrayToProto()
            case let .sint64List(list):
                $0.sint64List = list.arrayToProto()
            }
        }
    }
}

extension MultiArray {
    internal func multiArrayToProto() -> Flwr_Proto_Array {
        Flwr_Proto_Array.with {
            $0.dtype = self.dtype
            $0.shape = self.shape
            $0.stype = self.stype
            $0.data = self.data
        }
    }
}

extension ParametersRecord {
    internal func parametersRecordToProto() -> Flwr_Proto_ParametersRecord {
        Flwr_Proto_ParametersRecord.with {
            $0.dataKeys = self.dataKeys
            $0.dataValues = self.dataValues.map { $0.multiArrayToProto() }
        }
    }
}

extension MetricsRecord {
    internal func metricsRecordToProto() -> Flwr_Proto_MetricsRecord {
        Flwr_Proto_MetricsRecord.with {
            $0.data = self.data.compactMapValues { $0.metricsRecordValueToProto() }
        }
    }
}

extension ConfigsRecord {
    internal func configsRecordToProto() -> Flwr_Proto_ConfigsRecord {
        Flwr_Proto_ConfigsRecord.with {
            $0.data = self.data.compactMapValues { $0.configsRecordValueToProto() }
        }
    }
}

extension RecordSet {
    internal func recordSetToProto() -> Flwr_Proto_RecordSet {
        Flwr_Proto_RecordSet.with {
            $0.parameters = self.parameters.compactMapValues { $0.parametersRecordToProto() }
            $0.metrics = self.metrics.compactMapValues { $0.metricsRecordToProto() }
            $0.configs = self.configs.compactMapValues { $0.configsRecordToProto() }
        }
    }
}

extension FlwrError {
    internal func errorToProto() -> Flwr_Proto_Error {
        Flwr_Proto_Error.with {
            $0.code = self.code
            $0.reason = self.reason ?? ""
        }
    }
}

extension Message {
    internal func messageToTaskResProto() -> Flwr_Proto_TaskRes {
        Flwr_Proto_TaskRes.with {
            $0.taskID = ""
            $0.groupID = self.metadata.groupID
            $0.runID = self.metadata.runID
            $0.task = Flwr_Proto_Task.with {
                $0.producer = Flwr_Proto_Node.with {
                    $0.nodeID = self.metadata.srcNodeID
                    $0.anonymous = false
                }
                $0.consumer = Flwr_Proto_Node.with {
                    $0.nodeID = 0
                    $0.anonymous = false
                }
                $0.createdAt = self.metadata.createdAt
                $0.ttl = self.metadata.ttl
                $0.ancestry = [self.metadata.replyToMessage]
                $0.taskType = self.metadata.messageType
                
                if let recordset = self.content {
                    $0.recordset = recordset.recordSetToProto()
                }
                
                if let error = self.error {
                    $0.error = error.errorToProto()
                }
            }
        }
    }
}
