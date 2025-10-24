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
        throw FlowerException.TypeException("Accepted Types : Bool, Data, Float, Int, Str")
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
        throw FlowerException.TypeException("Accepted Types : Bool, Data, Float, Int, Str")
    }
    return ret
}
