//
//  Serde.swift
//  FlowerSDK
//
//  Created by Daniel Nugraha on 09.01.22.
//

import Foundation

func parametersToProto(parameters: Parameters) -> Flower_Transport_Parameters {
    var ret = Flower_Transport_Parameters()
    ret.tensors = parameters.tensors
    ret.tensorType = parameters.tensorType
    return ret
}

func parametersFromProto(msg: Flower_Transport_Parameters) -> Parameters {
    return Parameters(tensors: msg.tensors, tensorType: msg.tensorType)
}

func reconnectToProto(reconnect: Reconnect) -> Flower_Transport_ServerMessage.Reconnect {
    if let seconds = reconnect.seconds {
        var ret = Flower_Transport_ServerMessage.Reconnect()
        ret.seconds = Int64(seconds)
        return ret
    }
    return Flower_Transport_ServerMessage.Reconnect()
}

func reconnectFromProto(msg: Flower_Transport_ServerMessage.Reconnect) -> Reconnect {
    return Reconnect(seconds: Int(msg.seconds))
}

func disconnectToProto(disconnect: Disconnect) -> Flower_Transport_ClientMessage.Disconnect {
    var reason: Flower_Transport_Reason = .unknown
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
    var ret = Flower_Transport_ClientMessage.Disconnect()
    ret.reason = reason
    return ret
}

func disconnectFromProto(msg: Flower_Transport_ClientMessage.Disconnect) -> Disconnect {
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

func getParametersToProto() -> Flower_Transport_ServerMessage.GetParameters {
    return Flower_Transport_ServerMessage.GetParameters()
}

func parametersResToProto(res: ParametersRes) -> Flower_Transport_ClientMessage.ParametersRes {
    let parametersProto = parametersToProto(parameters: res.parameters)
    var ret = Flower_Transport_ClientMessage.ParametersRes()
    ret.parameters = parametersProto
    return ret
}

func parametersResFromProto(msg: Flower_Transport_ClientMessage.ParametersRes) -> ParametersRes {
    let parameters = parametersFromProto(msg: msg.parameters)
    return ParametersRes(parameters: parameters)
}

func fitInsToProto(ins: FitIns) -> Flower_Transport_ServerMessage.FitIns {
    let parametersProto = parametersToProto(parameters: ins.parameters)
    let configMsg = metricsToProto(metrics: ins.config)
    var ret = Flower_Transport_ServerMessage.FitIns()
    ret.parameters = parametersProto
    ret.config = configMsg
    return ret
}

func fitInsFromProto(msg: Flower_Transport_ServerMessage.FitIns) -> FitIns {
    let parameters = parametersFromProto(msg: msg.parameters)
    let config = metricsFromProto(proto: msg.config)
    return FitIns(parameters: parameters, config: config)
}

func fitResToProto(res: FitRes) -> Flower_Transport_ClientMessage.FitRes {
    var ret = Flower_Transport_ClientMessage.FitRes()
    let parameters = parametersToProto(parameters: res.parameters)
    if let metrics = res.metrics {
       ret.metrics = metricsToProto(metrics: metrics)
    }
    ret.parameters = parameters
    ret.numExamples = Int64(res.numExamples)
    return ret
}

func fitResFromProto(msg: Flower_Transport_ClientMessage.FitRes) -> FitRes {
    let parameters = parametersFromProto(msg: msg.parameters)
    let metrics = metricsFromProto(proto: msg.metrics)
    return FitRes(parameters: parameters, numExamples: Int(msg.numExamples), metrics: metrics)
}

func propertiesInsToProto(ins: PropertiesIns) -> Flower_Transport_ServerMessage.PropertiesIns {
    var ret = Flower_Transport_ServerMessage.PropertiesIns()
    let config = propertiesToProto(properties: ins.config)
    ret.config = config
    return ret
}

func propertiesInsFromProto(msg: Flower_Transport_ServerMessage.PropertiesIns) -> PropertiesIns {
    let config = propertiesFromProto(proto: msg.config)
    return PropertiesIns(config: config)
}

func propertiesResToProto(res: PropertiesRes) -> Flower_Transport_ClientMessage.PropertiesRes {
    let properties = propertiesToProto(properties: res.properties)
    var ret = Flower_Transport_ClientMessage.PropertiesRes()
    ret.properties = properties
    return ret
}

func propertiesResFromProto(msg: Flower_Transport_ClientMessage.PropertiesRes) -> PropertiesRes {
    let properties = propertiesFromProto(proto: msg.properties)
    return PropertiesRes(properties: properties)
}

func evaluateInsToProto(ins: EvaluateIns) -> Flower_Transport_ServerMessage.EvaluateIns {
    let parametersProto = parametersToProto(parameters: ins.parameters)
    let configMsg = metricsToProto(metrics: ins.config)
    var ret = Flower_Transport_ServerMessage.EvaluateIns()
    ret.config = configMsg
    ret.parameters = parametersProto
    return ret
}

func evaluateInsFromProto(msg: Flower_Transport_ServerMessage.EvaluateIns) -> EvaluateIns {
    let parameters = parametersFromProto(msg: msg.parameters)
    let config = metricsFromProto(proto: msg.config)
    return EvaluateIns(parameters: parameters, config: config)
}

func evaluateResToProto(res: EvaluateRes) -> Flower_Transport_ClientMessage.EvaluateRes {
    var ret = Flower_Transport_ClientMessage.EvaluateRes()
    if let metrics = res.metrics {
        ret.metrics = metricsToProto(metrics: metrics)
    }
    ret.loss = res.loss
    ret.numExamples = Int64(res.numExamples)
    return ret
}

func evaluateResFromProto(msg: Flower_Transport_ClientMessage.EvaluateRes) -> EvaluateRes {
    return EvaluateRes(loss: msg.loss, numExamples: Int(msg.numExamples), metrics: metricsFromProto(proto: msg.metrics))
}

func propertiesToProto(properties: Properties) -> [String: Flower_Transport_Scalar] {
    var proto: [String: Flower_Transport_Scalar] = [:]
    for (key, value) in properties {
        if let scalar = try? scalarToProto(scalar: value) {
            proto[key] = scalar
        }
    }
    return proto
}

func propertiesFromProto(proto: [String: Flower_Transport_Scalar]) -> Properties {
    var properties: Properties = [:]
    for (key, value) in proto {
        if let scalarMsg = try? scalarFromProto(scalarMsg: value) {
            properties[key] = scalarMsg
        }
    }
    return properties
}

func metricsToProto(metrics: Metrics) -> [String: Flower_Transport_Scalar] {
    var proto: [String: Flower_Transport_Scalar] = [:]
    for (key, value) in metrics {
        if let scalar = try? scalarToProto(scalar: value) {
            proto[key] = scalar
        }
    }
    return proto
}

func metricsFromProto(proto: [String: Flower_Transport_Scalar]) -> Metrics {
    var metrics: Metrics = [:]
    for (key, value) in proto {
        if let scalarMsg = try? scalarFromProto(scalarMsg: value) {
            metrics[key] = scalarMsg
        }
    }
    return metrics
}

func scalarToProto(scalar: Scalar) throws -> Flower_Transport_Scalar {
    var ret = Flower_Transport_Scalar()
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
    }
    throw FlowerException.TypeException("Accepted Types : Bool, Data, Float, Int, Str")
}

func scalarFromProto(scalarMsg: Flower_Transport_Scalar) throws -> Scalar {
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
