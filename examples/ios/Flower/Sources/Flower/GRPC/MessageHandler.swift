//
//  MessageHandler.swift
//  FlowerSDK
//
//  Created by Daniel Nugraha on 09.01.22.
//

import Foundation

func handle(client: Client, serverMsg: Flower_Transport_ServerMessage) throws -> (Flower_Transport_ClientMessage, Int, Bool) {
    switch serverMsg.msg {
    case .reconnect:
        let tuple = reconnect(reconnectMsg: serverMsg.reconnect)
        let disconnectMsg = tuple.0
        let sleepDuration = tuple.1
        return (disconnectMsg, sleepDuration, false)
    case .getParameters:
        return (getParameters(client: client), 0, true)
    case .fitIns:
        return (fit(client: client, fitMsg: serverMsg.fitIns), 0, true)
    case .evaluateIns:
        return (evaluate(client: client, evaluateMsg: serverMsg.evaluateIns), 0, true)
    case .propertiesIns:
        return (getProperties(client: client, propertiesMsg: serverMsg.propertiesIns), 0, true)
    default:
        throw FlowerException.UnknownServerMessage
    }
}

func reconnect(reconnectMsg: Flower_Transport_ServerMessage.Reconnect) -> (Flower_Transport_ClientMessage, Int) {
    var reason: Flower_Transport_Reason = .ack
    var sleepDuration: Int = 0
    if reconnectMsg.seconds != 0 {
        reason = .reconnect
        sleepDuration = Int(reconnectMsg.seconds)
    }
    var disconnect = Flower_Transport_ClientMessage.Disconnect()
    disconnect.reason = reason
    var ret = Flower_Transport_ClientMessage()
    ret.disconnect = disconnect
    return (ret, sleepDuration)
 }

func getParameters(client: Client) -> Flower_Transport_ClientMessage {
    let parametersRes = client.getParameters()
    let parametersResProto = parametersResToProto(res: parametersRes)
    var ret = Flower_Transport_ClientMessage()
    ret.parametersRes = parametersResProto
    return ret
}

func getProperties(client: Client, propertiesMsg: Flower_Transport_ServerMessage.PropertiesIns) -> Flower_Transport_ClientMessage {
    let propertiesIns = propertiesInsFromProto(msg: propertiesMsg)
    let propertiesRes = client.getProperties(ins: propertiesIns)
    let propertiesResProto = propertiesResToProto(res: propertiesRes)
    var ret = Flower_Transport_ClientMessage()
    ret.propertiesRes = propertiesResProto
    return ret
}

func fit(client: Client, fitMsg: Flower_Transport_ServerMessage.FitIns) -> Flower_Transport_ClientMessage {
    let fitIns = fitInsFromProto(msg: fitMsg)
    let fitRes = client.fit(ins: fitIns)
    let fitResProto = fitResToProto(res: fitRes)
    var ret = Flower_Transport_ClientMessage()
    ret.fitRes = fitResProto
    return ret
}

func evaluate(client: Client, evaluateMsg: Flower_Transport_ServerMessage.EvaluateIns) -> Flower_Transport_ClientMessage {
    let evaluateIns = evaluateInsFromProto(msg: evaluateMsg)
    let evaluateRes = client.evaluate(ins: evaluateIns)
    let evaluateResProto = evaluateResToProto(res: evaluateRes)
    var ret = Flower_Transport_ClientMessage()
    ret.evaluateRes = evaluateResProto
    return ret
}
