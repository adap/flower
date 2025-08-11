//
//  MessageHandler.swift
//  FlowerSDK
//
//  Created by Daniel Nugraha on 09.01.22.
//

import Foundation

/// Handles the incoming messages from the server
///
/// - Parameters:
///   - client: The implementation of the Client containing executable routines based on server instructions
///   - serverMsg: The server's message parsed as a Flwr_Proto_ServerMessage struct.
/// - Returns:
///   - The client's answer after proofing or executing the server's instructions
func handle(client: Client, serverMsg: Flwr_Proto_ServerMessage) throws -> (Flwr_Proto_ClientMessage, Int, Bool) {
    switch serverMsg.msg {
    case .reconnectIns:
        let tuple = reconnect(reconnectMsg: serverMsg.reconnectIns)
        let disconnectMsg = tuple.0
        let sleepDuration = tuple.1
        return (disconnectMsg, sleepDuration, false)
    case .getParametersIns:
        return (getParameters(client: client), 0, true)
    case .fitIns:
        return (fit(client: client, fitMsg: serverMsg.fitIns), 0, true)
    case .evaluateIns:
        return (evaluate(client: client, evaluateMsg: serverMsg.evaluateIns), 0, true)
    case .getPropertiesIns:
        return (getProperties(client: client, propertiesMsg: serverMsg.getPropertiesIns), 0, true)
    default:
        throw FlowerException.UnknownServerMessage
    }
}

/// Handle for the server message that initiates a reconnection.
func reconnect(reconnectMsg: Flwr_Proto_ServerMessage.ReconnectIns) -> (Flwr_Proto_ClientMessage, Int) {
    var reason: Flwr_Proto_Reason = .ack
    var sleepDuration: Int = 0
    if reconnectMsg.seconds != 0 {
        reason = .reconnect
        sleepDuration = Int(reconnectMsg.seconds)
    }
    var disconnect = Flwr_Proto_ClientMessage.DisconnectRes()
    disconnect.reason = reason
    var ret = Flwr_Proto_ClientMessage()
    ret.disconnectRes = disconnect
    return (ret, sleepDuration)
 }

/// Handle for the server message that requests the local model parameters.
func getParameters(client: Client) -> Flwr_Proto_ClientMessage {
    let parametersRes = client.getParameters()
    let parametersResProto = parametersResToProto(res: parametersRes)
    var ret = Flwr_Proto_ClientMessage()
    ret.getParametersRes = parametersResProto
    return ret
}

/// Handle for the server message that requests the local model properties.
func getProperties(client: Client, propertiesMsg: Flwr_Proto_ServerMessage.GetPropertiesIns) -> Flwr_Proto_ClientMessage {
    let propertiesIns = propertiesInsFromProto(msg: propertiesMsg)
    let propertiesRes = client.getProperties(ins: propertiesIns)
    let propertiesResProto = propertiesResToProto(res: propertiesRes)
    var ret = Flwr_Proto_ClientMessage()
    ret.getPropertiesRes = propertiesResProto
    return ret
}

/// Handle for the server message that instructs to optimize the local model.
func fit(client: Client, fitMsg: Flwr_Proto_ServerMessage.FitIns) -> Flwr_Proto_ClientMessage {
    let fitIns = fitInsFromProto(msg: fitMsg)
    let fitRes = client.fit(ins: fitIns)
    let fitResProto = fitResToProto(res: fitRes)
    var ret = Flwr_Proto_ClientMessage()
    ret.fitRes = fitResProto
    return ret
}

/// Handle for the server message that instructs to evaluate the local model.
func evaluate(client: Client, evaluateMsg: Flwr_Proto_ServerMessage.EvaluateIns) -> Flwr_Proto_ClientMessage {
    let evaluateIns = evaluateInsFromProto(msg: evaluateMsg)
    let evaluateRes = client.evaluate(ins: evaluateIns)
    let evaluateResProto = evaluateResToProto(res: evaluateRes)
    var ret = Flwr_Proto_ClientMessage()
    ret.evaluateRes = evaluateResProto
    return ret
}
