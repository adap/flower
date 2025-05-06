package dev.flower.android

import flwr.proto.Transport.ClientMessage
import flwr.proto.Transport.ServerMessage
import flwr.proto.Transport.Reason
import flwr.proto.TaskOuterClass.TaskIns
import flwr.proto.TaskOuterClass.TaskRes
import java.lang.IllegalArgumentException


internal fun handle(client: Client, taskIns: TaskIns): Triple<TaskRes, Int, Boolean> {
    val serverMsg = getServerMessageFromTaskIns(taskIns, excludeReconnectIns = false)
        ?: throw NotImplementedError()
    val (clientMsg, sleepDuration, keepGoing) = handleLegacyMessage(client, serverMsg)
    val taskRes = wrapClientMessageInTaskRes(clientMsg)
    return Triple(taskRes, sleepDuration, keepGoing)
}

internal fun handleLegacyMessage(client: Client, serverMsg: ServerMessage): Triple<ClientMessage, Int, Boolean> {
    return when (serverMsg.msgCase.number) {
        1 -> {
            val (disconnectMsg, sleepDuration) = reconnect(serverMsg.reconnectIns)
            Triple(disconnectMsg, sleepDuration, false)
        }
        2 -> {
            Triple(getProperties(client, serverMsg.getPropertiesIns), 0, true)
        }
        3 -> {
            Triple(getParameters(client, serverMsg.getParametersIns), 0, true)
        }
        4 -> {
            Triple(fit(client, serverMsg.fitIns), 0, true)
        }
        5 -> {
            Triple(evaluate(client, serverMsg.evaluateIns), 0, true)
        }
        else -> throw IllegalArgumentException("Unknown Server Message")
    }
}

internal fun reconnect(reconnectMsg: ServerMessage.ReconnectIns): Pair<ClientMessage, Int> {
    var reason = Reason.ACK
    var sleepDuration = 0
    if (reconnectMsg.seconds > 0) {
        reason = Reason.RECONNECT
        sleepDuration = reconnectMsg.seconds.toInt()
    }
    val disconnectRes = ClientMessage.DisconnectRes.newBuilder()
        .setReason(reason)
        .build()
    return Pair(ClientMessage.newBuilder().setDisconnectRes(disconnectRes).build(), sleepDuration)
}

internal fun getProperties(client: Client, getPropertiesMsg: ServerMessage.GetPropertiesIns): ClientMessage {
    val getPropertiesIns = getPropertiesInsFromProto(getPropertiesMsg)
    val getPropertiesRes = client.getProperties(getPropertiesIns)
    return ClientMessage.newBuilder().setGetPropertiesRes(getPropertiesResToProto(getPropertiesRes)).build()
}

internal fun getParameters(client: Client, getParametersMsg: ServerMessage.GetParametersIns): ClientMessage {
    val getParametersIns = getParametersInsFromProto(getParametersMsg)
    val getParametersRes = client.getParameters(getParametersIns)
    return ClientMessage.newBuilder().setGetParametersRes(getParametersResToProto(getParametersRes)).build()
}

internal fun fit(client: Client, fitMsg: ServerMessage.FitIns): ClientMessage {
    val fitIns = fitInsFromProto(fitMsg)
    val fitRes = client.fit(fitIns)
    return ClientMessage.newBuilder().setFitRes(fitResToProto(fitRes)).build()
}

internal fun evaluate(client: Client, evaluateMsg: ServerMessage.EvaluateIns): ClientMessage {
    val evaluateIns = evaluateInsFromProto(evaluateMsg)
    val evaluateRes = client.evaluate(evaluateIns)
    return ClientMessage.newBuilder().setEvaluateRes(evaluateResToProto(evaluateRes)).build()
}
