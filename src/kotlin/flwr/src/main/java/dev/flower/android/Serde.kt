package dev.flower.android

import com.google.protobuf.ByteString
import flwr.proto.Transport.ClientMessage
import flwr.proto.Transport.Reason
import flwr.proto.Transport.ServerMessage
import java.nio.ByteBuffer
import flwr.proto.Transport.Parameters as ProtoParameters
import flwr.proto.Transport.Scalar as ProtoScalar
import flwr.proto.Transport.Status as ProtoStatus

internal fun parametersToProto(parameters: Parameters): ProtoParameters {
    val tensors: MutableList<ByteString> = ArrayList()
    for (tensor in parameters.tensors) {
        tensors.add(ByteString.copyFrom(tensor))
    }
    return ProtoParameters.newBuilder().addAllTensors(tensors).setTensorType(parameters.tensorType).build()
}

internal fun parametersFromProto(msg: ProtoParameters): Parameters {
    return Parameters(msg.tensorsList.map { ByteBuffer.wrap(it.toByteArray()) }.toTypedArray(), msg.tensorType)
}

internal fun statusToProto(status: Status): ProtoStatus {
    return ProtoStatus.newBuilder().setCodeValue(status.code.value).setMessage(status.message).build()
}

internal fun statusFromProto(msg: ProtoStatus): Status {
    return Status(Code.fromInt(msg.codeValue), msg.message)
}

internal fun reconnectInsToProto(ins: ReconnectIns): ServerMessage.ReconnectIns {
    ins.seconds?.let { value ->
        return ServerMessage.ReconnectIns.newBuilder().setSeconds(value).build()
    }
    return ServerMessage.ReconnectIns.newBuilder().build()
}

internal fun reconnectInsFromProto(msg: ServerMessage.ReconnectIns): ReconnectIns {
    return ReconnectIns(msg.seconds)
}

internal fun disconnectResToProto(res: DisconnectRes): ClientMessage.DisconnectRes {
    val reason = when (res.reason) {
        "RECONNECT" -> Reason.RECONNECT
        "POWER_DISCONNECTED" -> Reason.POWER_DISCONNECTED
        "WIFI_UNAVAILABLE" -> Reason.WIFI_UNAVAILABLE
        else -> Reason.UNKNOWN
    }
    return ClientMessage.DisconnectRes.newBuilder().setReason(reason).build()
}

internal fun disconnectResFromProto(msg: ClientMessage.DisconnectRes): DisconnectRes {
    val reason = when (msg.reason) {
        Reason.RECONNECT -> "RECONNECT"
        Reason.POWER_DISCONNECTED -> "POWER_DISCONNECTED"
        Reason.WIFI_UNAVAILABLE -> "WIFI_UNAVAILABLE"
        else -> "UNKNOWN"
    }
    return DisconnectRes(reason)
}

internal fun getParametersInsFromProto(msg: ServerMessage.GetParametersIns): GetParametersIns {
    return GetParametersIns(metricsFromProto(msg.configMap))
}

internal fun getParametersInsToProto(ins: GetParametersIns): ServerMessage.GetParametersIns {
    return ServerMessage.GetParametersIns.newBuilder().putAllConfig(metricsToProto(ins.config)).build()
}

internal fun getParametersResToProto(res: GetParametersRes): ClientMessage.GetParametersRes {
    val parametersProto = parametersToProto(res.parameters)
    return ClientMessage.GetParametersRes.newBuilder().setParameters(parametersProto).build()
}

internal fun getParametersResFromProto(msg: ClientMessage.GetParametersRes): GetParametersRes {
    return GetParametersRes(statusFromProto(msg.status), parametersFromProto(msg.parameters))
}

internal fun fitInsToProto(ins: FitIns): ServerMessage.FitIns {
    return ServerMessage.FitIns
        .newBuilder()
        .setParameters(parametersToProto(ins.parameters))
        .putAllConfig(metricsToProto(ins.config))
        .build()
}

internal fun fitInsFromProto(msg: ServerMessage.FitIns): FitIns {
    return FitIns(parametersFromProto(msg.parameters), metricsFromProto(msg.configMap))
}

internal fun fitResToProto(res: FitRes): ClientMessage.FitRes {
    return ClientMessage.FitRes
        .newBuilder()
        .setParameters(parametersToProto(res.parameters))
        .putAllMetrics(metricsToProto(res.metrics))
        .setNumExamples(res.numExamples.toLong())
        .setStatus(statusToProto(res.status))
        .build()
}

internal fun fitResFromProto(msg: ClientMessage.FitRes) : FitRes {
    return FitRes(
        statusFromProto(msg.status),
        parametersFromProto(msg.parameters),
        msg.numExamples.toInt(),
        metricsFromProto(msg.metricsMap)
    )
}

internal fun getPropertiesInsToProto(ins: GetPropertiesIns): ServerMessage.GetPropertiesIns {
    return ServerMessage.GetPropertiesIns
        .newBuilder()
        .putAllConfig(propertiesToProto(ins.config))
        .build()
}

internal fun getPropertiesInsFromProto(msg: ServerMessage.GetPropertiesIns): GetPropertiesIns {
    return GetPropertiesIns(propertiesFromProto(msg.configMap))
}

internal fun getPropertiesResToProto(res: GetPropertiesRes) : ClientMessage.GetPropertiesRes {
    return ClientMessage.GetPropertiesRes
        .newBuilder()
        .putAllProperties(propertiesToProto(res.properties))
        .setStatus(statusToProto(res.status))
        .build()
}

internal fun getPropertiesResFromProto(msg: ClientMessage.GetPropertiesRes): GetPropertiesRes {
    return GetPropertiesRes(statusFromProto(msg.status), propertiesFromProto(msg.propertiesMap))
}

internal fun evaluateInsToProto(ins: EvaluateIns): ServerMessage.EvaluateIns {
    return ServerMessage.EvaluateIns
        .newBuilder()
        .setParameters(parametersToProto(ins.parameters))
        .putAllConfig(metricsToProto(ins.config))
        .build()
}

internal fun evaluateInsFromProto(msg: ServerMessage.EvaluateIns): EvaluateIns {
    return EvaluateIns(parametersFromProto(msg.parameters), metricsFromProto(msg.configMap))
}

internal fun evaluateResToProto(res: EvaluateRes): ClientMessage.EvaluateRes {
    return ClientMessage.EvaluateRes
        .newBuilder()
        .putAllMetrics(metricsToProto(res.metrics))
        .setLoss(res.loss)
        .setStatus(statusToProto(res.status))
        .setNumExamples(res.numExamples.toLong())
        .build()
}

internal fun evaluateResFromProto(msg: ClientMessage.EvaluateRes): EvaluateRes {
    return EvaluateRes(statusFromProto(msg.status), msg.loss, msg.numExamples.toInt(), metricsFromProto(msg.metricsMap))
}

internal fun propertiesToProto(properties: Properties): Map<String, ProtoScalar> {
    return properties.mapValues { (_, value) -> scalarToProto(value) }
}

internal fun propertiesFromProto(proto: Map<String, ProtoScalar>): Properties {
    return proto.mapValues { (_, value) -> scalarFromProto(value) }
}

internal fun metricsToProto(metrics: Metrics): Map<String, ProtoScalar> {
    return metrics.mapValues { (_, value) -> scalarToProto(value) }
}

internal fun metricsFromProto(proto: Map<String, ProtoScalar>): Metrics {
    return proto.mapValues { (_, value) -> scalarFromProto(value) }
}

internal fun scalarToProto(scalar: Scalar): ProtoScalar {
    return when (scalar) {
        is Scalar.BoolValue -> ProtoScalar.newBuilder().setBool(scalar.value).build()
        is Scalar.BytesValue -> ProtoScalar.newBuilder().setBytes(ByteString.copyFrom(scalar.value)).build()
        is Scalar.SInt64Value -> ProtoScalar.newBuilder().setSint64(scalar.value).build()
        is Scalar.DoubleValue -> ProtoScalar.newBuilder().setDouble(scalar.value).build()
        is Scalar.StringValue -> ProtoScalar.newBuilder().setString(scalar.value).build()
    }
}

internal fun scalarFromProto(scalarMsg: ProtoScalar): Scalar {
    return when (scalarMsg.scalarCase) {
        ProtoScalar.ScalarCase.BOOL -> Scalar.BoolValue(scalarMsg.bool)
        ProtoScalar.ScalarCase.BYTES -> Scalar.BytesValue(ByteBuffer.wrap(scalarMsg.bytes.toByteArray()))
        ProtoScalar.ScalarCase.DOUBLE -> Scalar.DoubleValue(scalarMsg.double)
        ProtoScalar.ScalarCase.SINT64 -> Scalar.SInt64Value(scalarMsg.sint64)
        ProtoScalar.ScalarCase.STRING -> Scalar.StringValue(scalarMsg.string)
        else -> throw IllegalArgumentException("Accepted Types : Bool, Data, Float, Int, Str")
    }
}
