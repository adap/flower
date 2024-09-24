package dev.flower.android

import java.nio.ByteBuffer

/**
 * Represents a map of metric values.
 */
typealias Metrics = Map<String, Scalar>

/**
 * Represents a map of configuration values.
 */
typealias Config = Map<String, Scalar>

/**
 * Represents a map of properties.
 */
typealias Properties = Map<String, Scalar>



/**
 * The `Code` class defines client status codes used in the application.
 */
enum class Code(val value: Int) {
    OK(0),
    GET_PROPERTIES_NOT_IMPLEMENTED(1),
    GET_PARAMETERS_NOT_IMPLEMENTED(2),
    FIT_NOT_IMPLEMENTED(3),
    EVALUATE_NOT_IMPLEMENTED(4);

    companion object {
        fun fromInt(value: Int): Code = values().first { it.value == value }
    }
}

/**
 * Client status.
 */
data class Status(val code: Code, val message: String)

/**
 * The `Scalar` class represents a scalar value that can have different data types.
 *
 * @param <T> The type parameter specifying the data type of the scalar value. It contains types
 * corresponding to ProtoBuf types that ProtoBuf considers to be "Scalar Value Types", even though
 * some of them arguably do not conform to other definitions of what a scalar is. Source:
 * https://developers.google.com/protocol-buffers/docs/overview#scalar
 */
sealed class Scalar {
    class BoolValue(val value: Boolean) : Scalar()
    class BytesValue(val value: ByteBuffer) : Scalar()
    class SInt64Value(val value: Long) : Scalar()
    class DoubleValue(val value: Double) : Scalar()
    class StringValue(val value: String) : Scalar()
}

/**
 * Model parameters.
 */
data class Parameters(val tensors: Array<ByteBuffer>, val tensorType: String) {

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as Parameters

        if (!tensors.contentEquals(other.tensors)) return false
        if (tensorType != other.tensorType) return false

        return true
    }

    override fun hashCode(): Int {
        var result = tensors.contentHashCode()
        result = 31 * result + tensorType.hashCode()
        return result
    }
}

/**
 * Parameters request for a client.
 */
data class GetParametersIns(val config: Config)

/**
 * Response when asked to return parameters.
 */
data class GetParametersRes(val status: Status, val parameters: Parameters)

/**
 * Fit instructions for a client.
 */
data class FitIns(val parameters: Parameters, val config: Config)

/**
 * Fit response from a client.
 */
data class FitRes(val status: Status, val parameters: Parameters, val numExamples: Int, val metrics: Metrics)

/**
 * Evaluate instructions for a client.
 */
data class EvaluateIns(val parameters: Parameters, val config: Config)

/**
 * Evaluate response from a client.
 */
data class EvaluateRes(val status: Status, val loss: Float, val numExamples: Int, val metrics: Metrics)

/**
 * Properties request for a client.
 */
data class GetPropertiesIns(val config: Config)

/**
 * Properties response from a client.
 */
data class GetPropertiesRes(val status: Status, val properties: Properties)

/**
 * ReconnectIns message from server to client.
 */
data class ReconnectIns(val seconds: Long?)

/**
 * DisconnectRes message from client to server.
 */
data class DisconnectRes(val reason: String)
