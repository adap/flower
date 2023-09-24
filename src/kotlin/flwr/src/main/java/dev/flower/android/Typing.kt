package dev.flower.android

import androidx.annotation.IntDef
import com.google.protobuf.ByteString
import java.nio.ByteBuffer

/**
 * Represents a map of metric values.
 */
typealias Metrics = Map<String, Scalar<Any>>

/**
 * Represents a map of configuration values.
 */
typealias Config = Map<String, Scalar<Any>>

/**
 * Represents a map of properties.
 */
typealias Properties = Map<String, Scalar<Any>>

@IntDef(
    Code.OK,
    Code.GET_PROPERTIES_NOT_IMPLEMENTED,
    Code.GET_PARAMETERS_NOT_IMPLEMENTED,
    Code.FIT_NOT_IMPLEMENTED,
    Code.EVALUATE_NOT_IMPLEMENTED
)
@Retention(AnnotationRetention.SOURCE)
annotation class CodeAnnotation

/**
 * The `Code` class defines client status codes used in the application.
 */
object Code {
    // Client status codes.
    const val OK: Int = 0
    const val GET_PROPERTIES_NOT_IMPLEMENTED: Int = 1
    const val GET_PARAMETERS_NOT_IMPLEMENTED: Int = 2
    const val FIT_NOT_IMPLEMENTED: Int = 3
    const val EVALUATE_NOT_IMPLEMENTED: Int = 4
}

/**
 * Client status.
 */
data class Status(val code: Int, val message: String)

/**
 * The `Scalar` class represents a scalar value that can have different data types.
 *
 * @param <T> The type parameter specifying the data type of the scalar value. It contains types
 * corresponding to ProtoBuf types that ProtoBuf considers to be "Scalar Value Types", even though
 * some of them arguably do not conform to other definitions of what a scalar is. Source:
 * https://developers.google.com/protocol-buffers/docs/overview#scalar
 */
sealed class Scalar<T> {
    abstract val value: T
    data class BoolValue(override val value: Boolean): Scalar<Boolean>()
    data class BytesValue(override val value: ByteString): Scalar<ByteString>()
    data class SInt64Value(override val value: Long): Scalar<Long>()
    data class DoubleValue(override val value: Double): Scalar<Double>()
    data class StringValue(override val value: String): Scalar<String>()
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
