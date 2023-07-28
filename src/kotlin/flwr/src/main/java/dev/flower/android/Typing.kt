package dev.flower.android

import androidx.annotation.IntDef
import com.google.protobuf.ByteString
import java.nio.ByteBuffer

// Flower type definitions

typealias NDArray = Any // Assuming it's defined elsewhere
typealias NDArrays = List<NDArray>

// The following union type contains Kotlin types corresponding to ProtoBuf types that
// ProtoBuf considers to be "Scalar Value Types", even though some of them arguably do
// not conform to other definitions of what a scalar is. Source:
// https://developers.google.com/protocol-buffers/docs/overview#scalar
 // Assuming it represents Union of supported scalar types in Kotlin
typealias Value = Any // Assuming it represents Union of supported value types in Kotlin

typealias Metrics = Map<String, Scalar<Any>>
typealias MetricsAggregationFn = (List<Pair<Int, Metrics>>) -> Metrics

typealias Config = Map<String, Scalar<Any>>
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

object Code {
    // Client status codes.
    const val OK: Int = 0
    const val GET_PROPERTIES_NOT_IMPLEMENTED: Int = 1
    const val GET_PARAMETERS_NOT_IMPLEMENTED: Int = 2
    const val FIT_NOT_IMPLEMENTED: Int = 3
    const val EVALUATE_NOT_IMPLEMENTED: Int = 4
}

data class Status(val code: Int, val message: String)

sealed class Scalar<T> {
    abstract val value: T
    data class BoolValue(override val value: Boolean): Scalar<Boolean>()
    data class BytesValue(override val value: ByteString): Scalar<ByteString>()
    data class SInt64Value(override val value: Long): Scalar<Long>()
    data class DoubleValue(override val value: Double): Scalar<Double>()
    data class StringValue(override val value: String): Scalar<String>()
}

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

data class GetParametersIns(val config: Config)

data class GetParametersRes(val status: Status, val parameters: Parameters)

data class FitIns(val parameters: Parameters, val config: Config)

data class FitRes(val status: Status, val parameters: Parameters, val numExamples: Int, val metrics: Metrics)

data class EvaluateIns(val parameters: Parameters, val config: Config)

data class EvaluateRes(val status: Status, val loss: Float, val numExamples: Int, val metrics: Metrics)

data class GetPropertiesIns(val config: Config)

data class GetPropertiesRes(val status: Status, val properties: Properties)

data class ReconnectIns(val seconds: Long?)

data class DisconnectRes(val reason: String)
