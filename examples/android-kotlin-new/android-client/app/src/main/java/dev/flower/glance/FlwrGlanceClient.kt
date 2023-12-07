package dev.flower.glance

import android.content.Context
import android.util.Log
import dev.flower.android.Client
import dev.flower.android.Code
import dev.flower.android.EvaluateIns
import dev.flower.android.EvaluateRes
import dev.flower.android.FitIns
import dev.flower.android.FitRes
import dev.flower.android.GetParametersIns
import dev.flower.android.GetParametersRes
import dev.flower.android.GetPropertiesIns
import dev.flower.android.GetPropertiesRes
import dev.flower.android.Parameters
import dev.flower.android.Status
import org.tensorflow.lite.Interpreter
import java.lang.Integer.min
import java.nio.ByteBuffer
import java.nio.FloatBuffer
import java.nio.MappedByteBuffer
import java.util.concurrent.locks.ReentrantLock
import java.util.concurrent.locks.ReentrantReadWriteLock
import kotlin.concurrent.read
import kotlin.concurrent.withLock
import kotlin.concurrent.write

/**
 * Flower client that handles TensorFlow Lite model [Interpreter] and sample data.
 * @param tfliteFileBuffer TensorFlow Lite model file.
 * @param layersSizes Sizes of model parameters layers in bytes.
 * @param spec Specification for the samples, see [SampleSpec].
 */
class FlowerClient<X : Any, Y : Any>(
    tfliteFileBuffer: MappedByteBuffer? = null,
    private val layersSizes: IntArray,
    private val context: Context? = null,
    private val spec: SampleSpec<X, Y>,
) : AutoCloseable, Client {
    private val interpreter: Interpreter? = tfliteFileBuffer?.let { Interpreter(it) }
    private val interpreterLock = ReentrantLock()
    private val trainingSamples = mutableListOf<Sample<X, Y>>()
    private val testSamples = mutableListOf<Sample<X, Y>>()
    private val trainSampleLock = ReentrantReadWriteLock()
    private val testSampleLock = ReentrantReadWriteLock()

    /**
     * Add one sample point ([bottleneck], [label]) for training or testing later.
     * Thread-safe.
     */
    fun addSample(
        bottleneck: X, label: Y, isTraining: Boolean
    ) {
        val samples = if (isTraining) trainingSamples else testSamples
        val lock = if (isTraining) trainSampleLock else testSampleLock
        lock.write {
            samples.add(Sample(bottleneck, label))
        }
    }

    override fun evaluate(ins: EvaluateIns): EvaluateRes {
        // Update the local parameters
        val layers = ins.parameters.tensors
        assertIntsEqual(layers.size, layersSizes.size)
        updateParameters(layers)

        // Start local evaluate
        val result = testSampleLock.read {
            val bottlenecks = testSamples.map { it.bottleneck }
            val logits = inference(spec.convertX(bottlenecks))
            spec.loss(testSamples, logits) to spec.accuracy(testSamples, logits)
        }
        Log.d(TAG, "Evaluate loss & accuracy: $result.")
        // Return loss as EvaluateRes
        return EvaluateRes(Status(Code.OK, "Success"), result.first, trainingSamples.size, emptyMap())
    }

    private fun parametersFromMap(map: Map<String, Any>): Array<ByteBuffer> {
        assertIntsEqual(layersSizes.size, map.size)
        return (0 until map.size).map {
            val buffer = map["a$it"] as ByteBuffer
            buffer.rewind()
            buffer
        }.toTypedArray()
    }

    private fun parametersToMap(parameters: Array<ByteBuffer>): Map<String, Any> {
        assertIntsEqual(layersSizes.size, parameters.size)
        return parameters.mapIndexed { index, bytes -> "a$index" to bytes }.toMap()
    }

    private fun runSignatureLocked(
        inputs: Map<String, Any>,
        outputs: Map<String, Any>,
        signatureKey: String
    ) {
        interpreterLock.withLock {
            interpreter?.runSignature(inputs, outputs, signatureKey)
        }
    }

    private fun updateParameters(parameters: Array<ByteBuffer>): Array<ByteBuffer> {
        val outputs = emptyParameterMap()
        runSignatureLocked(parametersToMap(parameters), outputs, "restore")
        return parametersFromMap(outputs)
    }

    private fun inference(x: Array<X>): Array<Y> {
        val inputs = mapOf("x" to x)
        val logits = spec.emptyY(x.size)
        val outputs = mapOf("logits" to logits)
        runSignatureLocked(inputs, outputs, "infer")
        return logits
    }

    override fun fit(ins: FitIns): FitRes {
        val modelBytes = ins.config.getOrDefault(
            "tf_lite", null
        )

        // Update the local parameters
        val layers = ins.parameters.tensors
        assertIntsEqual(layers.size, layersSizes.size)
        val epochs = ins.config.getOrDefault(
            "local_epochs", 1
        )!!
        updateParameters(layers)

        // Start local training
        trainingSamples.shuffle()
        trainingBatches(min(32, trainingSamples.size)).map {
            val bottlenecks = it.map { sample -> sample.bottleneck }
            val labels = it.map { sample -> sample.label }
            training(spec.convertX(bottlenecks), spec.convertY(labels))
        }.toList()

        // Retrieve trained parameters
        val inputs: Map<String, Any> = FakeNonEmptyMap()
        val outputs = emptyParameterMap()
        runSignatureLocked(inputs, outputs, "parameters")
        Log.i(TAG, "Raw weights: $outputs.")
        val parameters =  parametersFromMap(outputs)

        // Return trained parameters as FitRes
        return FitRes(
            Status(Code.OK, "Success"),
            Parameters(parameters, "ndarray"),
            trainingSamples.size,
            emptyMap()
        )
    }

    private fun trainingBatches(trainBatchSize: Int): Sequence<List<Sample<X, Y>>> {
        return sequence {
            var nextIndex = 0

            while (nextIndex < trainingSamples.size) {
                val fromIndex = nextIndex
                nextIndex += trainBatchSize

                val batch = if (nextIndex >= trainingSamples.size) {
                    trainingSamples.subList(
                        trainingSamples.size - trainBatchSize, trainingSamples.size
                    )
                } else {
                    trainingSamples.subList(fromIndex, nextIndex)
                }

                yield(batch)
            }
        }
    }

    private fun training(
        bottlenecks: Array<X>, labels: Array<Y>
    ): Float {
        val inputs = mapOf<String, Any>(
            "x" to bottlenecks,
            "y" to labels,
        )
        val loss = FloatBuffer.allocate(1)
        val outputs = mapOf<String, Any>(
            "loss" to loss,
        )
        runSignatureLocked(inputs, outputs, "train")
        return loss.get(0)
    }

    override fun getParameters(ins: GetParametersIns): GetParametersRes {
        val inputs: Map<String, Any> = FakeNonEmptyMap()
        val outputs = emptyParameterMap()
        runSignatureLocked(inputs, outputs, "parameters")
        Log.i(TAG, "Raw weights: $outputs.")
        val parameters = parametersFromMap(outputs)
        return GetParametersRes(Status(Code.OK, "Success"), Parameters(parameters, "byteArray"))
    }

    private fun getTfliteModel(id: String): MappedByteBuffer? {
        return null
    }

    override fun getProperties(ins: GetPropertiesIns): GetPropertiesRes {
        return GetPropertiesRes(Status(Code.GET_PROPERTIES_NOT_IMPLEMENTED, "GetProperties not implemented"), emptyMap())
    }

    private fun emptyParameterMap(): Map<String, Any> {
        return layersSizes.mapIndexed { index, size -> "a$index" to ByteBuffer.allocate(size) }
            .toMap()
    }

    companion object {
        private const val TAG = "Flower Client"
    }

    override fun close() {
        interpreter?.close()
    }
}

data class Sample<X, Y>(val bottleneck: X, val label: Y)

class FakeNonEmptyMap<K, V> : HashMap<K, V>() {
    override fun isEmpty(): Boolean {
        return false
    }
}

@Throws(AssertionError::class)
fun assertIntsEqual(expected: Int, actual: Int) {
    if (expected != actual) {
        throw AssertionError("Test failed: expected `$expected`, got `$actual` instead.")
    }
}

data class SampleSpec<X, Y>(
    val convertX: (List<X>) -> Array<X>,
    val convertY: (List<Y>) -> Array<Y>,
    val emptyY: (Int) -> Array<Y>,
    val loss: (MutableList<Sample<X, Y>>, Array<Y>) -> Float,
    val accuracy: (MutableList<Sample<X, Y>>, Array<Y>) -> Float,
)