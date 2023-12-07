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
import dev.flower.android.Scalar
import dev.flower.android.Status
import org.tensorflow.lite.Interpreter
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.FloatBuffer
import java.nio.IntBuffer
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock

class FlowerGlanceClient(
    tfliteFileBuffer: MappedByteBuffer? = null,
    private val datasets: Array<FloatArray>,
    private val labels: FloatArray,
    private val context: Context? = null,
) : Client {
    private var interpreter: Interpreter? = tfliteFileBuffer?.let { Interpreter(it) }
    private val interpreterLock = ReentrantLock()
    private var layerShape: List<Int> = ArrayList()
    private val modelTflite = "model.tflite"
    private val modelIdKey = "model_id"
    private val tfliteKey = "tf_lite"
    private val preprocessKey = "preprocess"
    private var batchSize = 32
    private var localEpochs = 1
    private val localEpochKey = "local_epochs"
    private val batchSizeKey = "batch_size"

    override fun getParameters(ins: GetParametersIns): GetParametersRes {
        val modelId = when (val scalar = ins.config.getOrDefault(modelIdKey, Scalar.StringValue("1"))) {
            is Scalar.StringValue -> scalar.value
            else -> throw IllegalArgumentException("Unknown Scalar type: $scalar")
        }
        Log.i(TAG, "GetParameters: $ins")
        initialiseInterpreter(modelId)
        initialiseLayerShape()
        val parameters = getModelParameters()
        return GetParametersRes(Status(Code.OK, "Success"), Parameters(parameters, "byteArray"))
    }

    fun deepCopy(array: Array<FloatArray>): Array<FloatArray> {
        val arrayCopy = Array(array.size) { FloatArray(array.first().size) }
        array.indices.forEach { i ->
            array.first().indices.forEach { j ->
                arrayCopy[i][j] = array[i][j]
            }
        }
        return arrayCopy
    }

    fun initialiseLayerShape() {
        val layerCountInputs: Map<String, Any> = FakeNonEmptyMap()
        val layerCountOutput = IntBuffer.allocate(1)
        val layerCountOutputs = mapOf("count" to layerCountOutput)
        runSignatureLocked(layerCountInputs, layerCountOutputs, "layer_count")
        val layerCount = layerCountOutput.get(0)
        Log.i(TAG, "Layer Count: $layerCount")
        val layerShapeInputs: Map<String, Any> = FakeNonEmptyMap()
        val layerShapeOutputs = (0 until layerCount).associate {
            "a$it" to IntBuffer.allocate(2)
        }
        runSignatureLocked(layerShapeInputs, layerShapeOutputs, "layers")
        layerShape = layerShapeOutputs.map {
            val intBuffer = it.value as IntBuffer
            var second = intBuffer.get(1)
            if (second == 0) {
                second = 1
            }
            intBuffer.get(0) * second
        }
        Log.i(TAG, "Layer Shape: $layerShape")
    }

    fun getModelParameters(): Array<ByteBuffer> {
        val inputs: Map<String, Any> = FakeNonEmptyMap()
        val outputs = layerShape.indices.associate {
            "a$it" to ByteBuffer.allocate(4 * layerShape[it])
        }
        runSignatureLocked(inputs, outputs, "parameters")
        return layerShape.indices.map {
            val buffer = outputs["a$it"] as ByteBuffer
            buffer.rewind()
            buffer
        }.toTypedArray()
    }

    fun preprocess(dataset: Array<FloatArray>, columns: Array<Int>, preprocessSignature: String): Array<FloatArray> {
        val inputArray: Array<FloatArray> = columns.map { column ->
            dataset.map { it[column] }.toFloatArray()
        }.toTypedArray()
        val inputs = mapOf("input" to inputArray)
        val outputArray = inputArray.map {(FloatArray(dataset.size))}.toTypedArray()
        Log.i(TAG, "Output Array: ${outputArray.size}, ${outputArray[0].size}")
        Log.i(TAG, "Input Array: ${inputArray.size}, ${inputArray[0].size}")

        val outputs = mapOf("output_0" to outputArray)
        runSignatureLocked(inputs, outputs, preprocessSignature)
        return outputArray
    }

    fun inference(dataset: Array<FloatArray>, label: FloatArray): Float {
        val inputs = mapOf("x" to dataset, "y" to label)
        val accuracy = dataset.map { FloatArray(1)}.toTypedArray()
        val outputs = mapOf("accuracy" to accuracy)
        runSignatureLocked(inputs, outputs, "evaluate")
        val array = accuracy.indices.map {i ->
            if (accuracy[i][0] == label[i]) {
                1.0
            } else {
                0.0
            }
        }.average()
        Log.i(TAG, "Inference: $array")
        return array.toFloat()
    }

    private fun trainingBatches(dataset: Array<FloatArray>, label: FloatArray, batchSize: Int): Sequence<Pair<List<FloatArray>, FloatArray>> {
        val datasetList = dataset.toList()
        val labelList = label.toList()
        assertIntsEqual(dataset.size, label.size)

        return sequence {
            var nextIndex = 0

            while (nextIndex < dataset.size) {
                val fromIndex = nextIndex
                nextIndex += batchSize

                val batch = if (nextIndex >= dataset.size) {
                    Pair(datasetList.subList(
                        dataset.size - batchSize, dataset.size
                    ), labelList.subList(dataset.size - batchSize, dataset.size).toFloatArray())
                } else {
                    Pair(datasetList.subList(fromIndex, nextIndex), labelList.subList(fromIndex, nextIndex).toFloatArray())
                }
                yield(batch)
            }
        }
    }

    fun train(dataset: Array<FloatArray>, label: FloatArray) {
        val losses = trainingBatches(dataset, label, batchSize).map {
            val inputs = mapOf("x" to it.first.toTypedArray(), "y" to it.second)
            val loss = FloatBuffer.allocate(1)
            val outputs = mapOf("loss" to loss)
            runSignatureLocked(inputs, outputs, "train")
            loss.get(0)
        }
        val meanLoss = losses.average()
        Log.i(TAG, "Loss: $meanLoss")
    }

    private fun emptyParameterMap(): Map<String, Any> {
        return layerShape.mapIndexed { index, size -> "a$index" to ByteBuffer.allocate(size*4) }
            .toMap()
    }

    private fun initialiseInterpreter(modelId: String) {
        val modelBytes = context?.let { getTfliteFromDir(it, modelId) }
        interpreter = modelBytes?.let { Interpreter(it) }
    }

    private fun getTfliteFromDir(context: Context, modelId: String): MappedByteBuffer? {
        val modelDir = File(context.getExternalFilesDir(null), modelId)
        if(modelDir.exists() && modelDir.isDirectory) {
            val modelFile = File(modelDir, modelTflite)
            if(modelFile.exists()) {
                val inputStream = FileInputStream(modelFile)
                val channel = inputStream.channel
                val ret = channel.map(FileChannel.MapMode.READ_ONLY, 0, channel.size())
                channel.close()
                return ret
            }
        }
        return null
    }

    private fun saveTfliteToDir(context: Context, modelId: String, modelBytes: ByteBuffer) {
        val modelDir = File(context.getExternalFilesDir(null), modelId)
        if(!modelDir.exists()) {
            modelDir.mkdirs()
        }
        val modelFile = File(modelDir, modelTflite)
        if(modelFile.createNewFile()) {
            val outputStream = FileOutputStream(modelFile)
            val channel = outputStream.channel
            channel.write(modelBytes)
            channel.close()
        }
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

    private fun parametersFromMap(map: Map<String, Any>): Array<ByteBuffer> {
        // assertIntsEqual(layerShape.size, map.size)
        return (0 until map.size).map {
            val buffer = map["a$it"] as ByteBuffer
            buffer.rewind()
            buffer
        }.toTypedArray()
    }

    override fun fit(ins: FitIns): FitRes {
        //Check model id
        val modelId = when (val scalar = ins.config.getOrDefault(modelIdKey, Scalar.StringValue("1"))) {
            is Scalar.StringValue -> scalar.value
            else -> throw IllegalArgumentException("Unknown Scalar type: $scalar")
        }

        //Retrieve model based on model id and initialise tflite and layer shape
        ins.config[tfliteKey]?.let { scalar ->
            val modelBytes = when (scalar) {
                is Scalar.BytesValue -> scalar.value
                else -> throw IllegalArgumentException("Unknown Scalar type: $scalar")
            }
            context?.let { saveTfliteToDir(it, modelId, modelBytes) }
        }

        ins.config[batchSizeKey]?.let { scalar ->
            val batchSize = when (scalar) {
                is Scalar.SInt64Value -> scalar.value
                else -> throw IllegalArgumentException("Unknown Scalar type: $scalar")
            }
            this.batchSize = batchSize.toInt()
        }

        ins.config[localEpochKey]?.let { scalar ->
            val localEpochs = when (scalar) {
                is Scalar.SInt64Value -> scalar.value
                else -> throw IllegalArgumentException("Unknown Scalar type: $scalar")
            }
            this.localEpochs = localEpochs.toInt()
        }

        initialiseInterpreter(modelId)
        initialiseLayerShape()

        //Preprocess dataset
        val datasetsCopy = deepCopy(datasets)
        ins.config[preprocessKey]?.let { scalar ->
            val prepFunctions = when (scalar) {
                is Scalar.StringValue -> scalar.value
                else -> throw IllegalArgumentException("Unknown Scalar type: $scalar")
            }
            val prepFunctionsAsArray = prepFunctions.split(", ")
            prepFunctionsAsArray.forEach { key ->
                ins.config[key].let { indexScalar ->
                    val indices = when (indexScalar) {
                        is Scalar.StringValue -> indexScalar.value
                        else -> throw IllegalArgumentException("Unknown Scalar type: $scalar")
                    }
                    val indicesAsArray = indices.split(", ")
                    val indicesArray: Array<Int> = indicesAsArray.map { it.toInt() }.toTypedArray()
                    Log.i(TAG, "Running preprocessing function: $key for index ${indicesArray.contentDeepToString()}")
                    val preprocessedDataset = preprocess(datasets, indicesArray, key)
                    Log.i(TAG, "Running preprocessing function: $key result: ${preprocessedDataset.size}")
                    datasetsCopy.indices.forEach { i ->
                        var k = 0
                        indicesArray.forEach { j ->
                            datasetsCopy[i][j] = preprocessedDataset[k][i]
                            k++
                        }
                    }
                }
            }
        }
        //Initialise model parameters with parameters from fit
        val layers = ins.parameters.tensors
        assertIntsEqual(layers.size, layerShape.size)
        updateParameters(layers)

        repeat(localEpochs) {
            train(datasetsCopy, labels)
        }

        val parameters = getModelParameters()
        return FitRes(
            Status(Code.OK, "Success"),
            Parameters(parameters, "byteArray"),
            datasetsCopy.size,
            emptyMap()
        )
    }

    private fun parametersToMap(parameters: Array<ByteBuffer>): Map<String, Any> {
        assertIntsEqual(layerShape.size, parameters.size)
        return parameters.mapIndexed { index, bytes -> "a$index" to bytes }.toMap()
    }

    private fun updateParameters(parameters: Array<ByteBuffer>): Array<ByteBuffer> {
        val outputs = emptyParameterMap()
        val inputs = parametersToMap(parameters)
        Log.i(TAG, "Inputs Array for restore: $inputs")
        runSignatureLocked(inputs, outputs, "restore")
        return parametersFromMap(outputs)
    }

    override fun evaluate(ins: EvaluateIns): EvaluateRes {
        val layers = ins.parameters.tensors
        assertIntsEqual(layers.size, layerShape.size)
        updateParameters(layers)
        val result = inference(datasets, labels)
        return EvaluateRes(Status(Code.OK, "Success"), result, datasets.size, emptyMap())
    }

    override fun getProperties(ins: GetPropertiesIns): GetPropertiesRes {
        return GetPropertiesRes(Status(Code.GET_PROPERTIES_NOT_IMPLEMENTED, "GetProperties not implemented"), emptyMap())
    }

    companion object {
        private const val TAG = "Flower Glance Client"
    }
}
