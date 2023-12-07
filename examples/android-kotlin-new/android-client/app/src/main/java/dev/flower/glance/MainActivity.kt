package dev.flower.glance

import android.content.Context
import android.os.Bundle
import android.text.method.ScrollingMovementMethod
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.EditText
import android.widget.Switch
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.widget.SwitchCompat
import dev.flower.android.HUNDRED_MEBIBYTE
import dev.flower.android.startClient
import dev.flower.android.startFlowerRere
import io.grpc.ManagedChannelBuilder
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.MainScope
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.BufferedReader
import java.io.InputStreamReader
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.ln

class MainActivity : AppCompatActivity() {
    private val scope = MainScope()
    private lateinit var ip: EditText
    private lateinit var port: EditText
    private lateinit var enableBackgroundTask: SwitchCompat
    private lateinit var trainButton: Button
    private lateinit var resultText: TextView
    private lateinit var schedulingMinutes: EditText
    private lateinit var flowerClient: FlowerClient<Float3DArray, FloatArray>
    private lateinit var flowerGlanceClient: FlowerGlanceClient
    private lateinit var dataset: Array<FloatArray>
    private lateinit var label: FloatArray

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.flower_activity)
        resultText = findViewById(R.id.grpc_response_text)
        resultText.movementMethod = ScrollingMovementMethod()
        enableBackgroundTask = findViewById(R.id.enableBackgroundTask)
        schedulingMinutes = findViewById(R.id.scheduling)
        ip = findViewById(R.id.serverIP)
        port = findViewById(R.id.serverPort)
        trainButton = findViewById(R.id.trainFederated)
        scope.launch {
            dataset = loadDataset()
            label = loadLabel()
            createFlowerClient()
            flowerGlanceClient.initialiseLayerShape()
            flowerGlanceClient.getModelParameters()
            flowerGlanceClient.inference(dataset, label)
            var datasetsCopy = flowerGlanceClient.deepCopy(dataset)
            val indices = "0"
            val indicesAsArray = indices.split(", ")
            val indicesArray: Array<Int> = indicesAsArray.map { it.toInt() }.toTypedArray()
            val preprocessedDataset = flowerGlanceClient.preprocess(dataset, indicesArray, "multiply01")
            datasetsCopy.indices.forEach { i ->
                var k = 0
                indicesArray.forEach { j ->
                    datasetsCopy[i][j] = preprocessedDataset[k][i]
                    k++
                }
            }
            val beforeArray = dataset.take(3).toTypedArray().contentDeepToString()
            val afterArray = datasetsCopy.take(3).toTypedArray().contentDeepToString()
            Log.i(TAG, "Before preprocess: $beforeArray.")
            Log.i(TAG, "After preprocess: $afterArray.")
            repeat((0 until 70).count()) { flowerGlanceClient.train(dataset, label) }
            flowerGlanceClient.inference(dataset, label)
        }

    }

    fun runGrpc(view: View) {
        scope.launch { startClient("65.108.122.72:9092", false, flowerGlanceClient) }
    }

    private fun createFlowerClient() {
        val tfliteBuffer = loadMappedAssetFile(this, "model/logistic_regression.tflite")
        val layersSizes = intArrayOf(30, 1)
        val sampleSpec = SampleSpec<Float3DArray, FloatArray>(
            { it.toTypedArray() },
            { it.toTypedArray() },
            { Array(it) { FloatArray(10) } },
            ::negativeLogLikelihoodLoss,
            ::classifierAccuracy,
        )
        flowerClient = FlowerClient(tfliteBuffer, layersSizes, null, sampleSpec)
        flowerGlanceClient = FlowerGlanceClient(tfliteBuffer, dataset, label, this)
    }

    private fun loadMappedAssetFile(context: Context, filePath: String): MappedByteBuffer {
        val fileDescriptor = context.assets.openFd(filePath)
        val fileChannel = fileDescriptor.createInputStream().channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    suspend fun readAssetLines(
        context: Context,
        fileName: String,
        call: suspend (String) -> Unit
    ) {
        withContext(Dispatchers.IO) {
            val iterator = BufferedReader(InputStreamReader(context.assets.open(fileName))).lineSequence().iterator()
            while(iterator.hasNext()) {
                var line = iterator.next()
                call(line)
            }
        }
    }


    suspend fun loadDataset():Array<FloatArray> {
        var dataset: MutableList<FloatArray> = emptyList<FloatArray>().toMutableList()

        readAssetLines(this, "data/datasets.txt") { line ->
            val splitOpenBracket = line.split('[')
            Log.i(TAG, "SplitOpenBracket: ${splitOpenBracket.size}.")
            //splitOpenBracket.map { row -> row.removeSuffix("]").split(", ").map { i -> i.toFloat() } }
            for (row in splitOpenBracket) {
                val removeSuffix = row.removeSuffix("]")
                val data = removeSuffix.split(", ").filter { i -> i != "" }
                val dataRow = data.map { i -> i.toFloat() }
                if (dataRow.size == 30) {
                    dataset.add(dataRow.toFloatArray())
                }
            }
            Log.i(TAG, "Dataset size: ${dataset.size}.")
        }

        return dataset.toTypedArray()
    }

    suspend fun loadLabel():FloatArray {
        var label: MutableList<Float> = emptyList<Float>().toMutableList()
        readAssetLines(this, "data/labels.txt") { line ->
            line.map { i -> i.toString().toFloat() }.toCollection(label)
        }
        return label.toFloatArray()
    }

    companion object {
        private const val TAG = "Flower Main Activity"
    }
}

typealias Float3DArray = Array<Array<FloatArray>>

fun <X> classifierAccuracy(
    samples: MutableList<Sample<X, FloatArray>>,
    logits: Array<FloatArray>
): Float = averageLossWith(samples, logits) { sample, logit ->
    sample.label[logit.argmax()]
}

fun <X> negativeLogLikelihoodLoss(
    samples: MutableList<Sample<X, FloatArray>>,
    logits: Array<FloatArray>
): Float = averageLossWith(samples, logits) { sample, logit ->
    -ln(logit[sample.label.argmax()])
}

fun <X, Y> averageLossWith(
    samples: MutableList<Sample<X, Y>>,
    logits: Array<Y>,
    loss: (Sample<X, Y>, logit: Y) -> Float
): Float {
    var lossSum = 0f
    for ((sample, logit) in samples lazyZip logits) {
        lossSum += loss(sample, logit)
    }
    return lossSum / samples.size
}

infix fun <T, R> Iterable<T>.lazyZip(other: Array<out R>): Sequence<Pair<T, R>> {
    val ours = iterator()
    val theirs = other.iterator()

    return sequence {
        while (ours.hasNext() && theirs.hasNext()) {
            yield(ours.next() to theirs.next())
        }
    }
}

fun FloatArray.argmax(): Int = indices.maxBy { this[it] }