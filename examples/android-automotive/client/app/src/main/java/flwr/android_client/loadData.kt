package flwr.android_client

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import dev.flower.flower_tflite.FlowerClient
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.BufferedReader
import java.io.InputStreamReader
import java.util.concurrent.ExecutionException

suspend fun readAssetLines(
    context: Context,
    fileName: String,
    call: suspend (Int, String) -> Unit
) {
    withContext(Dispatchers.IO) {
        BufferedReader(InputStreamReader(context.assets.open(fileName))).useLines {
            it.forEachIndexed { i, l -> launch { call(i, l) } }
        }
    }
}

/**
 * Load training data from disk.
 */
@Throws
suspend fun loadData(
    context: Context,
    flowerClient: FlowerClient<FeatureArray, FloatArray>,
    device_id: Int
) {
    readAssetLines(context, "data/part_0.csv") { index, line ->
        addSample(context, flowerClient, "data/$line", true)
    }
}

@Throws
private fun addSample(
    context: Context,
    flowerClient: FlowerClient<FeatureArray, FloatArray>,
    row: String,
    isTraining: Boolean
) {
    if (ENABLE_DEBUG){
        println("Adding sample: $row")
    }
    val parts = row.split(",".toRegex())
    val index = parts[0][5].toString().toInt()
    val className = CLASSES[index]
    val labelArray = classToLabel(className)
    val features = parts.subList(1, parts.size).map { it.toFloat() }.toFloatArray()

    if (ENABLE_DEBUG) {
        println("Index: $index")
        print("Feature Array: ")
        for (feature in features) {
            print("$feature ")
        }
        print("\nLabel Array: ")
        for (label in labelArray) {
            print("$label ")
        }
        println()
    }

    // add to the list.
    try {
        //flowerClient.addSample(features, labelArray, isTraining)
    } catch (e: ExecutionException) {
        throw RuntimeException("Failed to add sample to model", e.cause)
    } catch (e: InterruptedException) {
        // no-op
    }
}

private const val TAG = "Load Data"

const val INPUT_LAYER_SIZE = 14

val CLASSES = listOf(
    "NO FAULT",
    "RICH MIXTURE",
    "LEAN MIXTURE",
    "LOW VOLTAGE",
)

//CSV Fault,MAP,TPS,Force,Power,RPM,Consumption L/H,Consumption L/100KM,Speed,CO,HC,CO2,O2,Lambda,AFR
val FEATURES = listOf(
    "Fault",
    "MAP",
    "TPS",
    "Force",
    "Power",
    "RPM",
    "Consumption L/H",
    "Consumption L/100KM",
    "Speed",
    "CO",
    "HC",
    "CO2",
    "O2",
    "Lambda",
    "AFR"
)

const val ENABLE_DEBUG = false


fun classToLabel(className: String): FloatArray {
    return CLASSES.map {
        if (className == it) 1f else 0f
    }.toFloatArray()
}


