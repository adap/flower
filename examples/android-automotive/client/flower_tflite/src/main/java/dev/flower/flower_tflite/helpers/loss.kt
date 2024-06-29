package dev.flower.flower_tflite.helpers

import dev.flower.flower_tflite.Sample
import kotlin.math.ln
import kotlin.math.exp

/*
fun <X> negativeLogLikelihoodLoss(
    samples: MutableList<Sample<X, FloatArray>>,
    logits: Array<FloatArray>
): Float = averageLossWith(samples, logits) { sample, logit ->
    -ln(logit[sample.label.argmax()])
}
*/

fun softmax(logits: Array<FloatArray>): Array<FloatArray> {
    return logits.map { logit ->
        val maxLogit = logit.maxOrNull() ?: throw IllegalArgumentException("Logit cannot be empty")
        val expLogits = logit.map { exp(it - maxLogit) }
        val sumExpLogits = expLogits.sum()
        expLogits.map { it / sumExpLogits }.toFloatArray()
    }.toTypedArray()
}

fun <X> categoricalCrossEntropyLoss(
    samples: MutableList<Sample<X, FloatArray>>,
    logits: Array<FloatArray>
): Float = averageLossWith(samples, logits) { sample, logit ->
    val softmaxLogit = softmax(arrayOf(logit)).first()
    -ln(softmaxLogit[sample.label.argmax()])
}

fun <X, Y> averageLossWith(
    samples: MutableList<Sample<X, Y>>,
    logits: Array<Y>,
    loss: (Sample<X, Y>, logit: Y) -> Float
): Float {
    var lossSum = 0f
    for (i in samples.indices) {
        val sample = samples[i]
        val logit = logits[i]
        lossSum += loss(sample, logit)
    }
    return lossSum / samples.size
}

/*
fun <X, Y> sparseCategoricalCrossentropyLoss(
    samples: MutableList<Sample<X, Y>>,
    logits: Array<Y>
): Float = averageLossWith(samples, logits) { sample, logit ->

}
*/
