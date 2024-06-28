package dev.flower.flower_tflite.helpers

import dev.flower.flower_tflite.Sample
import kotlin.math.ln

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

fun softmax(logits: Array<FloatArray>): Array<FloatArray> {
    val maxLogit = logits.maxOrNull() ?: throw IllegalArgumentException("Logits cannot be empty")
    val expLogits = logits.map { exp(it - maxLogit) }
    val sumExpLogits = expLogits.sum()
    return expLogits.map { it / sumExpLogits }
}

fun <X> categoricalCrossEntropyLoss(
    samples: MutableList<Sample<X, FloatArray>>,
    logits: Array<FloatArray>
): Float = averageLossWith(samples, logits) { sample, logit ->
    -ln(softmax(logit[sample.label.argmax()]))
}

