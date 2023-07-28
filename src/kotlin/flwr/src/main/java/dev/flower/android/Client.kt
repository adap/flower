package dev.flower.android

// Flower client (abstract base class).
interface Client {
    fun getProperties(ins: GetPropertiesIns): GetPropertiesRes

    fun getParameters(ins: GetParametersIns): GetParametersRes

    fun fit(ins: FitIns): FitRes

    fun evaluate(ins: EvaluateIns): EvaluateRes
}
