package dev.flower.android

/**
 * Abstract base class for Flower clients.
 *
 * This interface defines the basic operations that a Flower client must implement.
 */
interface Client {
    /**
     * Return set of client’s properties.
     *
     * @param ins The get properties instructions received from the server containing a dictionary
     * of configuration values.
     * @return The current client properties.
     */
    fun getProperties(ins: GetPropertiesIns): GetPropertiesRes

    /**
     * Return the current local model parameters.
     *
     * @param ins The get parameters instructions received from the server containing a dictionary
     * of configuration values.
     * @return The current local model parameters.
     */
    fun getParameters(ins: GetParametersIns): GetParametersRes

    /**
     * Refine the provided parameters using the locally held dataset.
     *
     * @param ins The training instructions containing (global) model parameters received from the
     * server and a dictionary of configuration values used to customize the local training process.
     * @return The training result containing updated parameters and other details such as the
     * number of local training examples used for training.
     */
    fun fit(ins: FitIns): FitRes

    /**
     * Evaluate the provided parameters using the locally held dataset.
     *
     * @param ins The evaluation instructions containing (global) model parameters received from
     * the server and a dictionary of configuration values used to customize the local evaluation
     * process.
     * @return The evaluation result containing the loss on the local dataset and other details
     * such as the number of local data examples used for evaluation.
     */
    fun evaluate(ins: EvaluateIns): EvaluateRes
}
