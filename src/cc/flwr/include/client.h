/*************************************************************************************************
 *
 * @file client.h
 *
 * @brief C++ Flower client (abstract base class)
 *
 * @author Lekang Jiang
 *
 * @version 1.0
 *
 * @date 03/09/2021
 *
 *************************************************************************************************/

#pragma once
#include "typing.h"

namespace flwr_local {
/**
 *
 * Abstract base class for C++ Flower clients
 *
 */
class Client {
public:
  /**
   *
   * @brief Return the current local model parameters
   * @return ParametersRes
   *             The current local model parameters
   *
   */
  virtual ParametersRes get_parameters() = 0;

  virtual PropertiesRes get_properties(PropertiesIns ins) = 0;
  /**
   *
   * @brief Refine the provided weights using the locally held dataset
   * @param FitIns
   *             The training instructions containing (global) model parameters
   *             received from the server and a dictionary of configuration
   * values used to customize the local training process.
   * @return FitRes
   *             The training result containing updated parameters and other
   * details such as the number of local training examples used for training.
   */
  virtual FitRes fit(FitIns ins) = 0;

  /**
   *
   * @brief Evaluate the provided weights using the locally held dataset.
   * @param EvaluateIns
   *             The evaluation instructions containing (global) model
   * parameters received from the server and a dictionary of configuration
   * values used to customize the local evaluation process.
   * @return EvaluateRes
   *             The evaluation result containing the loss on the local dataset
   * and other details such as the number of local data examples used for
   *             evaluation.
   */
  virtual EvaluateRes evaluate(EvaluateIns ins) = 0;
};
} // namespace flwr_local
