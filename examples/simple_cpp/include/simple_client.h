/***********************************************************************************************************
 * 
 * @file libtorch_client.h
 *
 * @brief Define an example flower client, train and test method
 *
 * @version 1.0
 *
 * @date 06/09/2021
 *
 * ********************************************************************************************************/
#pragma once
#include "client.h"
#include "synthetic_dataset.h"
#include "line_fit_model.h"
#include <ctime>
#include <memory>
#include <string>
#include <tuple>
#include <iostream>
#include <sstream>
#include <vector>
#include <list>
/**
 * Validate the network on the entire test set
 *
 */

class SimpleFlwrClient : public flwr::Client {
 public:
  SimpleFlwrClient(std::string client_id,
                   LineFitModel &model,
                   SyntheticDataset &training_dataset,
                   SyntheticDataset &validation_dataset,
                   SyntheticDataset &test_dataset);
  void set_parameters(flwr::Parameters params);

  virtual flwr::ParametersRes get_parameters() override;
  virtual flwr::PropertiesRes get_properties(flwr::PropertiesIns ins) override;
  virtual flwr::EvaluateRes evaluate(flwr::EvaluateIns ins) override;
  virtual flwr::FitRes fit(flwr::FitIns ins) override;

 private:
  int64_t client_id;
  LineFitModel &model;
  SyntheticDataset &training_dataset;
  SyntheticDataset &validation_dataset;
  SyntheticDataset &test_dataset;

};
