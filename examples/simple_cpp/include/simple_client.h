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
#include <iostream>
#include <ctime>

typedef std::vector<std::tuple<float,float> Dataset

/**
 * Validate the network on the entire test set
 *
 */
std::tuple<size_t, float, double> test(float alpha, float beta, Dataset& testset);
std::tuple<size_t, float, double> train(float alpha, float beta, Dataset& trainset);

class SimpleFlwrClient : public flwr::Client {
  private:
    float alpha, beta;
    int64_t client_id;
    Dataset& trainset;
    Dataset& testset;

  public:
    SimpleFlwrClient(std::string client_id, Dataset& trainset, Dataset& testset);

    virtual flwr::ParametersRes get_parameters() override;
    virtual flwr::PropertiesRes get_properties(flwr::PropertiesIns ins) override;
    virtual flwr::EvaluateRes evaluate(flwr::EvaluateIns ins) override;
    virtual flwr::FitRes fit(flwr::FitIns ins) override;

};