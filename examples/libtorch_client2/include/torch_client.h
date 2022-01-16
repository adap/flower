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
#include <torch/torch.h>
#include <torchvision/vision.h>
#include <torchvision/models/resnet.h>
#include "cifar10.h"
#include <ctime>

/**
 * Validate the network on the entire test set
 *
 */
template <typename DataLoader>
std::tuple<size_t, float, double> test(vision::models::ResNet18& net,
    DataLoader& test_loader,
    torch::Device device);

/**
 * Train the net work on the training set
 */
template <typename DataLoader>
std::tuple<size_t, float, double> train(vision::models::ResNet18& net, 
    DataLoader& train_loader,
    torch::optim::Optimizer& optimizer,
    torch::Device device);

/**
 * A concrete libtorch flower example extends from flwr::Client abstract class
 * Uses ResNet18 as base model
 *
 */

template<typename DataLoader>
class TorchClient : public flwr::Client {
  private:
    vision::models::ResNet18& net;
    DataLoader& train_loader;
    DataLoader& test_loader;
    torch::optim::Optimizer& optimizer;
    torch::Device device;
    int64_t client_id;

  public:
    TorchClient(std::string client_id,
        vision::models::ResNet18& net,
        DataLoader& trainset,
        DataLoader& testset,
        torch::optim::Optimizer& optimizer,
        torch::Device device);

    virtual flwr::ParametersRes get_parameters() override;
    virtual flwr::EvaluateRes evaluate(flwr::EvaluateIns ins) override;
    virtual flwr::FitRes fit(flwr::FitIns ins) override;

};

#include "torch_client_impl.h"
