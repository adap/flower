#pragma once
#include "client.h"
#include <iostream>
#include <torch/torch.h>
#include <torchvision/vision.h>
#include <torchvision/models/resnet.h>
#include "cifar10.h"
#include <ctime>

/*template <typename DataLoader>
std::tuple<size_t, float, double> test(vision::models::ResNet18& net,
    DataLoader& test_loader,
    torch::Device device)
{
    net->to(device);
    net->eval();
    size_t num_samples = 0;
    size_t num_correct = 0;
    float running_loss = 0.0;

    // Iterate the data loader to yield batches from the dataset.
    for (auto& batch : *test_loader)
    {
        auto data = batch.data.to(device), target = batch.target.to(device);
        num_samples += data.size(0);
        // Execute the model on the input data.
        torch::Tensor output = net->forward(data);

        // Compute a loss value to judge the prediction of our model.
        torch::Tensor loss = torch::nn::functional::cross_entropy(output, target);

        AT_ASSERT(!std::isnan(loss.template item<float>()));
        running_loss += loss.item<float>() * data.size(0); // CHECK IF IT IS DOUBLE OR FLOAT!!!!
        auto prediction = output.argmax(1);
        num_correct += prediction.eq(target).sum().template item<int64_t>();
    }
    auto mean_loss = running_loss / num_samples;
    auto accuracy = static_cast<double>(num_correct) / num_samples;

    std::cout << "Testset - Loss: " << mean_loss << ", Accuracy: " << accuracy << std::endl;

    return std::make_tuple(num_samples, running_loss, accuracy);
}

template <typename DataLoader>
std::tuple<size_t, float, double> train(vision::models::ResNet18& net, 
    DataLoader& train_loader,
    torch::optim::Optimizer& optimizer,
    torch::Device device)
{
    net->to(device);
    net->train();
    size_t num_samples = 0;
    size_t num_correct = 0;
    float running_loss = 0.0;

    // Iterate the data loader to yield batches from the dataset.
    for (auto& batch : *train_loader){
        auto data = batch.data.to(device), target = batch.target.to(device);
        num_samples += data.size(0);
        // Reset gradients.
        optimizer.zero_grad();

        // Execute the model on the input data.
        torch::Tensor output = net->forward(data);

        // Compute a loss value to judge the prediction of our model.
        torch::Tensor loss = torch::nn::functional::cross_entropy(output, target);

        AT_ASSERT(!std::isnan(loss.template item<float>()));
        //std::cout << loss.item<float>() << std::endl;
        running_loss += loss.item<float>() * data.size(0); // CHECK IF IT IS DOUBLE OR FLOAT!!!!
        auto prediction = output.argmax(1);
        num_correct += prediction.eq(target).sum().template item<int64_t>();

        // Compute gradients of the loss w.r.t. the parameters of our model.
        loss.backward();
        // Update the parameters based on the calculated gradients.
        optimizer.step();
    }
    auto mean_loss = running_loss / num_samples;
    auto accuracy = static_cast<double>(num_correct) / num_samples;

    return std::make_tuple(num_samples, mean_loss, accuracy);
}
*/
/*
* pytorch client
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
    TorchClient(int64_t client_id,
        vision::models::ResNet18& net,
        DataLoader& trainset,
        DataLoader& testset,
        torch::optim::Optimizer& optimizer,
        torch::Device device);

    flwr::ParametersRes get_parameters() override;
    flwr::EvaluateRes evaluate(flwr::EvaluateIns ins) override;
    flwr::FitRes fit(flwr::FitIns ins) override;

};
