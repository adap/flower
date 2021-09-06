/*************************************************************************************************
 * 
 * @file main.cc
 *
 * @brief Define a network, load datasets and start the flower client
 *
 * @version 1.0
 *
 * @date 06/09/2021
 *
 * ***********************************************************************************************/

#include <iostream>
#include <memory>
#include <string>
#include "start.h"
#include "torch_client.cc"

int main(int argc, char** argv) {

    // Check if we can train using CUDA
    auto cuda_available = torch::cuda::is_available();
    torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
    std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

    // Load ResNet18 model
    int num_classes = 10;
    auto net = vision::models::ResNet18(num_classes);

    // Load CIFAR10 Training Dataset and DataLoader
    int64_t kTrainBatchSize = 64;
    const std::string CIFAR10_DATASET_PATH = "/home/lekang/myflwr/flower/src/cc/data/";
    std::vector<double> norm_mean = { 0.4914, 0.4822, 0.4465 };
    std::vector<double> norm_std = { 0.247, 0.243, 0.261 };
    auto train_dataset = CIFAR10(CIFAR10_DATASET_PATH, CIFAR10::Mode::kTrain)
        .map(torch::data::transforms::Normalize<>(norm_mean, norm_std))
        .map(torch::data::transforms::Stack<>());
    const size_t train_dataset_size = train_dataset.size().value();
    auto train_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(train_dataset), kTrainBatchSize);

    // Load CIFAR10 Testing Dataset and DataLoader
    int64_t kTestBatchSize = 64;
    auto test_dataset = CIFAR10(CIFAR10_DATASET_PATH, CIFAR10::Mode::kTest)
        .map(torch::data::transforms::Normalize<>(norm_mean, norm_std))
        .map(torch::data::transforms::Stack<>());
    const size_t test_dataset_size = test_dataset.size().value();
    auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(test_dataset), kTestBatchSize);

    // Instantiate an SGD optimization algorithm to update our Net's parameters.
    float lr = 0.1;
    torch::optim::SGD optimizer(net->parameters(), lr);

    int64_t num_epochs = 1;
    
    // Initialize TorchClient
    TorchClient client(0, net, train_loader, test_loader, optimizer, device);
    
    // Define a server address
    std::string server_add = "localhost:50051";
    
    // Start client
    start_client(server_add, &client);
    
    return 0;
}
