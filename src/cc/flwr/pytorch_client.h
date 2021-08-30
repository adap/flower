#pragma once
#include "client.h"
#include <iostream>
#include <torch/torch.h>
#include <torchvision/vision.h>
#include <torchvision/models/resnet.h>
#include "cifar10.h"
#include <ctime>

template <typename DataLoader>
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
    auto sample_mean_loss = running_loss / num_samples;
    auto accuracy = static_cast<double>(num_correct) / num_samples;

    std::cout << "Testset - Loss: " << sample_mean_loss << ", Accuracy: " << accuracy << std::endl;

    return std::make_tuple(num_samples, running_loss, accuracy);
}

template <typename DataLoader>
int train(vision::models::ResNet18& net, int64_t num_epochs,
    DataLoader& train_loader,
    torch::optim::Optimizer& optimizer,
    torch::Device device)
{
    net->to(device);
    net->train();
    for (size_t epoch = 1; epoch <= num_epochs; ++epoch)
    {
        size_t num_samples = 0;
        size_t num_correct = 0;
        float running_loss = 0.0;

        // Iterate the data loader to yield batches from the dataset.
        for (auto& batch : *train_loader)
        {
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
        auto sample_mean_loss = running_loss / num_samples;
        auto accuracy = static_cast<double>(num_correct) / num_samples;

        std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs << "], Trainset - Loss: "
            << sample_mean_loss << ", Accuracy: " << accuracy << std::endl;
    }
    return num_samples;
}

/*
* pytorch client
*
*/
class pytorch_client : public Client {
public:
    pytorch_client(vision::models::ResNet18& net, int64_t num_epochs,
        DataLoader& train_loader,
        DataLoader& test_loader,
        torch::optim::Optimizer& optimizer,
        torch::Device device)
    : net(net), num_epochs(num_epochs), train_loader(train_loader), test_loader(test_loader), optimizer(optimizer), device(device){};
    

    Parameters getWeight(){
        std::list<std::string> tensors;
        for (auto p : net->parameters()) {
            tensors.push_back(p.toString());
        }
        return Parameters(tensors, "pytorch example");
    }
    
    void setWeight(Parameters parameter){
        std::list<std::string> tensors = parameter.getTensors();
        for (auto p : net->parameters()) {
            p.set_data(torch::tensor(tensors.front()));
            tensors.erase();
        }
    }
    
    virtual ParametersRes get_parameters() override {
        //std::cout << "DEBUG: get parameters" << std::endl;
        
        return ParametersRes(getWeight());
    }

    virtual FitRes fit(FitIns ins) override {

        clock_t startTime, endTime;
        startTime = clock();
        
        // set weight
        setWeight(ins.getParameters());

        int num_samples = train(net, num_epochs, train_loader, optimizer, device);

        endTime = clock();
        
        FitRes f;
        f.setParameters(getWeight());
        f.setNum_example(num_samples);
        f.setNum_examples_ceil(num_samples);
        f.setFit_duration((float)(endTime - startTime) / CLOCKS_PER_SEC);
        return f;
    }

    virtual EvaluateRes evaluate(EvaluateIns ins) override {
        
        //set weight 
        setWeight(ins.getParameters());

        std::tuple<size_t, float, double> result = test(net, test_loader, device);

        EvaluateRes e;
        e.setLoss(std::get<1>(result));
        e.setNum_example(std::get<0>(result));
        e.setAccuracy(std::get<2>(result));      
        return e;
    }

private:
    vision::models::ResNet18& net;
    int64_t num_epochs;
    DataLoader& train_loader;
    DataLoader& test_loader;
    torch::optim::Optimizer& optimizer;
    torch::Device device;
};
