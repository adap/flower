/***********************************************************************************************************
 * 
 * @file torch_client.cc
 * 
 * @brief Implementation of TorchClient methods
 *
 * @version 1.0
 *
 * @date 06/09/2021
 *
 * ********************************************************************************************************/

#include "torch_client.h"

/**
 * Initializer
 */
template<typename DataLoader>
TorchClient<DataLoader>::TorchClient(
  std::string client_id,
  vision::models::ResNet18& net,
  DataLoader& train_loader,
  DataLoader& test_loader,
  torch::optim::Optimizer& optimizer,
  torch::Device device) : net(net), train_loader(train_loader), test_loader(test_loader), optimizer(optimizer), device(device){
};

/**
 * Return the current local model parameters
 * Simple string are used for now to test communication, needs updates in the future
 */
template<typename DataLoader>
flwr::ParametersRes TorchClient<DataLoader>::get_parameters() {
  // Serialize
  //std::ostringstream stream;
  //torch::save(net, stream);
  //std::string str = stream.str();
  //const char* chr = str.c_str();
  //std::list<std::string> tensors;
  //tensors.push_back(str);
  std::list<std::string> tensors;
  std::string tensor = "my_bytes";
  std::string type_str = "float32";
  tensors.push_back(tensor);
  tensors.push_back(type_str);
  return flwr::Parameters(tensors, "Pytorch example");
};

/**
 * Refine the provided weights using the locally held dataset
 * Simple settings are used for testing, needs updates in the future
 */
template<typename DataLoader>
flwr::FitRes TorchClient<DataLoader>::fit(flwr::FitIns ins) {
  // int num_samples = train(net, train_loader, optimizer, device);
  flwr::FitRes resp;

  resp.setParameters(this->get_parameters().getParameters());
  resp.setNum_example(30);
  return resp;
};

/**
 * Evaluate the provided weights using the locally held dataset
 * Needs updates in the future
 */
template<typename DataLoader>
flwr::EvaluateRes TorchClient<DataLoader>::evaluate(flwr::EvaluateIns ins) {
 flwr::EvaluateRes resp;  

  return resp;
};


