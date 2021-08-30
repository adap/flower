#include "torch_client.h"

TorchClient::TorchClient(int64_t client_id, std::string connection_string) : client_id(client_id), connection_string(connection_string) {

};

flwr::Parameters TorchClient::get_parameters() {
  // Serialize
  //std::ostringstream stream;
  //torch::save(net, stream);
  //std::string str = stream.str();
  //const char* chr = str.c_str();
  //std::list<std::string> tensors;
  //tensors.push_back(str);
  std::string tensors = "my_bytes";
  std::string type_str = "float32"
  return flwr::Parameters(tensors, "Pytorch example");
        
}

virtual FitRes TorchClient::fit(flwr::FitIns ins) override {
  clock_t startTime, endTime;
  //startTime = clock();
  // int num_samples = train(net, train_loader, optimizer, device);
  endTime = clock();
  FitRes resp;
  resp.setParameters(getModel());
  resp.setNum_example(30);
  return resp;
}
