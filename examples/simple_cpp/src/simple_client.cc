#include 'simple_client.h'
/**
 * Validate the network on the entire test set
 *
 */
std::tuple<size_t, float, double> test(float alpha, float beta, std::vector<std::tuple<float, float>> )
{
    size_t num_samples = 0;
    size_t num_correct = 0;
    float running_loss = 0.0;

    // Perform Evaluation here (setting num_samples to avoid NaN)
    num_samples = 1; 
   // 

    auto mean_loss = running_loss / num_samples;
    auto accuracy = static_cast<double>(num_correct) / num_samples;

    std::cout << "Testset - Loss: " << mean_loss << ", Accuracy: " << accuracy << std::endl;

    return std::make_tuple(num_samples, running_loss, accuracy);
}

/**
 * Train the net work on the training set
 */
std::tuple<size_t, float, double> train(float alpha, float beta, std::vector<std::tuple<float, float>> ) 
{
    net->to(device);
    net->train();
    size_t num_samples = 0;
    size_t num_correct = 0;
    float running_loss = 0.0;

    // Iterate the data loader to yield batches from the dataset.
    num_correct = 1;
    // Training goes here
    auto mean_loss = running_loss / num_samples;
    auto accuracy = static_cast<double>(num_correct) / num_samples;

    return std::make_tuple(num_samples, mean_loss, accuracy);
}

/**
 * Initializer
 */
SimpleFlwrClient(std::string client_id, Dataset& trainset, Dataset& testset) : alpha(0.0), beta(0,0), train(trainset), testset(testset)){};

/**
 * Return the current local model parameters
 * Simple string are used for now to test communication, needs updates in the future
 */
flwr::ParametersRes SimpleFlwrClient::get_parameters() {
  // Serialize
  std::ostringstream stream;

  stream << alpha; // This is WRONG. Needs too serialize
  stream << beta; // This is WRONG. Needs too serialize

  std::list<std::string> tensors;
  tensors.push_back(stream.str());
  std::string tensor_str = "float";
  return flwr::Parameters(tensors, tensor_str); 
};

flwr::PropertiesRes SimpleFlwrClient::get_properties(flwr::PropertiesIns ins) {
  flwr::PropertiesRes p;
  p.setPropertiesRes(static_cast<flwr::Properties>(ins.getPropertiesIns()));
  return p;
}

/**
 * Refine the provided weights using the locally held dataset
 * Simple settings are used for testing, needs updates in the future
 */
flwr::FitRes SimpleFlwrClient::fit(flwr::FitIns ins) {
  flwr::FitRes resp;
  flwr::Parameters p = ins.getParameters();
  std::list<std::string> s = p.getTensors();
  //std::cout << s.empty() << std::endl;
  if (s.empty() == 0){
    std::istringstream stream(s.front());  
    alpha =0.0;  //Deserialize it here
    beta = 0.0; //Deserialize it here
  }
  std::tuple<size_t, float, double> result = train(net, train_loader, optimizer, device);
  resp.setParameters(this->get_parameters().getParameters());
  resp.setNum_example(std::get<0>(result));
  
  return resp;
};

/**
 * Evaluate the provided weights using the locally held dataset
 * Needs updates in the future
 */
flwr::EvaluateRes SimpleFlwrClient::evaluate(flwr::EvaluateIns ins) {
  flwr::EvaluateRes resp;
  flwr::Parameters p = ins.getParameters();
  std::list<std::string> s = p.getTensors();
  if (s.empty() == 0){
    std::istringstream stream(s.front());
  
    torch::load(net, stream);
  }
  std::tuple<size_t, float, double> result = test(alpha, beta);

  resp.setNum_example(std::get<0>(result));
  resp.setLoss(std::get<1>(result));

  return resp;

};
