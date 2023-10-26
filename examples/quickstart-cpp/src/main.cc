#include "simple_client.h"
#include "start.h"

int main(int argc, char **argv) {
  if (argc != 3 && argc != 4) {
    std::cout << "Client takes three mandatory arguments and one optional as "
                 "follows: "
              << std::endl;
    std::cout << "./client  CLIENT_ID  SERVER_URL [GRPC_MODE]" << std::endl;
    std::cout
        << "GRPC_MODE is optional and can be either 'bidi' (default) or 'rere'."
        << std::endl;
    std::cout << "Example: ./flwr_client 0 '127.0.0.1:8080' bidi" << std::endl;
    std::cout << "This is the same as: ./flwr_client 0 '127.0.0.1:8080'"
              << std::endl;
    return 0;
  }

  // Parsing arguments
  const std::string CLIENT_ID = argv[1];
  const std::string SERVER_URL = argv[2];

  // Populate local datasets
  std::vector<double> ms{3.5, 9.3}; //  b + m_0*x0 + m_1*x1
  double b = 1.7;
  std::cout << "Training set:" << std::endl;
  SyntheticDataset local_training_data = SyntheticDataset(ms, b, 1000);
  std::cout << std::endl;

  std::cout << "Validation set:" << std::endl;
  SyntheticDataset local_validation_data = SyntheticDataset(ms, b, 100);
  std::cout << std::endl;

  std::cout << "Test set:" << std::endl;
  SyntheticDataset local_test_data = SyntheticDataset(ms, b, 500);
  std::cout << std::endl;

  // Define a model
  LineFitModel model = LineFitModel(500, 0.01, ms.size());

  // Initialize TorchClient
  SimpleFlwrClient client(CLIENT_ID, model, local_training_data,
                          local_validation_data, local_test_data);

  // Define a server address
  std::string server_add = SERVER_URL;

  if (argc == 4 && std::string(argv[3]) == "rere") {
    std::cout << "Starting rere client" << std::endl;
    // Start rere client
    start::start_rere_client(server_add, &client);
  } else {
    std::cout << "Starting bidi client" << std::endl;
    // Start bidi client
    start::start_client(server_add, &client);
  }

  return 0;
}
