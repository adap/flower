#include <iostream>
#include <memory>
#include <string>
#include "simple_client.h"
#include "start.h"

int  main(int argc, char** argv){
  if (argc != 3){
    std::cout << "Client takes three arguments as follows: " << std::endl;
    std::cout << "./client  CLIENT_ID  SERVER_URL" << std::endl;
    std::cout << "Example: ./client  0 localhost:8888" << std::endl;
    return 0;
  }

  // Parsing arguments
  const std::string CLIENT_ID  = argv[1];
  const std::string SERVER_URL = argv[2];

  // Populate training set and validation set
  const size_t NUM_SAMPLES = 1000;
  const float TRUE_ALPHA = 0.5;
  const float TRUE_BETA = 1.5;
  Dataset trainset, testset;  // Defined in simple_client.h
  //Initialize them here 

  // Initialize TorchClient
  SimpleFlwrClient client(CLIENT_ID, trainset, testset);
    
  // Define a server address
  std::string server_add = SERVER_URL;
    
  // Start client
  start s;
  s.start_client(server_add, &client);
    
  return 0;
}

