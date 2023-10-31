#include "start.h"

// cppcheck-suppress unusedFunction
void start::start_client(std::string server_address, flwr_local::Client *client,
                         int grpc_max_message_length) {
  while (true) {
    int sleep_duration = 0;

    // Set channel parameters
    grpc::ChannelArguments args;
    args.SetMaxReceiveMessageSize(grpc_max_message_length);
    args.SetMaxSendMessageSize(grpc_max_message_length);

    // Establish an insecure gRPC connection to a gRPC server
    std::shared_ptr<grpc::Channel> channel = grpc::CreateCustomChannel(
        server_address, grpc::InsecureChannelCredentials(), args);

    // Create stub
    std::unique_ptr<flwr::proto::FlowerService::Stub> stub_ =
        flwr::proto::FlowerService::NewStub(channel);

    // Read and write messages
    grpc::ClientContext context;
    std::shared_ptr<grpc::ClientReaderWriter<flwr::proto::ClientMessage,
                                             flwr::proto::ServerMessage>>
        reader_writer(stub_->Join(&context));
    ServerMessage sm;
    while (reader_writer->Read(&sm)) {
      std::tuple<ClientMessage, int, bool> receive = handle(client, sm);
      sleep_duration = std::get<1>(receive);
      reader_writer->Write(std::get<0>(receive));
      if (std::get<2>(receive) == false) {
        break;
      }
    }
    reader_writer->WritesDone();

    // Check connection status
    grpc::Status status = reader_writer->Finish();

    if (sleep_duration == 0) {
      std::cout << "Disconnect and shut down." << std::endl;
      break;
    }

    // Sleep and reconnect afterwards
    // std::cout << "Disconnect, then re-establish connection after" <<
    // sleep_duration << "second(s)" << std::endl; Sleep(sleep_duration * 1000);
  }
}

// cppcheck-suppress unusedFunction
void start::start_rere_client(std::string server_address,
                              flwr_local::Client *client,
                              int grpc_max_message_length) {
  while (true) {
    int sleep_duration = 0;

    // Set channel parameters
    grpc::ChannelArguments args;
    args.SetMaxReceiveMessageSize(grpc_max_message_length);
    args.SetMaxSendMessageSize(grpc_max_message_length);

    // Establish an insecure gRPC connection to a gRPC server
    std::shared_ptr<grpc::Channel> channel = grpc::CreateCustomChannel(
        server_address, grpc::InsecureChannelCredentials(), args);

    // Create stub
    std::unique_ptr<flwr::proto::Fleet::Stub> stub_ =
        flwr::proto::Fleet::NewStub(channel);

    // Read and write messages

    create_node(stub_);

    while (true) {
      auto task_ins = receive(stub_);
      if (!task_ins) {
        std::this_thread::sleep_for(std::chrono::seconds(3));
        continue;
      }
      auto [task_res, sleep_duration, keep_going] =
          handle_task(client, task_ins.value());
      send(stub_, task_res);
      if (!keep_going) {
        break;
      }
    }

    delete_node(stub_);
    if (sleep_duration == 0) {
      std::cout << "Disconnect and shut down." << std::endl;
      break;
    }

    std::cout << "Disconnect, then re-establish connection after"
              << sleep_duration << "second(s)" << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(sleep_duration));

    if (sleep_duration == 0) {
      std::cout << "Disconnect and shut down." << std::endl;
      break;
    }
  }
}
