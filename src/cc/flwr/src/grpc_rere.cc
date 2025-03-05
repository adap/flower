#include "grpc_rere.h"
#include "flwr/proto/fleet.grpc.pb.h"

gRPCRereCommunicator::gRPCRereCommunicator(std::string server_address,
                                           int grpc_max_message_length) {
  grpc::ChannelArguments args;
  args.SetMaxReceiveMessageSize(grpc_max_message_length);
  args.SetMaxSendMessageSize(grpc_max_message_length);

  // Establish an insecure gRPC connection to a gRPC server
  std::shared_ptr<grpc::Channel> channel = grpc::CreateCustomChannel(
      server_address, grpc::InsecureChannelCredentials(), args);

  // Create stub
  stub = flwr::proto::Fleet::NewStub(channel);
}

bool gRPCRereCommunicator::send_create_node(
    flwr::proto::CreateNodeRequest request,
    flwr::proto::CreateNodeResponse *response) {
  grpc::ClientContext context;
  grpc::Status status = stub->CreateNode(&context, request, response);
  if (!status.ok()) {
    std::cerr << "CreateNode RPC failed: " << status.error_message()
              << std::endl;
    return false;
  }

  return true;
}

bool gRPCRereCommunicator::send_delete_node(
    flwr::proto::DeleteNodeRequest request,
    flwr::proto::DeleteNodeResponse *response) {
  grpc::ClientContext context;
  grpc::Status status = stub->DeleteNode(&context, request, response);

  if (!status.ok()) {
    std::cerr << "DeleteNode RPC failed with status: " << status.error_message()
              << std::endl;
    return false;
  }

  return true;
}

bool gRPCRereCommunicator::send_pull_task_ins(
    flwr::proto::PullTaskInsRequest request,
    flwr::proto::PullTaskInsResponse *response) {
  grpc::ClientContext context;
  grpc::Status status = stub->PullTaskIns(&context, request, response);

  if (!status.ok()) {
    std::cerr << "PullTaskIns RPC failed with status: "
              << status.error_message() << std::endl;
    return false;
  }

  return true;
}

bool gRPCRereCommunicator::send_push_task_res(
    flwr::proto::PushTaskResRequest request,
    flwr::proto::PushTaskResResponse *response) {
  grpc::ClientContext context;
  grpc::Status status = stub->PushTaskRes(&context, request, response);

  if (!status.ok()) {
    std::cerr << "PushTaskRes RPC failed with status: "
              << status.error_message() << std::endl;
    return false;
  }

  return true;
}
