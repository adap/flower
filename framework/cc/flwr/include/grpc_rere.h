/*************************************************************************************************
 *
 * @file grpc-rere.h
 *
 * @brief Provide functions for establishing gRPC request-response communication
 *
 * @author The Flower Authors
 *
 * @version 1.0
 *
 * @date 06/11/2023
 *
 *************************************************************************************************/

#ifndef GRPC_RERE_H
#define GRPC_RERE_H
#pragma once
#include "communicator.h"
#include "flwr/proto/fleet.grpc.pb.h"
#include "message_handler.h"
#include <grpcpp/grpcpp.h>

class gRPCRereCommunicator : public Communicator {
public:
  gRPCRereCommunicator(std::string server_address, int grpc_max_message_length);

  bool send_create_node(flwr::proto::CreateNodeRequest request,
                        flwr::proto::CreateNodeResponse *response);

  bool send_delete_node(flwr::proto::DeleteNodeRequest request,
                        flwr::proto::DeleteNodeResponse *response);

  bool send_pull_task_ins(flwr::proto::PullTaskInsRequest request,
                          flwr::proto::PullTaskInsResponse *response);

  bool send_push_task_res(flwr::proto::PushTaskResRequest request,
                          flwr::proto::PushTaskResResponse *response);

private:
  std::unique_ptr<flwr::proto::Fleet::Stub> stub;
};

#endif
