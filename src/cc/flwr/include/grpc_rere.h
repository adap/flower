/*************************************************************************************************
 *
 * @file start.h
 *
 * @brief Create a gRPC channel to connect to the server and enable message
 *communication
 *
 * @author Lekang Jiang
 *
 * @version 1.0
 *
 * @date 06/09/2021
 *
 *************************************************************************************************/

#ifndef GRPC_RERE_H
#define GRPC_RERE_H
#pragma once
#include "message_handler.h"
#include <grpcpp/grpcpp.h>

void create_node(const std::unique_ptr<flwr::proto::Fleet::Stub> &stub);
void delete_node(const std::unique_ptr<flwr::proto::Fleet::Stub> &stub);
void send(const std::unique_ptr<flwr::proto::Fleet::Stub> &stub,
          flwr::proto::TaskRes task_res);
std::optional<flwr::proto::TaskIns>
receive(const std::unique_ptr<flwr::proto::Fleet::Stub> &stub);

#endif
