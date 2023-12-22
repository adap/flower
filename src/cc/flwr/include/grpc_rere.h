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
#include "message_handler.h"
#include "task_handler.h"
#include <grpcpp/grpcpp.h>

void create_node(const std::unique_ptr<flwr::proto::Fleet::Stub> &stub);
void delete_node(const std::unique_ptr<flwr::proto::Fleet::Stub> &stub);
void send(const std::unique_ptr<flwr::proto::Fleet::Stub> &stub,
          flwr::proto::TaskRes task_res);
std::optional<flwr::proto::TaskIns>
receive(const std::unique_ptr<flwr::proto::Fleet::Stub> &stub);

#endif
