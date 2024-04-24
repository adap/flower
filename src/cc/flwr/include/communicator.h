#ifndef COMMUNICATOR_H
#define COMMUNICATOR_H

#include "flwr/proto/fleet.pb.h"
#include <chrono>
#include <optional>

class Communicator {
public:
  virtual bool send_create_node(flwr::proto::CreateNodeRequest request,
                                flwr::proto::CreateNodeResponse *response) = 0;

  virtual bool send_delete_node(flwr::proto::DeleteNodeRequest request,
                                flwr::proto::DeleteNodeResponse *response) = 0;

  virtual bool
  send_pull_task_ins(flwr::proto::PullTaskInsRequest request,
                     flwr::proto::PullTaskInsResponse *response) = 0;

  virtual bool
  send_push_task_res(flwr::proto::PushTaskResRequest request,
                     flwr::proto::PushTaskResResponse *response) = 0;
};

void create_node(Communicator *communicator);
void delete_node(Communicator *communicator);
void send(Communicator *communicator, flwr::proto::TaskRes task_res);
std::optional<flwr::proto::TaskIns> receive(Communicator *communicator);

#endif
