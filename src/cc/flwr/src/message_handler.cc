#include "message_handler.h"

std::tuple<ClientMessage, int>
_reconnect(ServerMessage_Reconnect reconnect_msg) {
  // Determine the reason for sending Disconnect message
  Reason reason = Reason::ACK;
  int sleep_duration = 0;
  if (reconnect_msg.seconds() != 0) {
    reason = Reason::RECONNECT;
    sleep_duration = reconnect_msg.seconds();
  }

  // Build Disconnect message
  ClientMessage_Disconnect disconnect;
  disconnect.set_reason(reason);
  ClientMessage cm;
  *cm.mutable_disconnect_res() = disconnect;

  return std::make_tuple(cm, sleep_duration);
}

ClientMessage _get_parameters(flwr_local::Client *client) {
  ClientMessage cm;
  *(cm.mutable_get_parameters_res()) =
      parameters_res_to_proto(client->get_parameters());
  return cm;
}

ClientMessage _fit(flwr_local::Client *client, ServerMessage_FitIns fit_msg) {
  // Deserialize fit instruction
  flwr_local::FitIns fit_ins = fit_ins_from_proto(fit_msg);
  // Perform fit
  flwr_local::FitRes fit_res = client->fit(fit_ins);
  // Serialize fit result
  ClientMessage cm;
  *cm.mutable_fit_res() = fit_res_to_proto(fit_res);
  return cm;
}

ClientMessage _evaluate(flwr_local::Client *client,
                        ServerMessage_EvaluateIns evaluate_msg) {
  // Deserialize evaluate instruction
  flwr_local::EvaluateIns evaluate_ins = evaluate_ins_from_proto(evaluate_msg);
  // Perform evaluation
  flwr_local::EvaluateRes evaluate_res = client->evaluate(evaluate_ins);
  // Serialize evaluate result
  ClientMessage cm;
  *cm.mutable_evaluate_res() = evaluate_res_to_proto(evaluate_res);
  return cm;
}

std::tuple<ClientMessage, int, bool> handle(flwr_local::Client *client,
                                            ServerMessage server_msg) {
  if (server_msg.has_reconnect_ins()) {
    std::tuple<ClientMessage, int> rec = _reconnect(server_msg.reconnect_ins());
    return std::make_tuple(std::get<0>(rec), std::get<1>(rec), false);
  }
  if (server_msg.has_get_parameters_ins()) {
    return std::make_tuple(_get_parameters(client), 0, true);
  }
  if (server_msg.has_fit_ins()) {
    return std::make_tuple(_fit(client, server_msg.fit_ins()), 0, true);
  }
  if (server_msg.has_evaluate_ins()) {
    return std::make_tuple(_evaluate(client, server_msg.evaluate_ins()), 0,
                           true);
  }
  throw "Unkown server message";
}

std::tuple<flwr::proto::TaskRes, int, bool>
handle_task(flwr_local::Client *client, const flwr::proto::TaskIns &task_ins) {

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  if (!task_ins.task().has_legacy_server_message()) {
    // TODO: Handle SecureAggregation
    throw std::runtime_error("Task still needs legacy server message");
  }
  ServerMessage server_msg = task_ins.task().legacy_server_message();
#pragma GCC diagnostic pop

  std::tuple<ClientMessage, int, bool> legacy_res = handle(client, server_msg);
  std::unique_ptr<ClientMessage> client_message =
      std::make_unique<ClientMessage>(std::get<0>(legacy_res));

  flwr::proto::TaskRes task_res;
  task_res.set_task_id("");
  task_res.set_group_id("");
  task_res.set_workload_id(0);

  std::unique_ptr<flwr::proto::Task> task =
      std::make_unique<flwr::proto::Task>();

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  task->set_allocated_legacy_client_message(
      client_message.release()); // Ownership transferred to `task`
#pragma GCC diagnostic pop

  task_res.set_allocated_task(task.release());
  return std::make_tuple(task_res, std::get<1>(legacy_res),
                         std::get<2>(legacy_res));
}
