#include "message_handler.h"
#include "flwr/proto/task.pb.h"
#include <variant>

std::tuple<flwr_local::RecordSet, int>
_reconnect(flwr::proto::RecordSet proto_recordset) {

  // Determine the reason for sending Disconnect message
  flwr::proto::Reason reason = flwr::proto::Reason::ACK;
  int sleep_duration = 0;

  // Build Disconnect message
  return std::make_tuple(
      flwr_local::RecordSet({}, {}, {{"config", {{"reason", reason}}}}),
      sleep_duration);
}

flwr_local::RecordSet _get_parameters(flwr_local::Client *client) {
  return recordset_from_get_parameters_res(client->get_parameters());
}

flwr_local::RecordSet _fit(flwr_local::Client *client,
                           flwr::proto::RecordSet proto_recordset) {
  flwr_local::RecordSet recordset = recordset_from_proto(proto_recordset);
  flwr_local::FitIns fit_ins = recordset_to_fit_ins(recordset, true);
  // Perform fit
  flwr_local::FitRes fit_res = client->fit(fit_ins);

  flwr_local::RecordSet out_recordset = recordset_from_fit_res(fit_res);
  return out_recordset;
}

flwr_local::RecordSet _evaluate(flwr_local::Client *client,
                                flwr::proto::RecordSet proto_recordset) {
  flwr_local::RecordSet recordset = recordset_from_proto(proto_recordset);
  flwr_local::EvaluateIns evaluate_ins =
      recordset_to_evaluate_ins(recordset, true);
  // Perform evaluation
  flwr_local::EvaluateRes evaluate_res = client->evaluate(evaluate_ins);

  flwr_local::RecordSet out_recordset =
      recordset_from_evaluate_res(evaluate_res);
  return out_recordset;
}

std::tuple<flwr_local::RecordSet, int, bool> handle(flwr_local::Client *client,
                                                    flwr::proto::Task task) {
  if (task.task_type() == "reconnect") {
    std::tuple<flwr_local::RecordSet, int> rec = _reconnect(task.recordset());
    return std::make_tuple(std::get<0>(rec), std::get<1>(rec), false);
  }
  if (task.task_type() == "get_parameters") {
    return std::make_tuple(_get_parameters(client), 0, true);
  }
  if (task.task_type() == "train") {
    return std::make_tuple(_fit(client, task.recordset()), 0, true);
  }
  if (task.task_type() == "evaluate") {
    return std::make_tuple(_evaluate(client, task.recordset()), 0, true);
  }
  throw "Unkown server message";
}

std::tuple<flwr::proto::TaskRes, int, bool>
handle_task(flwr_local::Client *client, const flwr::proto::TaskIns &task_ins) {
  flwr::proto::Task received_task = task_ins.task();

  std::tuple<flwr_local::RecordSet, int, bool> legacy_res =
      handle(client, received_task);
  auto conf_records =
      recordset_from_proto(recordset_to_proto(std::get<0>(legacy_res)))
          .getConfigsRecords();
  for (const auto &[key, value] : conf_records) {
    std::cout << "initial: " << key << std::endl;
    for (const auto &[inner_key, inner_value] : conf_records[key]) {
      std::string print_key = inner_key;
      std::visit(
          [&](const auto &v) {
            using T = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<T, int> || std::is_same_v<T, double> ||
                          std::is_same_v<T, bool> ||
                          std::is_same_v<T, std::string>) {
              std::cout << "Key: " << print_key << ", "
                        << "value: " << v << std::endl;
            } else if constexpr (std::is_same_v<T, std::vector<int>> ||
                                 std::is_same_v<T, std::vector<double>> ||
                                 std::is_same_v<T, std::vector<bool>> ||
                                 std::is_same_v<T, std::vector<std::string>>) {
              std::cout << "Values: ";
              for (const auto &elem : v) {
                std::cout << elem << " ";
              }
              std::cout << std::endl;
            }
          },
          inner_value);
    }
  }

  flwr::proto::TaskRes task_res;
  task_res.set_task_id(task_ins.task_id());
  task_res.set_group_id(task_ins.group_id());
  task_res.set_run_id(task_ins.run_id());

  std::unique_ptr<flwr::proto::Task> task =
      std::make_unique<flwr::proto::Task>();

  std::unique_ptr<flwr::proto::RecordSet> proto_recordset_ptr =
      std::make_unique<flwr::proto::RecordSet>(
          recordset_to_proto(std::get<0>(legacy_res)));

  task->set_allocated_recordset(proto_recordset_ptr.release());
  task->set_task_type(received_task.task_type());
  task->set_allocated_consumer(
      std::make_unique<flwr::proto::Node>(received_task.producer()).release());
  task->set_allocated_producer(
      std::make_unique<flwr::proto::Node>(received_task.consumer()).release());

  task_res.set_allocated_task(task.release());

  std::tuple<flwr::proto::TaskRes, int, bool> tuple = std::make_tuple(
      task_res, std::get<1>(legacy_res), std::get<2>(legacy_res));

  return tuple;
}
