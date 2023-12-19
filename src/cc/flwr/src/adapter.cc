#include "adapter.h"

enum TaskType { kTraining, kEvaluation };

// Equivalent to BuildGetTasksPayload
std::optional<std::string> getPullTaskInsRequestString() {
  flwr::proto::PullTaskInsRequest request;
  request.set_allocated_node(new flwr::proto::Node());

  std::string requestString;
  const bool isSuccess = request.SerializeToString(&requestString);

  if (!isSuccess) {
    return std::nullopt;
  }

  return requestString;
}

// Equivalent to ParseTaskListFromResponseBody
std::optional<flwr::proto::TaskIns>
getTaskInsFromResponseBody(const std::string &responseBody) {
  flwr::proto::PullTaskInsResponse response;
  if (!response.ParseFromString(responseBody)) {
    std::cerr << "Failed to parse response body." << std::endl;
    return std::nullopt;
  }
  if (response.task_ins_list_size() > 0) {
    flwr::proto::TaskIns task_ins = response.task_ins_list().at(0);
    if (validate_task_ins(task_ins, true)) {
      return task_ins;
    }
  }
  std::cerr << "TaskIns list is empty." << std::endl;
  return std::nullopt;
}

std::optional<std::string> BuildUploadTaskResultsPayload(
    const TaskType task_type, const std::string task_id,
    const std::string group_id, const int64_t workload_id,
    const int64_t dataset_size, const flwr_local::Parameters parameters,
    const flwr_local::Metrics metrics,
    const std::optional<float_t> loss = std::nullopt) {

  flwr::proto::Task flower_task;

  flwr::proto::ClientMessage client_message;
  switch (task_type) {
  case kTraining: {
    flwr::proto::ClientMessage_FitRes fit_res;
    fit_res.set_num_examples(dataset_size);
    *fit_res.mutable_parameters() = parameters_to_proto(parameters);
    if (!metrics.empty()) {
      *fit_res.mutable_metrics() = metrics_to_proto(metrics);
    }
    *client_message.mutable_fit_res() = fit_res;
    break;
  }
  case kEvaluation: {
    flwr::proto::ClientMessage_EvaluateRes eval_res;
    eval_res.set_num_examples(dataset_size);
    if (!loss.has_value()) {
      return std::nullopt;
    }
    eval_res.set_loss(loss.value());
    if (!metrics.empty()) {
      *eval_res.mutable_metrics() = metrics_to_proto(metrics);
    }
    *client_message.mutable_evaluate_res() = eval_res;
    break;
  }
    return std::nullopt;
  }
  flower_task.add_ancestry(task_id);

  flwr::proto::Node producer_node;
  producer_node.set_node_id(0);
  producer_node.set_anonymous(true);

  flwr::proto::Node consumer_node;
  consumer_node.set_node_id(0);
  consumer_node.set_anonymous(true);

  *flower_task.mutable_consumer() = consumer_node;
  *flower_task.mutable_producer() = producer_node;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  *flower_task.mutable_legacy_client_message() = client_message;
#pragma GCC diagnostic pop

  flwr::proto::PushTaskResRequest task_results;
  flwr::proto::TaskRes *task_result = task_results.add_task_res_list();
  task_result->set_task_id("");
  task_result->set_group_id(group_id);
  task_result->set_workload_id(workload_id);
  *task_result->mutable_task() = flower_task;

  std::string result_payload;
  task_results.SerializeToString(&result_payload);

  return result_payload;
}
