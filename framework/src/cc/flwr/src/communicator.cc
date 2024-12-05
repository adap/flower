#include "communicator.h"

const std::string KEY_NODE = "node";
const std::string KEY_TASK_INS = "current_task_ins";

std::map<std::string, std::optional<flwr::proto::Node>> node_store;
std::map<std::string, std::optional<flwr::proto::TaskIns>> state;

std::mutex node_store_mutex;
std::mutex state_mutex;

std::optional<flwr::proto::Node> get_node_from_store() {
  std::lock_guard<std::mutex> lock(node_store_mutex);
  auto node = node_store.find(KEY_NODE);
  if (node == node_store.end() || !node->second.has_value()) {
    std::cerr << "Node instance missing" << std::endl;
    return std::nullopt;
  }
  return node->second;
}

bool validate_task_ins(const flwr::proto::TaskIns &task_ins,
                       const bool discard_reconnect_ins) {
  return task_ins.has_task() && task_ins.task().has_recordset();
}

bool validate_task_res(const flwr::proto::TaskRes &task_res) {
  // Retrieve initialized fields in TaskRes
  return true;
}

flwr::proto::TaskRes
configure_task_res(const flwr::proto::TaskRes &task_res,
                   const flwr::proto::TaskIns &ref_task_ins,
                   const flwr::proto::Node &producer) {
  flwr::proto::TaskRes result_task_res;

  // Setting scalar fields
  result_task_res.set_task_id("");
  result_task_res.set_group_id(ref_task_ins.group_id());
  result_task_res.set_run_id(ref_task_ins.run_id());

  // Merge the task from the input task_res
  *result_task_res.mutable_task() = task_res.task();

  // Construct and set the producer and consumer for the task
  std::unique_ptr<flwr::proto::Node> new_producer =
      std::make_unique<flwr::proto::Node>(producer);
  result_task_res.mutable_task()->set_allocated_producer(
      new_producer.release());

  std::unique_ptr<flwr::proto::Node> new_consumer =
      std::make_unique<flwr::proto::Node>(ref_task_ins.task().producer());
  result_task_res.mutable_task()->set_allocated_consumer(
      new_consumer.release());

  // Set ancestry in the task
  result_task_res.mutable_task()->add_ancestry(ref_task_ins.task_id());

  return result_task_res;
}

void delete_node_from_store() {
  std::lock_guard<std::mutex> lock(node_store_mutex);
  auto node = node_store.find(KEY_NODE);
  if (node == node_store.end() || !node->second.has_value()) {
    node_store.erase(node);
  }
}

std::optional<flwr::proto::TaskIns> get_current_task_ins() {
  std::lock_guard<std::mutex> state_lock(state_mutex);
  auto current_task_ins = state.find(KEY_TASK_INS);
  if (current_task_ins == state.end() ||
      !current_task_ins->second.has_value()) {
    std::cerr << "No current TaskIns" << std::endl;
    return std::nullopt;
  }
  return current_task_ins->second;
}

void create_node(Communicator *communicator) {
  flwr::proto::CreateNodeRequest create_node_request;
  flwr::proto::CreateNodeResponse create_node_response;

  create_node_request.set_ping_interval(300.0);

  communicator->send_create_node(create_node_request, &create_node_response);

  // Validate the response
  if (!create_node_response.has_node()) {
    std::cerr << "Received response does not contain a node." << std::endl;
    return;
  }

  {
    std::lock_guard<std::mutex> lock(node_store_mutex);
    node_store[KEY_NODE] = create_node_response.node();
  }
}

void delete_node(Communicator *communicator) {
  auto node = get_node_from_store();
  if (!node) {
    return;
  }
  flwr::proto::DeleteNodeRequest delete_node_request;
  flwr::proto::DeleteNodeResponse delete_node_response;

  auto heap_node = new flwr::proto::Node(*node);
  delete_node_request.set_allocated_node(heap_node);

  if (!communicator->send_delete_node(delete_node_request,
                                      &delete_node_response)) {
    delete heap_node; // Make sure to delete if status is not ok
    return;
  } else {
    delete_node_request.release_node(); // Release if status is ok
  }

  delete_node_from_store();
}

std::optional<flwr::proto::TaskIns> receive(Communicator *communicator) {
  auto node = get_node_from_store();
  if (!node) {
    return std::nullopt;
  }
  flwr::proto::PullTaskInsResponse response;
  flwr::proto::PullTaskInsRequest request;

  request.set_allocated_node(new flwr::proto::Node(*node));

  bool success = communicator->send_pull_task_ins(request, &response);

  // Release ownership so that the heap_node won't be deleted when request
  // goes out of scope.
  request.release_node();

  if (!success) {
    return std::nullopt;
  }

  if (response.task_ins_list_size() > 0) {
    flwr::proto::TaskIns task_ins = response.task_ins_list().at(0);
    if (validate_task_ins(task_ins, true)) {
      std::lock_guard<std::mutex> state_lock(state_mutex);
      state[KEY_TASK_INS] = task_ins;
      return task_ins;
    }
  }
  std::cerr << "TaskIns list is empty." << std::endl;
  return std::nullopt;
}

void send(Communicator *communicator, flwr::proto::TaskRes task_res) {
  auto node = get_node_from_store();
  if (!node) {
    return;
  }

  auto task_ins = get_current_task_ins();
  if (!task_ins) {
    return;
  }

  if (!validate_task_res(task_res)) {
    std::cerr << "TaskRes is invalid" << std::endl;
    std::lock_guard<std::mutex> state_lock(state_mutex);
    state[KEY_TASK_INS].reset();
    return;
  }

  flwr::proto::TaskRes new_task_res =
      configure_task_res(task_res, *task_ins, *node);

  flwr::proto::PushTaskResRequest request;
  *request.add_task_res_list() = new_task_res;
  flwr::proto::PushTaskResResponse response;

  communicator->send_push_task_res(request, &response);

  {
    std::lock_guard<std::mutex> state_lock(state_mutex);
    state[KEY_TASK_INS].reset();
  }
}
