#include "grpc_rere.h"

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

void create_node(const std::unique_ptr<flwr::proto::Fleet::Stub> &stub) {
  flwr::proto::CreateNodeRequest create_node_request;
  flwr::proto::CreateNodeResponse create_node_response;

  grpc::ClientContext context;
  grpc::Status status =
      stub->CreateNode(&context, create_node_request, &create_node_response);

  if (!status.ok()) {
    std::cerr << "CreateNode RPC failed: " << status.error_message()
              << std::endl;
    return;
  }

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

void delete_node(const std::unique_ptr<flwr::proto::Fleet::Stub> &stub) {
  auto node = get_node_from_store();
  if (!node) {
    return;
  }
  flwr::proto::DeleteNodeRequest delete_node_request;
  flwr::proto::DeleteNodeResponse delete_node_response;

  auto heap_node = new flwr::proto::Node(*node);
  delete_node_request.set_allocated_node(heap_node);

  grpc::ClientContext context;
  grpc::Status status =
      stub->DeleteNode(&context, delete_node_request, &delete_node_response);

  if (!status.ok()) {
    std::cerr << "DeleteNode RPC failed with status: " << status.error_message()
              << std::endl;
    delete heap_node; // Make sure to delete if status is not ok
    return;
  } else {
    delete_node_request.release_node(); // Release if status is ok
  }

  // TODO: Check if Node needs to be removed from local map
  // node_store.erase(node);
}

std::optional<flwr::proto::TaskIns>
receive(const std::unique_ptr<flwr::proto::Fleet::Stub> &stub) {
  auto node = get_node_from_store();
  if (!node) {
    return std::nullopt;
  }
  flwr::proto::PullTaskInsResponse response;
  flwr::proto::PullTaskInsRequest request;

  request.set_allocated_node(new flwr::proto::Node(*node));

  grpc::ClientContext context;
  grpc::Status status = stub->PullTaskIns(&context, request, &response);

  // Release ownership so that the heap_node won't be deleted when request
  // goes out of scope.
  request.release_node();

  if (!status.ok()) {
    std::cerr << "PullTaskIns RPC failed with status: "
              << status.error_message() << std::endl;
    return std::nullopt;
  }

  if (response.task_ins_list_size() > 0) {
    flwr::proto::TaskIns task_ins = response.task_ins_list().at(0);
    // TODO: Validate TaskIns

    {
      std::lock_guard<std::mutex> state_lock(state_mutex);
      state[KEY_TASK_INS] = task_ins;
    }

    return task_ins;
  } else {
    std::cerr << "TaskIns list is empty." << std::endl;
    return std::nullopt;
  }
}

void send(const std::unique_ptr<flwr::proto::Fleet::Stub> &stub,
          flwr::proto::TaskRes task_res) {
  auto node = get_node_from_store();
  if (!node) {
    return;
  }

  auto task_ins = get_current_task_ins();
  if (!task_ins) {
    return;
  }

  // TODO: Validate TaskIns

  flwr::proto::TaskRes new_task_res =
      configure_task_res(task_res, *task_ins, *node);

  flwr::proto::PushTaskResRequest request;
  *request.add_task_res_list() = new_task_res;
  flwr::proto::PushTaskResResponse response;

  grpc::ClientContext context;
  grpc::Status status = stub->PushTaskRes(&context, request, &response);

  if (!status.ok()) {
    std::cerr << "PushTaskRes RPC failed with status: "
              << status.error_message() << std::endl;
    return;
  } else {
    std::lock_guard<std::mutex> state_lock(state_mutex);
    state[KEY_TASK_INS].reset();
  }
}
