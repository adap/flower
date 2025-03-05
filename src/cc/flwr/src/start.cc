#include "start.h"

// cppcheck-suppress unusedFunction
void start::start_client(std::string server_address, flwr_local::Client *client,
                         int grpc_max_message_length) {

  gRPCRereCommunicator communicator(server_address, grpc_max_message_length);

  while (true) {
    int sleep_duration = 0;

    create_node(&communicator);

    while (true) {
      auto task_ins = receive(&communicator);
      if (!task_ins) {
        std::this_thread::sleep_for(std::chrono::seconds(3));
        continue;
      }

      auto [task_res, sleep_duration, keep_going] =
          handle_task(client, task_ins.value());

      send(&communicator, task_res);
      if (!keep_going) {
        break;
      }
    }

    delete_node(&communicator);
    if (sleep_duration == 0) {
      std::cout << "Disconnect and shut down." << std::endl;
      break;
    }

    std::cout << "Disconnect, then re-establish connection after"
              << sleep_duration << "second(s)" << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(sleep_duration));

    if (sleep_duration == 0) {
      std::cout << "Disconnect and shut down." << std::endl;
      break;
    }
  }
}
