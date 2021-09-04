/*************************************************************************************************
 *
 * @file start.h
 *
 * @brief Create a gRPC channel to connect to the server and enable message communication
 *
 * @autheor Lekang Jiang
 *
 * @version 1.0
 *
 * @date 04/09/2021
 *
 *************************************************************************************************/

#pragma once
//#include "transport.grpc.pb.h"
//#include "client.h"
#include "message_handler.h"
using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;
using grpc::ClientReaderWriter;
using flower::transport::ClientMessage;
using flower::transport::ServerMessage;
using flower::transport::FlowerService;

int GRPC_MAX_MESSAGE_LENGTH = 536870912;  //  == 512 * 1024 * 1024

/**
 * @brief Start a C++ Flower Client which connects to a gRPC server
 * @param  server_address 
 *                        The IPv6 address of the server. If the Flower server runs on the same 
 *                        machine on port 8080, then `server_address` would be `"[::]:8080"`.
 *
 *         client
 *                        An implementation of the abstract base class `flwr::Client`
 *
 *         grpc_max_message_length
 *                        int (default: 536_870_912, this equals 512MB).
 *                        The maximum length of gRPC messages that can be exchanged with the
 *                        Flower server. The default should be sufficient for most models.
 *                        Users who train very large models might need to increase this
 *                        value. Note that the Flower server needs to be started with the
 *                        same value (see `flwr.server.start_server`), otherwise it will not
 *                        know about the increased limit and block larger messages.
 *
 */
void start_client(std::string server_address, flwr::Client* client, int grpc_max_message_length = GRPC_MAX_MESSAGE_LENGTH) {

    while (true) {
        int sleep_duration = 0;

        // Set channel parameters
        grpc::ChannelArguments args;
        args.SetMaxReceiveMessageSize(grpc_max_message_length);
        args.SetMaxSendMessageSize(grpc_max_message_length);

        // Establish an insecure gRPC connection to a gRPC server
        std::shared_ptr<Channel> channel = grpc::CreateCustomChannel(server_address, grpc::InsecureChannelCredentials(), args);

        // Create stub
        std::unique_ptr<FlowerService::Stub> stub_ = FlowerService::NewStub(channel);

        // Read and write messages
        ClientContext context;
        std::shared_ptr<ClientReaderWriter<ClientMessage, ServerMessage>> reader_writer(stub_->Join(&context));
        ServerMessage sm;
        while (reader_writer->Read(&sm)) {
            std::cout << "Got message type: " << sm.GetTypeName() << " content: " << sm.DebugString() << std::endl;
            std::tuple<ClientMessage, int, bool> receive = handle(client, sm);
            //std::cout << "DEBUG: handle done" << std::endl;
            sleep_duration = std::get<1>(receive);
            //std::cout << "DEBUG: begin write" << std::endl;
            std::cout << "Send message type: " << std::get<0>(receive).GetTypeName() << " content: " << std::get<0>(receive).DebugString() << std::endl;
            reader_writer->Write(std::get<0>(receive));
            std::cout << "DEBUG: one time read&write done\n\n\n" << std::endl;
            if (std::get<2>(receive) == false) {
                break;
            }
        }
        reader_writer->WritesDone();

        // Check connection status
        Status status = reader_writer->Finish();
        if (!status.ok()) {
            std::cout << "RouteChat rpc failed." << std::endl;
        }

        if (sleep_duration == 0) {
            std::cout << "Disconnect and shut down." << std::endl;
            break;
        }

        // Sleep and reconnect afterwards
        std::cout << "Disconnect, then re-establish connection after" << sleep_duration << "second(s)" << std::endl;
        //Sleep(sleep_duration * 1000);
    }
}
