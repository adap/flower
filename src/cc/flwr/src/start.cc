#include "../include/start.h"

void start::start_client(std::string server_address, flwr::Client* client, int grpc_max_message_length) {

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
	    std::cout << "Got message type: " << sm.GetTypeName()<< std::endl;
	    //std::cout << "Got message type: " << sm.GetTypeName() << " content: " << sm.DebugString() << std::endl;
            std::tuple<ClientMessage, int, bool> receive = handle(client, sm);
            //std::cout << "DEBUG: handle done" << std::endl;
            sleep_duration = std::get<1>(receive);
            //std::cout << "DEBUG: begin write" << std::endl;
            //std::cout << "Send message type: " << std::get<0>(receive).GetTypeName() << " content: " << std::get<0>(receive).DebugString() << std::endl;
            reader_writer->Write(std::get<0>(receive));
            std::cout << "DEBUG: one time read&write done\n\n\n" << std::endl;
            if (std::get<2>(receive) == false) {
                break;
            }
        }
        reader_writer->WritesDone();

        // Check connection status
        Status status = reader_writer->Finish();
	std::cout << status.error_message() << std::endl;

        if (sleep_duration == 0) {
            std::cout << "Disconnect and shut down." << std::endl;
            break;
        }

        // Sleep and reconnect afterwards
        std::cout << "Disconnect, then re-establish connection after" << sleep_duration << "second(s)" << std::endl;
        //Sleep(sleep_duration * 1000);
    }
}
