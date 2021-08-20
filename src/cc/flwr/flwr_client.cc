/*
 *
 * This project is build on Visual Studio 2019 on Windows with C++ 17
 * There are some differences between Windows and Linux systems, so some code should be changed to work in Linux
 * Additional include directories, additional library directories and additional dependencies are set manually
 * Preprocessor defination are set to avoid some warnings
 * 
 * 
 * Author: Lekang Jiang 14/08/2021
 *
 */

#include <iostream>
#include <memory>
#include <string>
#include <grpcpp/grpcpp.h>
#include <queue>
#include <optional>
//#include <windows.h>
#include <map>
#include "transport.grpc.pb.h"
#include "typing.h"
#include "serde.h"
#include "message_handler.h"

using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;
using grpc::ClientReaderWriter;
using flower::transport::ClientMessage;
using flower::transport::ServerMessage;
using flower::transport::FlowerService;

int GRPC_MAX_MESSAGE_LENGTH = 536870912;  //  == 512 * 1024 * 1024


/*
* A self-defined class for testing message handling
*
*/
class Example_client : public Client {
public:
	virtual ParametersRes get_parameters() override {
		//std::cout << "DEBUG: get parameters" << std::endl;
		std::list<std::string> tensors;
		tensors.push_back("First example");
		tensors.push_back("Second example");
		tensors.push_back("Third example");
		Parameters p(tensors, "example tensor");

		return ParametersRes(p);
	}

	virtual FitRes fit(FitIns ins) override {
		std::list<std::string> tensors;
		for (auto& i : ins.getParameters().getTensors()) {
			tensors.push_back(i + " executed one fit");
		}
		Metrics m;
		for (auto& i : ins.getConfig()) {
			m[i.first + " one fit"] = i.second;
		}

		return FitRes(Parameters(tensors, ins.getParameters().getTensor_type()), 5, 1, 10, m);
	}

	virtual EvaluateRes evaluate(EvaluateIns ins) override {
		Metrics m;
		for (auto& i : ins.getConfig()) {
			m[i.first + " one evaluate"] = i.second;
		}

		return EvaluateRes(0.5, 5, 0.9, m);
	}
};




void start_client(std::string server_address, Client* client, int grpc_max_message_length = GRPC_MAX_MESSAGE_LENGTH) {
	
	while (true) {
		int sleep_duration = 0;
		grpc::ChannelArguments args;
		args.SetMaxReceiveMessageSize(grpc_max_message_length);
		args.SetMaxSendMessageSize(grpc_max_message_length);
		std::shared_ptr<Channel> channel = grpc::CreateCustomChannel(server_address, grpc::InsecureChannelCredentials(), args);
		
		//grpc_connectivity_state s = channel->GetState(true);

		std::unique_ptr<FlowerService::Stub> stub_ = FlowerService::NewStub(channel);

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

int main(int argc, char** argv) {
	
	std::string target_str = "localhost:50051";
	Example_client client;
	start_client(target_str, &client);
	//std::cin.get(); //keep the window
	return 0;
}
