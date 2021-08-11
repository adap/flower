/*
 *
 * Copyright 2015 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#include <iostream>
#include <memory>
#include <string>

#include <grpcpp/grpcpp.h>
#include <queue>
#include <optional>
#include <windows.h>

#ifdef BAZEL_BUILD
#include "examples/protos/helloworld.grpc.pb.h"
#else
#endif
#include "transport.grpc.pb.h"
using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;
using grpc::ClientReaderWriter;
using grpc::CompletionQueue;
using flower::transport::ClientMessage;
using flower::transport::ServerMessage;
using flower::transport::FlowerService;
using MessageParameters = flower::transport::Parameters;
using flower::transport::ServerMessage_EvaluateIns;
using flower::transport::ClientMessage_EvaluateRes;
using flower::transport::ServerMessage_FitIns;
using flower::transport::ClientMessage_FitRes;
using flower::transport::ClientMessage_ParametersRes;
using flower::transport::ServerMessage_Reconnect;
using flower::transport::Reason;
using flower::transport::ClientMessage_Disconnect;
using ProtoScalar = flower::transport::Scalar;
int GRPC_MAX_MESSAGE_LENGTH = 536870912;  //  == 512 * 1024 * 1024


/*
* typing
*/ 

class Scalar {
	//~Scalar();
	//Scalar(Scalar& s) { };
	//Scalar operator=(const Scalar& from) { return *this; }; // need change pb.h 1203
public:
	std::optional<bool> b = std::nullopt;
	std::optional<std::string> bytes = std::nullopt;
	std::optional<double> d = std::nullopt;
	std::optional<int> i = std::nullopt;
	std::optional<std::string> string = std::nullopt;
};

typedef std::map<std::string, Scalar> Metrics;

class Parameters {
public:
	std::list<std::string> tensors;
	std::string tensor_type;
};

class ParametersRes {
public:
	// Response when asked to return parameters
	Parameters parameters;
};
	
class FitIns {
public:
	// Fit instructions for a client
	Parameters parameters;
	std::map<std::string, Scalar> config;
};

class FitRes {
public:
	// Fit response from a client
	Parameters parameters;
	int num_examples;
	std::optional<int> num_examples_ceil = std::nullopt;	// Deprecated
	std::optional<float> fit_duration = std::nullopt;		// Deprecated
	std::optional<Metrics> metrics = std::nullopt;
};
	
class EvaluateIns {
public:
	// Evaluate instructions for a client
	Parameters parameters;
	std::map<std::string, Scalar> config;
};

class EvaluateRes {
public:
	// Evaluate response from a client
	float loss;
	int num_examples;
	std::optional<float> accuracy = std::nullopt;		// Deprecated
	std::optional<Metrics> metrics = std::nullopt;
};

/*
* Serde
*/
MessageParameters parameters_to_proto(Parameters parameters) {
	MessageParameters mp;
	mp.set_tensor_type(parameters.tensor_type);
	//std::list<std::string>::iterator i;
	for (auto i : parameters.tensors) {
		mp.add_tensors(i);
	}
	return mp;
}

Parameters parameters_from_proto(MessageParameters msg) {
	Parameters p;
	p.tensor_type = msg.tensor_type();
	std::list<std::string> tensors;
	for (int i = 0; i < msg.tensors_size(); i++) {
		tensors.push_back(msg.tensors(i));
	}
	return p;
}

google::protobuf::Map<std::string, ProtoScalar> metrics_to_proto(Metrics metrics) {
	// Serialize... .
	google::protobuf::Map<std::string, ProtoScalar> proto;
	for (auto key : metrics) {
		proto[key.first] = scalar_to_proto(key.second);
	}
	
	return proto;
}


Metrics metrics_from_proto(google::protobuf::Map<std::string, ProtoScalar> proto) {
	// Deserialize...
	Metrics metrics;
	
	for (auto i : proto) {
		metrics[i.first] = scalar_from_proto(i.second);
			
	}	
	return metrics;
}

ProtoScalar scalar_to_proto(Scalar scalar_msg) {
	// Deserialize... 
	ProtoScalar s;
	if (scalar_msg.b != std::nullopt) {
		s.set_bool_(scalar_msg.b.value());
		return s;
	}
	if (scalar_msg.bytes != std::nullopt) {
		s.set_bytes(scalar_msg.bytes.value());
	}
	if (scalar_msg.d != std::nullopt) {
		s.set_double_(scalar_msg.d.value());
		return s;
	}
	if (scalar_msg.i != std::nullopt) {
		s.set_sint64(scalar_msg.i.value());
		return s;
	}
	if (scalar_msg.string != std::nullopt) {
		s.set_string(scalar_msg.string.value());
		return s;
	}
	else {
		throw "Scalar to Proto failed";
	}

}

Scalar scalar_from_proto(ProtoScalar scalar_msg) {
	// Deserialize... 
	Scalar s;
	switch (scalar_msg.scalar_case()) {
	case 1:
		s.d = scalar_msg.double_();
		return s;	
	case 8:
		s.i = scalar_msg.sint64();
		return s;
	case 13:
		s.b = scalar_msg.bool_();
		return s;
	case 14:
		s.string = scalar_msg.string();
		return s;
	case 15:
		s.bytes = scalar_msg.bytes();
		return s;
	case 0:
		break;
	}
	
}


ClientMessage_ParametersRes parameters_res_to_proto(ParametersRes res) {
	MessageParameters mp = parameters_to_proto(res.parameters);
	ClientMessage_ParametersRes cpr;
	cpr.set_allocated_parameters(&mp);
	return cpr;
}

FitIns fit_ins_from_proto(ServerMessage_FitIns msg) {
	// Deserialize flower.FitIns from ProtoBuf message.
	Parameters parameters = parameters_from_proto(msg.parameters());
	Metrics config = metrics_from_proto(msg.config());
	FitIns fi;
	fi.parameters = parameters;
	fi.config = config;
	return fi;
}

ClientMessage_FitRes fit_res_to_proto(FitRes res) {
	ClientMessage_FitRes cres;
	// Serialize flower.FitIns to ProtoBuf message.
	MessageParameters parameters_proto = parameters_to_proto(res.parameters);
	google::protobuf::Map< ::std::string, ::flower::transport::Scalar >* metrics_msg;
	if (res.metrics == std::nullopt) {
		metrics_msg = NULL;
	}
	else {
		metrics_msg = &(metrics_to_proto(res.metrics.value()));
	}
	// Legacy case, will be removed in a future release
	if (res.num_examples_ceil != std::nullopt && res.fit_duration != std::nullopt) {
		cres.set_allocated_parameters(&parameters_proto);
		cres.set_num_examples(res.num_examples);
		cres.set_num_examples_ceil(res.num_examples_ceil.value());
		cres.set_fit_duration(res.fit_duration.value());
		google::protobuf::Map< ::std::string, ::flower::transport::Scalar >* temp = cres.mutable_metrics();
		temp = metrics_msg;
		return cres;
	}
	// Legacy case, will be removed in a future release
	if (res.num_examples_ceil != std::nullopt) {
		cres.set_allocated_parameters(&parameters_proto);
		cres.set_num_examples(res.num_examples);
		cres.set_num_examples_ceil(res.num_examples_ceil.value()); // Deprecated
		google::protobuf::Map< ::std::string, ::flower::transport::Scalar >* temp = cres.mutable_metrics();
		temp = metrics_msg;
		return cres;
	}			
	// Legacy case, will be removed in a future release
	if (res.fit_duration != std::nullopt) {
		cres.set_allocated_parameters(&parameters_proto);
		cres.set_num_examples(res.num_examples);
		cres.set_fit_duration(res.fit_duration.value());
		google::protobuf::Map< ::std::string, ::flower::transport::Scalar >* temp = cres.mutable_metrics();
		temp = metrics_msg;
		return cres;
	}
	// Forward - compatible case
	cres.set_allocated_parameters(&parameters_proto);
	cres.set_num_examples(res.num_examples);
	google::protobuf::Map< ::std::string, ::flower::transport::Scalar >* temp = cres.mutable_metrics();
	temp = metrics_msg;
	return cres;
	
}

EvaluateIns evaluate_ins_from_proto(ServerMessage_EvaluateIns msg) {
	// Deserialize flower.EvaluateIns from ProtoBuf message.
	Parameters parameters = parameters_from_proto(msg.parameters());
	Metrics config = metrics_from_proto(msg.config());
	EvaluateIns ei;
	ei.parameters = parameters;
	ei.config = config;
	return ei;
}

ClientMessage_EvaluateRes evaluate_res_to_proto(EvaluateRes res) {
	ClientMessage_EvaluateRes cres;
	// Serialize flower.EvaluateIns to ProtoBuf message.
	google::protobuf::Map< ::std::string, ::flower::transport::Scalar >* metrics_msg;
	if (res.metrics == std::nullopt) {
		metrics_msg = NULL;
	}
	else {
		metrics_msg = &metrics_to_proto(res.metrics.value());
	}
	// Legacy case, will be removed in a future release
	if (res.accuracy != std::nullopt) {
		cres.set_loss(res.loss);
		cres.set_num_examples(res.num_examples);
		cres.set_accuracy(res.accuracy.value());
		google::protobuf::Map< ::std::string, ::flower::transport::Scalar >* temp = cres.mutable_metrics();
		temp = metrics_msg;
		return cres;
	}
	// Forward - compatible case
	cres.set_loss(res.loss);
	cres.set_num_examples(res.num_examples);
	google::protobuf::Map< ::std::string, ::flower::transport::Scalar >* temp = cres.mutable_metrics();
	temp = metrics_msg;
	return cres;
}


/*
* Message Handler
*/

std::tuple<ClientMessage, int, bool> handle(Client client, ServerMessage server_msg) {
	if (server_msg.has_reconnect()) {
		std::tuple<ClientMessage, int> rec = _reconnect(server_msg.reconnect());
		return std::make_tuple(std::get<0>(rec), std::get<1>(rec), false);
	}
	if (server_msg.has_get_parameters()) {
		return std::make_tuple(_get_parameters(client), 0, true);
	}
	if (server_msg.has_fit_ins()) {
		return std::make_tuple(_fit(client, server_msg.fit_ins()), 0, true);
	}
	if (server_msg.has_evaluate_ins()) {
		return std::make_tuple(_evaluate(client, server_msg.evaluate_ins()), 0, true);
	}
	throw "Unkown server message";
	
}

std::tuple<ClientMessage, int> _reconnect(ServerMessage_Reconnect reconnect_msg) {
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
	cm.set_allocated_disconnect(&disconnect);

	std::tuple<ClientMessage, int> t = std::make_tuple(cm, sleep_duration);
	return t;
}

ClientMessage _get_parameters(Client client) {
	// No need to deserialize get_parameters_msg(it's empty)
	ParametersRes parameters_res = client.get_parameters();
	ClientMessage cm;
	ClientMessage_ParametersRes parameters_res_proto = parameters_res_to_proto(parameters_res);
	cm.set_allocated_parameters_res(&parameters_res_proto);
	return cm;
}

ClientMessage _fit(Client client, ServerMessage_FitIns fit_msg) {
	// Deserialize fit instruction
	FitIns fit_ins = fit_ins_from_proto(fit_msg);
	// Perform fit
	FitRes fit_res = client.fit(fit_ins);
	// Serialize fit result
	ClientMessage_FitRes fit_res_proto = fit_res_to_proto(fit_res);
	ClientMessage cm;
	cm.set_allocated_fit_res(&fit_res_proto);
	return cm;
}

ClientMessage _evaluate(Client client, ServerMessage_EvaluateIns evaluate_msg) {
	// Deserialize evaluate instruction
	EvaluateIns evaluate_ins = evaluate_ins_from_proto(evaluate_msg);
	// Perform evaluation
	EvaluateRes evaluate_res = client.evaluate(evaluate_ins);
	// Serialize evaluate result
	ClientMessage_EvaluateRes evaluate_res_proto = evaluate_res_to_proto(evaluate_res);
	ClientMessage cm;
	cm.set_allocated_evaluate_res(&evaluate_res_proto);
	return cm;
}


/*
* Abstract Client
*/
class Client {
public:
	virtual ParametersRes get_parameters() {};
	virtual FitRes fit(FitIns ins) {};
	virtual EvaluateRes evaluate(EvaluateIns ins) {};
};

	
void showState(int s) {
	switch (s) {
	case 0:
		std::cout << "GRPC_CHANNEL_IDLE " << std::endl;
		break;
	case 1:
		std::cout << "GRPC_CHANNEL_CONNECTING " << std::endl;
		break;
	case 2:
		std::cout << "GRPC_CHANNEL_READY " << std::endl;
		break;
	case 3:
		std::cout << "GRPC_CHANNEL_TRANSIENT_FAILURE " << std::endl;
		break;
	case 4:
		std::cout << "GRPC_CHANNEL_SHUTDOWN " << std::endl;
		break;
	default:
		std::cout << "UNKNOWN " << std::endl;
	}
}


void start_client(std::string server_address, Client client, int grpc_max_message_length = GRPC_MAX_MESSAGE_LENGTH) {
	while (true) {
		int sleep_duration = 0;
		grpc::ChannelArguments args;
		args.SetMaxReceiveMessageSize(grpc_max_message_length);
		args.SetMaxSendMessageSize(grpc_max_message_length);
		std::shared_ptr<Channel> channel = grpc::CreateCustomChannel(server_address, grpc::InsecureChannelCredentials(), args);
		grpc_connectivity_state s = channel->GetState(true);
		showState(s);
		std::unique_ptr<FlowerService::Stub> stub_ = FlowerService::NewStub(channel);

		ClientContext context;
		std::shared_ptr<ClientReaderWriter<ClientMessage, ServerMessage>> reader_writer(stub_->Join(&context));
		
		ServerMessage sm;
		while (reader_writer->Read(&sm)) {
			std::cout << "Got message type: " << sm.GetTypeName() << " content: " << sm.DebugString() << std::endl;
			std::tuple<ClientMessage, int, bool> receive = handle(client, sm);
			sleep_duration = std::get<1>(receive);
			reader_writer->Write(std::get<0>(receive));
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
		Sleep(sleep_duration * 1000);
	}
}

int main(int argc, char** argv) {
	
	std::string target_str = "localhost:50051";
	Client client;
	start_client(target_str, client);
	std::cin.get(); //keep the window
	return 0;
}
