#pragma once
#include "typing.h"
#include "transport.grpc.pb.h"
using flower::transport::ClientMessage;
using flower::transport::ServerMessage;
using MessageParameters = flower::transport::Parameters;
using flower::transport::Reason;
using ProtoScalar = flower::transport::Scalar;
using flower::transport::ClientMessage_ParametersRes;
using flower::transport::ServerMessage_EvaluateIns;
using flower::transport::ClientMessage_EvaluateRes;
using flower::transport::ServerMessage_FitIns;
using flower::transport::ClientMessage_FitRes;

/*
* Serde
*/

MessageParameters parameters_to_proto(Parameters parameters) {
	MessageParameters mp;
	//std::cout << "DEBUG: set tensor" << std::endl;
	mp.set_tensor_type(parameters.getTensor_type());
	//std::cout << "DEBUG: iterator" << std::endl;

	for (auto& i : parameters.getTensors()) {
		//std::cout << "DEBUG: add tensor" << std::endl;
		mp.add_tensors(i);
	}
	return mp;
}

Parameters parameters_from_proto(MessageParameters msg) {
	//Parameters(msg.tensor_type(), tensors);
	//p.setTensor_type(msg.tensor_type());
	std::list<std::string> tensors;
	for (int i = 0; i < msg.tensors_size(); i++) {
		tensors.push_back(msg.tensors(i));
	}
	//p.setTensors(tensors);
	return Parameters(tensors, msg.tensor_type());
}

ProtoScalar scalar_to_proto(Scalar scalar_msg) {
	// Deserialize... 
	ProtoScalar s;
	if (scalar_msg.getBool() != std::nullopt) {
		s.set_bool_(scalar_msg.getBool().value());
		return s;
	}
	if (scalar_msg.getBytes() != std::nullopt) {
		s.set_bytes(scalar_msg.getBytes().value());
	}
	if (scalar_msg.getFloat() != std::nullopt) {
		s.set_double_(scalar_msg.getFloat().value());
		return s;
	}
	if (scalar_msg.getInt() != std::nullopt) {
		s.set_sint64(scalar_msg.getInt().value());
		return s;
	}
	if (scalar_msg.getString() != std::nullopt) {
		s.set_string(scalar_msg.getString().value());
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
		s.setFloat(scalar_msg.double_());
		return s;
	case 8:
		s.setInt(scalar_msg.sint64());
		return s;
	case 13:
		s.setBool(scalar_msg.bool_());
		return s;
	case 14:
		s.setString(scalar_msg.string());
		return s;
	case 15:
		s.setBytes(scalar_msg.bytes());
		return s;
	case 0:
		break;
	}
	throw "Error scalar type";
}

// "Any" is used in Python, this part might be changed
google::protobuf::Map<std::string, ProtoScalar> metrics_to_proto(Metrics metrics) {
	// Serialize... .
	google::protobuf::Map<std::string, ProtoScalar> proto;

	for (auto& [key, value] : metrics) {
		proto[key] = scalar_to_proto(value);
	}

	return proto;
}

// "Any" is used in Python, this part might be changed
Metrics metrics_from_proto(google::protobuf::Map<std::string, ProtoScalar> proto) {
	// Deserialize...
	Metrics metrics;

	for (auto& [key, value] : proto) {
		metrics[key] = scalar_from_proto(value);

	}
	return metrics;
}


ClientMessage_ParametersRes parameters_res_to_proto(ParametersRes res) {
	//std::cout << "DEBUG: parameters to proto" << std::endl;
	//std::cout << res.getParameters().getTensor_type() << std::endl;
	MessageParameters mp = parameters_to_proto(res.getParameters());
	ClientMessage_ParametersRes cpr;
	*(cpr.mutable_parameters()) = mp;
	//std::cout << "DEBUG: parameter res to proto done" << cpr.DebugString() << std::endl;
	return cpr;
}

FitIns fit_ins_from_proto(ServerMessage_FitIns msg) {
	// Deserialize flower.FitIns from ProtoBuf message.
	Parameters parameters = parameters_from_proto(msg.parameters());
	Metrics config = metrics_from_proto(msg.config());
	return FitIns(parameters, config);
}

ClientMessage_FitRes fit_res_to_proto(FitRes res) {
	ClientMessage_FitRes cres;
	// Serialize flower.FitIns to ProtoBuf message.
	MessageParameters parameters_proto = parameters_to_proto(res.getParameters());
	google::protobuf::Map< ::std::string, ::flower::transport::Scalar >* metrics_msg;
	if (res.getMetrics() == std::nullopt) {
		metrics_msg = NULL;
	}
	else {
		google::protobuf::Map< ::std::string, ::flower::transport::Scalar > proto = metrics_to_proto(res.getMetrics().value());
		metrics_msg = &proto;
	}
	// Legacy case, will be removed in a future release
	if (res.getNum_examples_ceil() != std::nullopt && res.getFit_duration() != std::nullopt) {
		*(cres.mutable_parameters()) = parameters_proto;
		cres.set_num_examples(res.getNum_example());
		cres.set_num_examples_ceil(res.getNum_examples_ceil().value());
		cres.set_fit_duration(res.getFit_duration().value());
		if (metrics_msg != NULL) {
			*cres.mutable_metrics() = *metrics_msg;
		}
		return cres;
	}
	// Legacy case, will be removed in a future release
	if (res.getNum_examples_ceil() != std::nullopt) {
		*(cres.mutable_parameters()) = parameters_proto;
		cres.set_num_examples(res.getNum_example());
		cres.set_num_examples_ceil(res.getNum_examples_ceil().value()); // Deprecated
		if (metrics_msg != NULL) {
			*cres.mutable_metrics() = *metrics_msg;
		}
		return cres;
	}
	// Legacy case, will be removed in a future release
	if (res.getFit_duration() != std::nullopt) {
		*(cres.mutable_parameters()) = parameters_proto;
		cres.set_num_examples(res.getNum_example());
		cres.set_fit_duration(res.getFit_duration().value());
		if (metrics_msg != NULL) {
			*cres.mutable_metrics() = *metrics_msg;
		}
		return cres;
	}
	// Forward - compatible case
	*(cres.mutable_parameters()) = parameters_proto;
	cres.set_num_examples(res.getNum_example());
	if (metrics_msg != NULL) {
		*cres.mutable_metrics() = *metrics_msg;
	}
	return cres;

}

EvaluateIns evaluate_ins_from_proto(ServerMessage_EvaluateIns msg) {
	// Deserialize flower.EvaluateIns from ProtoBuf message.
	Parameters parameters = parameters_from_proto(msg.parameters());
	Metrics config = metrics_from_proto(msg.config());
	return EvaluateIns(parameters, config);
}

ClientMessage_EvaluateRes evaluate_res_to_proto(EvaluateRes res) {
	ClientMessage_EvaluateRes cres;
	// Serialize flower.EvaluateIns to ProtoBuf message.
	google::protobuf::Map< ::std::string, ::flower::transport::Scalar >* metrics_msg;
	if (res.getMetrics() == std::nullopt) {
		metrics_msg = NULL;
	}
	else {
		google::protobuf::Map< ::std::string, ::flower::transport::Scalar > proto = metrics_to_proto(res.getMetrics().value());
		metrics_msg = &proto;
	}
	// Legacy case, will be removed in a future release
	if (res.getAccuracy() != std::nullopt) {
		cres.set_loss(res.getLoss());
		cres.set_num_examples(res.getNum_example());
		cres.set_accuracy(res.getAccuracy().value());
		if (metrics_msg != NULL) {
			*cres.mutable_metrics() = *metrics_msg;
		}
		return cres;
	}
	// Forward - compatible case
	cres.set_loss(res.getLoss());
	cres.set_num_examples(res.getNum_example());
	if (metrics_msg != NULL) {
		*cres.mutable_metrics() = *metrics_msg;
	}
	return cres;
}