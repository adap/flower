#include "serde.h"

/**
 * Serialize client parameters to protobuf parameters message
 */
MessageParameters parameters_to_proto(flwr::Parameters parameters) {
  MessageParameters mp;
  mp.set_tensor_type(parameters.getTensor_type());

  for (auto& i : parameters.getTensors()) {
    mp.add_tensors(i);
  }
  return mp;
}

/**
 * Deserialize client protobuf parameters message to client parameters
 */
flwr::Parameters parameters_from_proto(MessageParameters msg) {
  std::list<std::string> tensors;
  for (int i = 0; i < msg.tensors_size(); i++) {
    tensors.push_back(msg.tensors(i));
  }

  return flwr::Parameters(tensors, msg.tensor_type());
}

/**
 * Serialize client scalar type to protobuf scalar type
 */
ProtoScalar scalar_to_proto(flwr::Scalar scalar_msg) {
  ProtoScalar s;
  if (scalar_msg.getBool() != std::nullopt) {
    s.set_bool_(scalar_msg.getBool().value());
    return s;
  }
  if (scalar_msg.getBytes() != std::nullopt) {
    s.set_bytes(scalar_msg.getBytes().value());
    return s;
  }
  if (scalar_msg.getDouble() != std::nullopt) {
    s.set_double_(scalar_msg.getDouble().value());
    return s;
  }
  if (scalar_msg.getInt() != std::nullopt) {
    s.set_sint64(scalar_msg.getInt().value());
    return s;
  }
  if (scalar_msg.getString() != std::nullopt) {
    s.set_string(scalar_msg.getString().value());
    return s;
  } else {
    throw "Scalar to Proto failed";
  }
}

/**
 * Deserialize protobuf scalar type to client scalar type
 */
flwr::Scalar scalar_from_proto(ProtoScalar scalar_msg) {
  flwr::Scalar scalar;
  switch (scalar_msg.scalar_case()) {
    case 1:
      scalar.setDouble(scalar_msg.double_());
      return scalar;
    case 8:
      scalar.setInt(scalar_msg.sint64());
      return scalar;
    case 13:
      scalar.setBool(scalar_msg.bool_());
      return scalar;
    case 14:
      scalar.setString(scalar_msg.string());
      return scalar;
    case 15:
      scalar.setBytes(scalar_msg.bytes());
      return scalar;
    case 0:
      break;
  }
  throw "Error scalar type";
}

/**
 * Serialize client metrics type to protobuf metrics type
 * "Any" is used in Python, this part might be changed if needed
 */
google::protobuf::Map<std::string, ProtoScalar> metrics_to_proto(
    flwr::Metrics metrics) {
  google::protobuf::Map<std::string, ProtoScalar> proto;

  for (auto& [key, value] : metrics) {
    proto[key] = scalar_to_proto(value);
  }

  return proto;
}

/**
 * Deserialize protobuf metrics type to client metrics type
 * "Any" is used in Python, this part might be changed if needed
 */
flwr::Metrics metrics_from_proto(
    google::protobuf::Map<std::string, ProtoScalar> proto) {
  flwr::Metrics metrics;

  for (auto& [key, value] : proto) {
    metrics[key] = scalar_from_proto(value);
  }
  return metrics;
}

/**
 * Serialize client ParametersRes type to protobuf ParametersRes type
 */
ClientMessage_ParametersRes parameters_res_to_proto(flwr::ParametersRes res) {
  MessageParameters mp = parameters_to_proto(res.getParameters());
  ClientMessage_ParametersRes cpr;
  *(cpr.mutable_parameters()) = mp;
  return cpr;
}

/**
 * Deserialize protobuf FitIns type to client FitIns type
 */
flwr::FitIns fit_ins_from_proto(ServerMessage_FitIns msg) {
  flwr::Parameters parameters = parameters_from_proto(msg.parameters());
  flwr::Metrics config = metrics_from_proto(msg.config());
  return flwr::FitIns(parameters, config);
}

/**
 * Serialize client FitRes type to protobuf FitRes type
 */
ClientMessage_FitRes fit_res_to_proto(flwr::FitRes res) {
  ClientMessage_FitRes cres;

  MessageParameters parameters_proto = parameters_to_proto(res.getParameters());
  google::protobuf::Map< ::std::string, ::flower::transport::Scalar>*
      metrics_msg;
  if (res.getMetrics() == std::nullopt) {
    metrics_msg = NULL;
  } else {
    google::protobuf::Map< ::std::string, ::flower::transport::Scalar> proto =
        metrics_to_proto(res.getMetrics().value());
    metrics_msg = &proto;
  }

  // Forward - compatible case
  *(cres.mutable_parameters()) = parameters_proto;
  cres.set_num_examples(res.getNum_example());
  if (metrics_msg != NULL) {
    *cres.mutable_metrics() = *metrics_msg;
  }
  return cres;
}

/**
 * Deserialize protobuf EvaluateIns type to client EvaluateIns type
 */
flwr::EvaluateIns evaluate_ins_from_proto(ServerMessage_EvaluateIns msg) {
  flwr::Parameters parameters = parameters_from_proto(msg.parameters());
  flwr::Metrics config = metrics_from_proto(msg.config());
  return flwr::EvaluateIns(parameters, config);
}

/**
 * Serialize client EvaluateRes type to protobuf EvaluateRes type
 */
ClientMessage_EvaluateRes evaluate_res_to_proto(flwr::EvaluateRes res) {
  ClientMessage_EvaluateRes cres;
  google::protobuf::Map< ::std::string, ::flower::transport::Scalar>*
      metrics_msg;
  google::protobuf::Map< ::std::string, ::flower::transport::Scalar> proto;
  if (res.getMetrics() == std::nullopt) {
    metrics_msg = NULL;
  } else {
    proto = metrics_to_proto(res.getMetrics().value());
    metrics_msg = &proto;
  }

  // Forward - compatible case
  cres.set_loss(res.getLoss());
  cres.set_num_examples(res.getNum_example());
  if (metrics_msg != NULL) {
    auto& map = *cres.mutable_metrics();

    for (auto& [key, value] : *metrics_msg) {
      map[key] = value;
    }
  }
  
  return cres;
}
