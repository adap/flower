#include "serde.h"

/**
 * Serialize client parameters to protobuf parameters message
 */
flwr::proto::Parameters parameters_to_proto(flwr_local::Parameters parameters) {
  flwr::proto::Parameters mp;
  mp.set_tensor_type(parameters.getTensor_type());

  for (auto &i : parameters.getTensors()) {
    mp.add_tensors(i);
  }
  return mp;
}

/**
 * Deserialize client protobuf parameters message to client parameters
 */
flwr_local::Parameters parameters_from_proto(flwr::proto::Parameters msg) {
  std::list<std::string> tensors;
  for (int i = 0; i < msg.tensors_size(); i++) {
    tensors.push_back(msg.tensors(i));
  }

  return flwr_local::Parameters(tensors, msg.tensor_type());
}

/**
 * Serialize client scalar type to protobuf scalar type
 */
flwr::proto::Scalar scalar_to_proto(flwr_local::Scalar scalar_msg) {
  flwr::proto::Scalar s;
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
flwr_local::Scalar scalar_from_proto(flwr::proto::Scalar scalar_msg) {
  flwr_local::Scalar scalar;
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
google::protobuf::Map<std::string, flwr::proto::Scalar>
metrics_to_proto(flwr_local::Metrics metrics) {
  google::protobuf::Map<std::string, flwr::proto::Scalar> proto;

  for (auto &[key, value] : metrics) {
    proto[key] = scalar_to_proto(value);
  }

  return proto;
}

/**
 * Deserialize protobuf metrics type to client metrics type
 * "Any" is used in Python, this part might be changed if needed
 */
flwr_local::Metrics metrics_from_proto(
    google::protobuf::Map<std::string, flwr::proto::Scalar> proto) {
  flwr_local::Metrics metrics;

  for (auto &[key, value] : proto) {
    metrics[key] = scalar_from_proto(value);
  }
  return metrics;
}

/**
 * Serialize client ParametersRes type to protobuf ParametersRes type
 */
flwr::proto::ClientMessage_GetParametersRes
parameters_res_to_proto(flwr_local::ParametersRes res) {
  flwr::proto::Parameters mp = parameters_to_proto(res.getParameters());
  flwr::proto::ClientMessage_GetParametersRes cpr;
  *(cpr.mutable_parameters()) = mp;
  return cpr;
}

/**
 * Deserialize protobuf FitIns type to client FitIns type
 */
flwr_local::FitIns fit_ins_from_proto(flwr::proto::ServerMessage_FitIns msg) {
  flwr_local::Parameters parameters = parameters_from_proto(msg.parameters());
  flwr_local::Metrics config = metrics_from_proto(msg.config());
  return flwr_local::FitIns(parameters, config);
}

/**
 * Serialize client FitRes type to protobuf FitRes type
 */
flwr::proto::ClientMessage_FitRes fit_res_to_proto(flwr_local::FitRes res) {
  flwr::proto::ClientMessage_FitRes cres;

  flwr::proto::Parameters parameters_proto =
      parameters_to_proto(res.getParameters());
  google::protobuf::Map<::std::string, ::flwr::proto::Scalar> metrics_msg;
  if (res.getMetrics() != std::nullopt) {
    metrics_msg = metrics_to_proto(res.getMetrics().value());
  }

  // Forward - compatible case
  *(cres.mutable_parameters()) = parameters_proto;
  cres.set_num_examples(res.getNum_example());
  if (!metrics_msg.empty()) {
    *cres.mutable_metrics() = metrics_msg;
  }
  return cres;
}

/**
 * Deserialize protobuf EvaluateIns type to client EvaluateIns type
 */
flwr_local::EvaluateIns
evaluate_ins_from_proto(flwr::proto::ServerMessage_EvaluateIns msg) {
  flwr_local::Parameters parameters = parameters_from_proto(msg.parameters());
  flwr_local::Metrics config = metrics_from_proto(msg.config());
  return flwr_local::EvaluateIns(parameters, config);
}

/**
 * Serialize client EvaluateRes type to protobuf EvaluateRes type
 */
flwr::proto::ClientMessage_EvaluateRes
evaluate_res_to_proto(flwr_local::EvaluateRes res) {
  flwr::proto::ClientMessage_EvaluateRes cres;
  google::protobuf::Map<::std::string, ::flwr::proto::Scalar> metrics_msg;
  if (res.getMetrics() != std::nullopt) {
    metrics_msg = metrics_to_proto(res.getMetrics().value());
  }
  // Forward - compatible case
  cres.set_loss(res.getLoss());
  cres.set_num_examples(res.getNum_example());
  if (!metrics_msg.empty()) {
    auto &map = *cres.mutable_metrics();

    for (auto &[key, value] : metrics_msg) {
      map[key] = value;
    }
  }

  return cres;
}
