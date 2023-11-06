/***********************************************************************************************************
 *
 * @file serde.h
 *
 * @brief ProtoBuf serialization and deserialization
 *
 * @author Lekang Jiang
 *
 * @version 1.0
 *
 * @date 03/09/2021
 *
 * ********************************************************************************************************/

#pragma once
// cppcheck-suppress missingInclude
#include "flwr/proto/transport.grpc.pb.h"
// cppcheck-suppress missingInclude
#include "flwr/proto/transport.pb.h"
// cppcheck-suppress missingInclude
#include "flwr/proto/fleet.grpc.pb.h"
// cppcheck-suppress missingInclude
#include "flwr/proto/fleet.pb.h"
#include "typing.h"
using flwr::proto::ClientMessage;
using flwr::proto::ServerMessage;
using MessageParameters = flwr::proto::Parameters;
using flwr::proto::Reason;
using ProtoScalar = flwr::proto::Scalar;
using flwr::proto::ClientMessage_EvaluateRes;
using flwr::proto::ClientMessage_FitRes;
using ClientMessage_ParametersRes = flwr::proto::ClientMessage_GetParametersRes;
using flwr::proto::ServerMessage_EvaluateIns;
using flwr::proto::ServerMessage_FitIns;

/**
 * Serialize client parameters to protobuf parameters message
 */
MessageParameters parameters_to_proto(flwr_local::Parameters parameters);

/**
 * Deserialize client protobuf parameters message to client parameters
 */
flwr_local::Parameters parameters_from_proto(MessageParameters msg);

/**
 * Serialize client scalar type to protobuf scalar type
 */
ProtoScalar scalar_to_proto(flwr_local::Scalar scalar_msg);

/**
 * Deserialize protobuf scalar type to client scalar type
 */
flwr_local::Scalar scalar_from_proto(ProtoScalar scalar_msg);

/**
 * Serialize client metrics type to protobuf metrics type
 * "Any" is used in Python, this part might be changed if needed
 */
google::protobuf::Map<std::string, ProtoScalar>
metrics_to_proto(flwr_local::Metrics metrics);

/**
 * Deserialize protobuf metrics type to client metrics type
 * "Any" is used in Python, this part might be changed if needed
 */
flwr_local::Metrics
metrics_from_proto(google::protobuf::Map<std::string, ProtoScalar> proto);

/**
 * Serialize client ParametersRes type to protobuf ParametersRes type
 */
ClientMessage_ParametersRes
parameters_res_to_proto(flwr_local::ParametersRes res);

/**
 * Deserialize protobuf FitIns type to client FitIns type
 */
flwr_local::FitIns fit_ins_from_proto(ServerMessage_FitIns msg);

/**
 * Serialize client FitRes type to protobuf FitRes type
 */
ClientMessage_FitRes fit_res_to_proto(flwr_local::FitRes res);

/**
 * Deserialize protobuf EvaluateIns type to client EvaluateIns type
 */
flwr_local::EvaluateIns evaluate_ins_from_proto(ServerMessage_EvaluateIns msg);

/**
 * Serialize client EvaluateRes type to protobuf EvaluateRes type
 */
ClientMessage_EvaluateRes evaluate_res_to_proto(flwr_local::EvaluateRes res);
