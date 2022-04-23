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
#include "transport.grpc.pb.h"
#include "typing.h"
using flower::transport::ClientMessage;
using flower::transport::ServerMessage;
using MessageParameters = flower::transport::Parameters;
using flower::transport::Reason;
using ProtoScalar = flower::transport::Scalar;
using flower::transport::ClientMessage_EvaluateRes;
using flower::transport::ClientMessage_FitRes;
using flower::transport::ClientMessage_ParametersRes;
using flower::transport::ServerMessage_EvaluateIns;
using flower::transport::ServerMessage_FitIns;

/**
 * Serialize client parameters to protobuf parameters message
 */
MessageParameters parameters_to_proto(flwr::Parameters parameters);

/**
 * Deserialize client protobuf parameters message to client parameters
 */
flwr::Parameters parameters_from_proto(MessageParameters msg);

/**
 * Serialize client scalar type to protobuf scalar type
 */
ProtoScalar scalar_to_proto(flwr::Scalar scalar_msg);

/**
 * Deserialize protobuf scalar type to client scalar type
 */
flwr::Scalar scalar_from_proto(ProtoScalar scalar_msg);

/**
 * Serialize client metrics type to protobuf metrics type
 * "Any" is used in Python, this part might be changed if needed
 */
google::protobuf::Map<std::string, ProtoScalar> metrics_to_proto(
    flwr::Metrics metrics);

/**
 * Deserialize protobuf metrics type to client metrics type
 * "Any" is used in Python, this part might be changed if needed
 */
flwr::Metrics metrics_from_proto(
    google::protobuf::Map<std::string, ProtoScalar> proto);

/**
 * Serialize client ParametersRes type to protobuf ParametersRes type
 */
ClientMessage_ParametersRes parameters_res_to_proto(flwr::ParametersRes res);

/**
 * Deserialize protobuf FitIns type to client FitIns type
 */
flwr::FitIns fit_ins_from_proto(ServerMessage_FitIns msg);

/**
 * Serialize client FitRes type to protobuf FitRes type
 */
ClientMessage_FitRes fit_res_to_proto(flwr::FitRes res);

/**
 * Deserialize protobuf EvaluateIns type to client EvaluateIns type
 */
flwr::EvaluateIns evaluate_ins_from_proto(ServerMessage_EvaluateIns msg);

/**
 * Serialize client EvaluateRes type to protobuf EvaluateRes type
 */
ClientMessage_EvaluateRes evaluate_res_to_proto(flwr::EvaluateRes res);
