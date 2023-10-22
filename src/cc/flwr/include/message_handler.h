/*************************************************************************************************
 *
 * @file message_handler.h
 *
 * @brief Handle server messages by calling appropriate client methods
 *
 * @author Lekang Jiang
 *
 * @version 1.0
 *
 * @date 04/09/2021
 *
 *************************************************************************************************/

#pragma once
#include "client.h"
#include "serde.h"
using flwr::proto::ClientMessage;
using ClientMessage_Disconnect = flwr::proto::ClientMessage_DisconnectRes;
using flwr::proto::ClientMessage_EvaluateRes;
using flwr::proto::ClientMessage_FitRes;
using flwr::proto::Reason;
using flwr::proto::ServerMessage;
using flwr::proto::ServerMessage_EvaluateIns;
using flwr::proto::ServerMessage_FitIns;
using ServerMessage_Reconnect = flwr::proto::ServerMessage_ReconnectIns;

std::tuple<ClientMessage, int>
_reconnect(ServerMessage_Reconnect reconnect_msg);

ClientMessage _get_parameters(flwr_local::Client *client);

ClientMessage _fit(flwr_local::Client *client, ServerMessage_FitIns fit_msg);

ClientMessage _evaluate(flwr_local::Client *client,
                        ServerMessage_EvaluateIns evaluate_msg);

std::tuple<ClientMessage, int, bool> handle(flwr_local::Client *client,
                                            ServerMessage server_msg);
