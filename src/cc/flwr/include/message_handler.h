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
using flower::transport::ClientMessage;
using flower::transport::ClientMessage_Disconnect;
using flower::transport::ClientMessage_EvaluateRes;
using flower::transport::ClientMessage_FitRes;
using flower::transport::Reason;
using flower::transport::ServerMessage;
using flower::transport::ServerMessage_EvaluateIns;
using flower::transport::ServerMessage_FitIns;
using flower::transport::ServerMessage_Reconnect;

std::tuple<ClientMessage, int> _reconnect(
    ServerMessage_Reconnect reconnect_msg);

ClientMessage _get_parameters(flwr::Client* client);

ClientMessage _fit(flwr::Client* client, ServerMessage_FitIns fit_msg);

ClientMessage _evaluate(flwr::Client* client,
                        ServerMessage_EvaluateIns evaluate_msg);

std::tuple<ClientMessage, int, bool> handle(flwr::Client* client,
                                            ServerMessage server_msg);
