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

std::tuple<flwr::proto::ClientMessage, int, bool>
handle(flwr_local::Client *client, flwr::proto::ServerMessage server_msg);

std::tuple<flwr::proto::TaskRes, int, bool>
handle_task(flwr_local::Client *client, const flwr::proto::TaskIns &task_ins);
