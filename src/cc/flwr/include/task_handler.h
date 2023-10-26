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

bool validate_task_ins(const flwr::proto::TaskIns &task_ins,
                       const bool discard_reconnect_ins);
bool validate_task_res(const flwr::proto::TaskRes &task_res);
flwr::proto::TaskRes configure_task_res(const flwr::proto::TaskRes &task_res,
                                        const flwr::proto::TaskIns &task_ins,
                                        const flwr::proto::Node &node);
