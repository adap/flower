/*************************************************************************************************
 *
 * @file task_handler.h
 *
 * @brief Handle incoming or outgoing tasks
 *
 * @author The Flower Authors
 *
 * @version 1.0
 *
 * @date 06/11/2023
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
