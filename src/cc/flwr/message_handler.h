#pragma once
#include "client.h"
#include "serde.h"
using flower::transport::ClientMessage;
using flower::transport::ServerMessage;
using flower::transport::Reason;
using flower::transport::ServerMessage_EvaluateIns;
using flower::transport::ClientMessage_EvaluateRes;
using flower::transport::ServerMessage_FitIns;
using flower::transport::ClientMessage_FitRes;
using flower::transport::ServerMessage_Reconnect;
using flower::transport::ClientMessage_Disconnect;

/*
* Message Handler
*/
std::tuple<ClientMessage, int> _reconnect(ServerMessage_Reconnect reconnect_msg) {
	// Determine the reason for sending Disconnect message
	Reason reason = Reason::ACK;
	int sleep_duration = 0;
	if (reconnect_msg.seconds() != 0) {
		reason = Reason::RECONNECT;
		sleep_duration = reconnect_msg.seconds();
	}

	// Build Disconnect message	
	ClientMessage_Disconnect disconnect;
	disconnect.set_reason(reason);
	ClientMessage cm;
	*cm.mutable_disconnect() = disconnect;

	return std::make_tuple(cm, sleep_duration);
}

ClientMessage _get_parameters(Client* client) {
	// No need to deserialize get_parameters_msg(it's empty)
	//std::cout << "DEBUG: client getparameter" << std::endl;
	//ParametersRes parameters_res = client->get_parameters();
	ClientMessage cm;
	//std::cout << "DEBUG: parameter res_to_proto" << std::endl;
	//std::cout << client->get_parameters().getParameters().getTensor_type() << std::endl;
	//ClientMessage_ParametersRes parameters_res_proto = parameters_res_to_proto(client->get_parameters());
	//std::cout << parameters_res_proto.parameters().DebugString() << std::endl;
	*(cm.mutable_parameters_res()) = parameters_res_to_proto(client->get_parameters());
	//cm.set_allocated_parameters_res(&parameters_res_proto);
	//std::cout << cm.parameters_res().DebugString() << std::endl;
	//std::cout << "DEBUG: getparameter done" << std::endl;
	return cm;
}

ClientMessage _fit(Client* client, ServerMessage_FitIns fit_msg) {
	// Deserialize fit instruction
	FitIns fit_ins = fit_ins_from_proto(fit_msg);
	// Perform fit
	FitRes fit_res = client->fit(fit_ins);
	// Serialize fit result	
	ClientMessage cm;
	*cm.mutable_fit_res() = fit_res_to_proto(fit_res);
	return cm;
}

ClientMessage _evaluate(Client* client, ServerMessage_EvaluateIns evaluate_msg) {
	// Deserialize evaluate instruction
	EvaluateIns evaluate_ins = evaluate_ins_from_proto(evaluate_msg);
	// Perform evaluation
	EvaluateRes evaluate_res = client->evaluate(evaluate_ins);
	// Serialize evaluate result
	ClientMessage cm;
	*cm.mutable_evaluate_res() = evaluate_res_to_proto(evaluate_res);
	return cm;
}

std::tuple<ClientMessage, int, bool> handle(Client* client, ServerMessage server_msg) {
	if (server_msg.has_reconnect()) {
		std::cout << "DEBUG: handle reconnect" << std::endl;
		std::tuple<ClientMessage, int> rec = _reconnect(server_msg.reconnect());
		return std::make_tuple(std::get<0>(rec), std::get<1>(rec), false);
	}
	if (server_msg.has_get_parameters()) {
		std::cout << "DEBUG: handle getparameter" << std::endl;
		return std::make_tuple(_get_parameters(client), 0, true);
	}
	if (server_msg.has_fit_ins()) {
		std::cout << "DEBUG: handle fitins" << std::endl;
		return std::make_tuple(_fit(client, server_msg.fit_ins()), 0, true);
	}
	if (server_msg.has_evaluate_ins()) {
		std::cout << "DEBUG: handle evaluateins" << std::endl;
		return std::make_tuple(_evaluate(client, server_msg.evaluate_ins()), 0, true);
	}
	throw "Unkown server message";

}