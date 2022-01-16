#include "message_handler.h"

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

ClientMessage _get_parameters(flwr::Client* client) {
	// No need to deserialize get_parameters_msg(it's empty)
	//std::cout << "DEBUG: client getparameter" << std::endl;

	ClientMessage cm;
	//std::cout << "DEBUG: parameter res_to_proto" << std::endl;
	//std::cout << parameters_res_proto.parameters().DebugString() << std::endl;

	*(cm.mutable_parameters_res()) = parameters_res_to_proto(client->get_parameters());

	//std::cout << cm.parameters_res().DebugString() << std::endl;

	return cm;
}

ClientMessage _fit(flwr::Client* client, ServerMessage_FitIns fit_msg) {
	// Deserialize fit instruction
	flwr::FitIns fit_ins = fit_ins_from_proto(fit_msg);
	// Perform fit
	flwr::FitRes fit_res = client->fit(fit_ins);
	// Serialize fit result	
	ClientMessage cm;
	*cm.mutable_fit_res() = fit_res_to_proto(fit_res);
	return cm;
}

ClientMessage _evaluate(flwr::Client* client, ServerMessage_EvaluateIns evaluate_msg) {
	// Deserialize evaluate instruction
	flwr::EvaluateIns evaluate_ins = evaluate_ins_from_proto(evaluate_msg);
	//std::cout << "DEBUG: evaluate_ins_from_proto" << std::endl;
	// Perform evaluation
	flwr::EvaluateRes evaluate_res = client->evaluate(evaluate_ins);
	//std::cout << "DEBUG: client_evaluate" << std::endl;
	// Serialize evaluate result
	ClientMessage cm;
	*cm.mutable_evaluate_res() = evaluate_res_to_proto(evaluate_res);
	return cm;
}

std::tuple<ClientMessage, int, bool> handle(flwr::Client* client, ServerMessage server_msg) {
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
