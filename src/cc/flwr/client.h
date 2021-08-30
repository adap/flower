#pragma once
#include "typing.h"
using flwr::EvaluateIns;
using flwr::EvaluateRes;
using flwr::FitIns;
using flwr::FitRes;
using flwr::Parameters;
using flwr::ParametersRes;
using flwr::scalar;
using flwr::Metrics;

/*
* Abstract Client
*/
class Client {
public:
	virtual ParametersRes get_parameters() {};
	virtual FitRes fit(FitIns ins) {};
	virtual EvaluateRes evaluate(EvaluateIns ins) {};
};
