#pragma once
#include "typing.h"

/*
* Abstract Client
*/
namespace flwr{
class Client {
public:
	virtual ParametersRes get_parameters() {};
	virtual FitRes fit(FitIns ins) {};
	virtual EvaluateRes evaluate(EvaluateIns ins) {};
};
}