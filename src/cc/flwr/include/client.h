#pragma once
#include "typing.h"

/*
* Abstract Client
*/
namespace flwr{
class Client {
public:
	virtual ParametersRes get_parameters()=0;
	virtual FitRes fit(FitIns ins)=0;
	virtual EvaluateRes evaluate(EvaluateIns ins)=0;
};
}
