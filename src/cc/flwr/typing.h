#pragma once
/*
* Typing
*
* There is no "bytes" type in C++, so "string" is used instead
*
*/

/*
* In C++ Class is easier to use than Union, and I think it can perform the same function as Union in Python
* The Scalar defined in "transport" uses "double" and "int64", I am not sure if there should be changed
*
*/
class LocalScalar {
public:
	// getters
	std::optional<bool> getBool() {
		return b;
	}
	std::optional<std::string> getBytes() {
		return bytes;
	}
	std::optional<float> getFloat() {
		return f;
	}
	std::optional<int> getInt() {
		return i;
	}
	std::optional<std::string> getString() {
		return string;
	}

	// setters
	void setBool(bool b) {
		this->b = b;
	}
	void setBytes(std::string bytes) {
		this->bytes = bytes;
	}
	void setFloat(float f) {
		this->f = f;
	}
	void setInt(int i) {
		this->i = i;
	}
	void setString(std::string string) {
		this->string = string;
	}

private:
	std::optional<bool> b = std::nullopt;
	std::optional<std::string> bytes = std::nullopt;
	std::optional<float> f = std::nullopt;
	std::optional<int> i = std::nullopt;
	std::optional<std::string> string = std::nullopt;
};

typedef std::map<std::string, LocalScalar> Metrics;

class Parameters {
public:
	Parameters() {};
	Parameters(std::list<std::string> tensors, std::string tensor_type)
		:tensors(tensors), tensor_type(tensor_type) {};

	// getters
	std::list<std::string> getTensors() {
		return tensors;
	}
	std::string getTensor_type() {
		return tensor_type;
	}

	// setters
	void setTensors(std::list<std::string> tensors) {
		this->tensors = tensors;
	}
	void setTensor_type(std::string tensor_type) {
		this->tensor_type = tensor_type;
	}
private:
	std::list<std::string> tensors;
	std::string tensor_type;
};

class ParametersRes {
public:
	ParametersRes(Parameters parameters)
		:parameters(parameters) {};

	Parameters getParameters() {
		return parameters;
	}
	void setParameters(Parameters p) {
		parameters = p;
	}
private:
	// Response when asked to return parameters
	Parameters parameters;
};

class FitIns {
public:
	FitIns(Parameters parameters, std::map<std::string, LocalScalar> config)
		: parameters(parameters), config(config) {};

	Parameters getParameters() {
		return parameters;
	}
	std::map<std::string, LocalScalar> getConfig() {
		return config;
	}

	void setParameters(Parameters p) {
		parameters = p;
	}
	void setConfig(std::map<std::string, LocalScalar> config) {
		this->config = config;
	}
private:
	// Fit instructions for a client
	Parameters parameters;
	std::map<std::string, LocalScalar> config;
};

class FitRes {
public:
	FitRes() {};
	FitRes(Parameters parameters, int num_examples, int num_examples_ceil, float fit_duration, Metrics metrics)
		: parameters(parameters), num_examples(num_examples), num_examples_ceil(num_examples_ceil),
		fit_duration(fit_duration), metrics(metrics) {};

	Parameters getParameters() {
		return parameters;
	}
	int getNum_example() {
		return num_examples;
	}
	std::optional<int> getNum_examples_ceil() {
		return num_examples_ceil;
	}
	std::optional<float> getFit_duration() {
		return fit_duration;
	}
	std::optional<Metrics> getMetrics() {
		return metrics;
	}

	void setParameters(Parameters p) {
		parameters = p;
	}
	void setNum_example(int n) {
		num_examples = n;
	}
	void setNum_examples_ceil(int n) {
		num_examples_ceil = n;
	}
	void setFit_duration(float f) {
		fit_duration = f;
	}
	void setMetrics(Metrics m) {
		metrics = m;
	}

private:
	// Fit response from a client
	Parameters parameters;
	int num_examples;
	std::optional<int> num_examples_ceil = std::nullopt;	// Deprecated
	std::optional<float> fit_duration = std::nullopt;		// Deprecated
	std::optional<Metrics> metrics = std::nullopt;
};

class EvaluateIns {
public:
	EvaluateIns(Parameters parameters, std::map<std::string, LocalScalar> config)
		: parameters(parameters), config(config) {};

	Parameters getParameters() {
		return parameters;
	}
	std::map<std::string, LocalScalar> getConfig() {
		return config;
	}

	void setParameters(Parameters p) {
		parameters = p;
	}
	void setConfig(std::map<std::string, LocalScalar> config) {
		this->config = config;
	}

private:
	// Evaluate instructions for a client
	Parameters parameters;
	std::map<std::string, LocalScalar> config;
};

class EvaluateRes {
public:
	EvaluateRes() {};
	EvaluateRes(float loss, int num_examples, float accuracy, Metrics metrics)
		: loss(loss), num_examples(num_examples), accuracy(accuracy), metrics(metrics) {};

	float getLoss() {
		return loss;
	}
	int getNum_example() {
		return num_examples;
	}
	std::optional<float> getAccuracy() {
		return accuracy;
	}
	std::optional<Metrics> getMetrics() {
		return metrics;
	}

	void setLoss(float f) {
		loss = f;
	}
	void setNum_example(int n) {
		num_examples = n;
	}
	void setAccuracy(float f) {
		accuracy = f;
	}
	void setMetrics(Metrics m) {
		metrics = m;
	}
private:
	// Evaluate response from a client
	float loss;
	int num_examples;
	std::optional<float> accuracy = std::nullopt;		// Deprecated
	std::optional<Metrics> metrics = std::nullopt;
};