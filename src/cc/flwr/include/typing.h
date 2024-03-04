/***********************************************************************************************************
 *
 * @file typing.h
 *
 * @brief C++ Flower type definitions
 *
 * @author Lekang Jiang
 *
 * @version 1.0
 *
 * @date 03/09/2021
 *
 * ********************************************************************************************************/

#pragma once
#include <list>
#include <map>
#include <optional>
#include <string>

namespace flwr_local {
/**
 * This class contains C++ types corresponding to ProtoBuf types that
 * ProtoBuf considers to be "Scalar Value Types", even though some of them
 * arguably do not conform to other definitions of what a scalar is. There is no
 * "bytes" type in C++, so "string" is used instead of bytes in Python (char*
 * can also be used if needed) In C++, Class is easier to use than Union (can be
 * changed if needed) Source:
 * https://developers.google.com/protocol-buffers/docs/overview#scalar
 *
 */
class Scalar {
public:
  // Getters
  std::optional<bool> getBool() { return b; }
  std::optional<std::string> getBytes() { return bytes; }
  std::optional<double> getDouble() { return d; }
  std::optional<int> getInt() { return i; }
  std::optional<std::string> getString() { return string; }

  // Setters
  void setBool(bool b) { this->b = b; }
  void setBytes(const std::string &bytes) { this->bytes = bytes; }
  void setDouble(double d) { this->d = d; }
  void setInt(int i) { this->i = i; }
  void setString(const std::string &string) { this->string = string; }

private:
  std::optional<bool> b = std::nullopt;
  std::optional<std::string> bytes = std::nullopt;
  std::optional<double> d = std::nullopt;
  std::optional<int> i = std::nullopt;
  std::optional<std::string> string = std::nullopt;
};

typedef std::map<std::string, flwr_local::Scalar> Metrics;

/**
 * Model parameters
 */
class Parameters {
public:
  Parameters() {}
  Parameters(const std::list<std::string> &tensors,
             const std::string &tensor_type)
      : tensors(tensors), tensor_type(tensor_type) {}

  // Getters
  std::list<std::string> getTensors() { return tensors; }
  std::string getTensor_type() { return tensor_type; }

  // Setters
  void setTensors(const std::list<std::string> &tensors) {
    this->tensors = tensors;
  }
  void setTensor_type(const std::string &tensor_type) {
    this->tensor_type = tensor_type;
  }

private:
  std::list<std::string> tensors;
  std::string tensor_type;
};

/**
 * Response when asked to return parameters
 */
class ParametersRes {
public:
  explicit ParametersRes(const Parameters &parameters)
      : parameters(parameters) {}

  Parameters getParameters() { return parameters; }
  void setParameters(const Parameters &p) { parameters = p; }

private:
  Parameters parameters;
};

/**
 * Fit instructions for a client
 */
class FitIns {
public:
  FitIns(const Parameters &parameters,
         const std::map<std::string, flwr_local::Scalar> &config)
      : parameters(parameters), config(config) {}

  // Getters
  Parameters getParameters() { return parameters; }
  std::map<std::string, Scalar> getConfig() { return config; }

  // Setters
  void setParameters(const Parameters &p) { parameters = p; }
  void setConfig(const std::map<std::string, Scalar> &config) {
    this->config = config;
  }

private:
  Parameters parameters;
  std::map<std::string, Scalar> config;
};

/**
 * Fit response from a client
 */
class FitRes {
public:
  FitRes() {}
  FitRes(const Parameters &parameters, int num_examples, int num_examples_ceil,
         float fit_duration, const Metrics &metrics)
      : parameters(parameters), num_examples(num_examples),
        fit_duration(fit_duration), metrics(metrics) {}

  // Getters
  Parameters getParameters() { return parameters; }
  int getNum_example() { return num_examples; }
  /*std::optional<int> getNum_examples_ceil()
  {
          return num_examples_ceil;
  }*/
  std::optional<float> getFit_duration() { return fit_duration; }
  std::optional<Metrics> getMetrics() { return metrics; }

  // Setters
  void setParameters(const Parameters &p) { parameters = p; }
  void setNum_example(int n) { num_examples = n; }
  /*void setNum_examples_ceil(int n)
  {
          num_examples_ceil = n;
  }*/
  void setFit_duration(float f) { fit_duration = f; }
  void setMetrics(const flwr_local::Metrics &m) { metrics = m; }

private:
  Parameters parameters;
  int num_examples;
  // std::optional<int> num_examples_ceil = std::nullopt;
  std::optional<float> fit_duration = std::nullopt;
  std::optional<Metrics> metrics = std::nullopt;
};

/**
 * Evaluate instructions for a client
 */
class EvaluateIns {
public:
  EvaluateIns(const Parameters &parameters,
              const std::map<std::string, Scalar> &config)
      : parameters(parameters), config(config) {}

  // Getters
  Parameters getParameters() { return parameters; }
  std::map<std::string, Scalar> getConfig() { return config; }

  // Setters
  void setParameters(const Parameters &p) { parameters = p; }
  void setConfig(const std::map<std::string, Scalar> &config) {
    this->config = config;
  }

private:
  Parameters parameters;
  std::map<std::string, Scalar> config;
};

/**
 * Evaluate response from a client
 */
class EvaluateRes {
public:
  EvaluateRes() {}
  EvaluateRes(float loss, int num_examples, float accuracy,
              const Metrics &metrics)
      : loss(loss), num_examples(num_examples), metrics(metrics) {}

  // Getters
  float getLoss() { return loss; }
  int getNum_example() { return num_examples; }
  std::optional<Metrics> getMetrics() { return metrics; }

  // Setters
  void setLoss(float f) { loss = f; }
  void setNum_example(int n) { num_examples = n; }
  void setMetrics(const Metrics &m) { metrics = m; }

private:
  float loss;
  int num_examples;
  std::optional<Metrics> metrics = std::nullopt;
};

typedef std::map<std::string, flwr_local::Scalar> Config;
typedef std::map<std::string, flwr_local::Scalar> Properties;

class PropertiesIns {
public:
  PropertiesIns() {}

  std::map<std::string, flwr_local::Scalar> getPropertiesIns() {
    return static_cast<std::map<std::string, flwr_local::Scalar>>(config);
  }

  void setPropertiesIns(const Config &c) { config = c; }

private:
  Config config;
};

class PropertiesRes {
public:
  PropertiesRes() {}

  Properties getPropertiesRes() { return properties; }

  void setPropertiesRes(const Properties &p) { properties = p; }

private:
  Properties properties;
};

struct Array {
  std::string dtype;
  std::vector<int32_t> shape;
  std::string stype;
  std::vector<uint8_t> data; // Using vector<uint8_t> to represent bytes
};

using ParametersRecord = std::map<std::string, Array>;
using MetricsRecord =
    std::map<std::string,
             std::variant<int, float, std::vector<int>, std::vector<float>>>;

using ConfigsRecord = std::map<
    std::string,
    std::variant<int, float, std::string, std::vector<uint8_t>, bool,
                 std::vector<int>, std::vector<float>, std::vector<std::string>,
                 std::vector<std::vector<uint8_t>>, std::vector<bool>>>;

class Metadata {
public:
  Metadata(int runId, const std::string &messageId, int srcNodeId,
           int dstNodeId, const std::string &replyToMessage,
           const std::string &groupId, const std::string &ttl,
           const std::string &messageType,
           std::optional<int> partitionId = std::nullopt)
      : _runId(runId), _messageId(messageId), _srcNodeId(srcNodeId),
        _dstNodeId(dstNodeId), _replyToMessage(replyToMessage),
        _groupId(groupId), _ttl(ttl), _messageType(messageType),
        _partitionId(partitionId) {}

  int getRunId() const { return _runId; }
  const std::string &getMessageId() const { return _messageId; }
  int getSrcNodeId() const { return _srcNodeId; }
  int getDstNodeId() const { return _dstNodeId; }
  const std::string &getReplyToMessage() const { return _replyToMessage; }
  const std::string &getGroupId() const { return _groupId; }
  const std::string &getTtl() const { return _ttl; }
  const std::string &getMessageType() const { return _messageType; }
  std::optional<int> getPartitionId() const { return _partitionId; }

  // Setters if needed
  void setSrcNodeId(int srcNodeId) { _srcNodeId = srcNodeId; }
  void setDstNodeId(int dstNodeId) { _dstNodeId = dstNodeId; }
  void setGroupId(const std::string &groupId) { _groupId = groupId; }
  void setRunId(const std::string &runId) { _runId = runId; }
  void setTtl(const std::string &ttl) { _ttl = ttl; }
  void setMessageType(const std::string &messageType) {
    _messageType = messageType;
  }
  void setPartitionId(std::optional<int> partitionId) {
    _partitionId = partitionId;
  }

private:
  int _runId;
  std::string _messageId;
  int _srcNodeId;
  int _dstNodeId;
  std::string _replyToMessage;
  std::string _groupId;
  std::string _ttl;
  std::string _messageType;
  std::optional<int> _partitionId;
};

class RecordSet {
public:
  RecordSet(
      const std::map<std::string, ParametersRecord> &parametersRecords = {},
      const std::map<std::string, MetricsRecord> &metricsRecords = {},
      const std::map<std::string, ConfigsRecord> &configsRecords = {})
      : _parametersRecords(parametersRecords), _metricsRecords(metricsRecords),
        _configsRecords(configsRecords) {}

  const std::map<std::string, ParametersRecord> &getParametersRecords() const {
    return _parametersRecords;
  }
  const std::map<std::string, MetricsRecord> &getMetricsRecords() const {
    return _metricsRecords;
  }
  const std::map<std::string, ConfigsRecord> &getConfigsRecords() const {
    return _configsRecords;
  }

  // Optionally, setters if needed

private:
  std::map<std::string, ParametersRecord> _parametersRecords;
  std::map<std::string, MetricsRecord> _metricsRecords;
  std::map<std::string, ConfigsRecord> _configsRecords;
};

class Message {
public:
  Message(const Metadata &metadata, const RecordSet &content)
      : _metadata(metadata), _content(content) {}

  const Metadata &getMetadata() const { return _metadata; }
  const RecordSet &getContent() const { return _content; }

  void setContent(const RecordSet &content) { _content = content; }

  // Implement createReply similar to the Python version
  Message createReply(const RecordSet &content, const std::string &ttl) const {
    Metadata replyMetadata(_metadata.getRunId(), "", _metadata.getDstNodeId(),
                           _metadata.getSrcNodeId(), _metadata.getMessageId(),
                           _metadata.getGroupId(), ttl,
                           _metadata.getMessageType(),
                           _metadata.getPartitionId());
    return Message(replyMetadata, content);
  }

private:
  Metadata _metadata;
  RecordSet _content;
};
} // namespace flwr_local
