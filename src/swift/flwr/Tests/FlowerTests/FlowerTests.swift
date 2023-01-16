import XCTest
@testable import flwr

@available(iOS 14.0, *)
final class FlowerTests: XCTestCase {
    
    let pConv = ParameterConverter.shared
    let input1: [Float] = [1, 2, 3, 4]
    let input2: [Float] = [1.1435, 2, 3, 4]
    let status = Status(code: .ok, message: "test")
    let reconnect = Reconnect(seconds: 4)
    let disconnect = Disconnect(reason: "RECONNECT")
    let metrics: Metrics = ["MAE": Scalar(float: 0.013), "MSE": Scalar(float: 0.03), "REC": Scalar(bool: true)]
    let properties: Properties = ["MSE": Scalar(str: "test"), "Property1": Scalar(str: "test"), "REC": Scalar(int: 4, str: "abc")]
    let scalar1 = Scalar(bool: true)
    let scalar2 = Scalar(bool: true, int: 2, str: "test")
    
    // ParameterConverter
    func testParameterConverter() throws {
        
        let output1 = pConv.dataToArray(
            data: pConv.arrayToData(
                array: input1,
                shape: [2, 2]
            )!
        )
        let output2 = pConv.dataToArray(
            data: pConv.arrayToData(
                array: input2,
                shape: [2, 2]
            )!
        )
        XCTAssertEqual(input1, output1)
        XCTAssertEqual(input2, output2)
    }
    
    // Serde
    func testParameterProtoConverter() throws {
        
        let data1 = pConv.arrayToData(array: input1, shape: [2, 2])!
        let data2 = pConv.arrayToData(array: input2, shape: [2, 2])!
        let parameters = Parameters(tensors: [data1, data2], tensorType: "testTensor")
        
        let proto = parametersToProto(parameters: parameters)
        let result = parametersFromProto(msg: proto)
        XCTAssertEqual(parameters, result)
    }
    
    func testStatusProtoConverter() throws {
        
        let proto = statusToProto(status: status)
        let result = statusFromProto(msg: proto)
        XCTAssertEqual(status, result)
    }
    
    func testReconnectProtoConverter() throws {
        
        let proto = reconnectToProto(reconnect: reconnect)
        let result = reconnectFromProto(msg: proto)
        XCTAssertEqual(reconnect, result)
    }
    
    func testDisconnectProtoConverter() throws {
        
        let proto = disconnectToProto(disconnect: disconnect)
        let result = disconnectFromProto(msg: proto)
        XCTAssertEqual(disconnect, result)
    }
    
    func testParametersResProtoConversion() throws {
        
        let data1 = pConv.arrayToData(array: input1, shape: [2, 2])!
        let data2 = pConv.arrayToData(array: input2, shape: [2, 2])!
        let parameters = Parameters(tensors: [data1, data2], tensorType: "testTensor")
        
        let parametersRes = GetParametersRes(parameters: parameters, status: status)
        let proto = parametersResToProto(res: parametersRes)
        let result = parametersResFromProto(msg: proto)
        XCTAssertEqual(parametersRes, result)
    }
    
    
    func testFitResProtoConversion() throws {
        
        let data1 = pConv.arrayToData(array: input1, shape: [2, 2])!
        let data2 = pConv.arrayToData(array: input2, shape: [2, 2])!
        let parameters = Parameters(tensors: [data1, data2], tensorType: "testTensor")
        
        let fitRes = FitRes(parameters: parameters, numExamples: 3, status: status)
        let proto = fitResToProto(res: fitRes)
        let result = fitResFromProto(msg: proto)
        XCTAssertEqual(fitRes, result)
    }
    
    func testFitInsProtoConversion() throws {
        
        let data1 = pConv.arrayToData(array: input1, shape: [2, 2])!
        let data2 = pConv.arrayToData(array: input2, shape: [2, 2])!
        
        let parameters = Parameters(tensors: [data1, data2], tensorType: "testTensor")
        let fitIns = FitIns(parameters: parameters, config: ["config1": scalar1, "config2": scalar2])
        
        let proto = fitInsToProto(ins: fitIns)
        let result = fitInsFromProto(msg: proto)
        XCTAssertEqual(fitIns, result)
    }
    
    func testPropertiesInsProtoConversion() throws {
        
        let propertiesIns = GetPropertiesIns(config: properties)
        let proto = propertiesInsToProto(ins: propertiesIns)
        let result = propertiesInsFromProto(msg: proto)
        XCTAssertEqual(propertiesIns, result)
    }
    
    func testPropertiesResProtoConversion() throws {
        let propertiesRes = GetPropertiesRes(properties: properties, status: status)
        let proto = propertiesResToProto(res: propertiesRes)
        let result = propertiesResFromProto(msg: proto)
        XCTAssertEqual(propertiesRes, result)
    }
    
    func testEvaluateInsProtoConversion() throws {
        
        let data1 = pConv.arrayToData(array: input1, shape: [2, 2])!
        let data2 = pConv.arrayToData(array: input2, shape: [2, 2])!
        let parameters = Parameters(tensors: [data1, data2], tensorType: "testTensor")
        
        let evaluateIns = EvaluateIns(parameters: parameters, config: ["config1": scalar1, "config2": scalar2])
        let proto = evaluateInsToProto(ins: evaluateIns)
        let result = evaluateInsFromProto(msg: proto)
        XCTAssertEqual(evaluateIns, result)
    }
    
    func testEvaluateResProtoConversion() throws {
        let evaluateRes = EvaluateRes(loss: 0.3, numExamples: 3, metrics: metrics, status: status)
        let proto = evaluateResToProto(res: evaluateRes)
        let result = evaluateResFromProto(msg: proto)
        XCTAssertEqual(evaluateRes, result)
    }
    
    func testPropertiesProtoConversion() throws {
        let proto = propertiesToProto(properties: properties)
        let result = propertiesFromProto(proto: proto)
        XCTAssertEqual(properties, result)
    }
    
    func testMetricsProtoConversion() throws {
        let proto = metricsToProto(metrics: metrics)
        let result = metricsFromProto(proto: proto)
        XCTAssertEqual(metrics, result)
    }
    
    func testScalarProtoConversion() throws {
        let pConv = ParameterConverter.shared
        let data = pConv.arrayToData(array: [1.1435, 2, 3, 4], shape: [2, 2])
        let scalar = Scalar(bool: true, bytes: data, float: 3.14159, int: 42, str: nil)
        let proto = try scalarToProto(scalar: scalar)
        let result = try scalarFromProto(scalarMsg: proto)
        XCTAssertEqual(scalar, result)
    }
}
