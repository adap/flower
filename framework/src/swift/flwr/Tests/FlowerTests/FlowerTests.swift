//
//  FlowerTest.swift
//
//
//  Created by Maximilian Kapsecker on 23.01.23.
//

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
        guard let data1 = pConv.arrayToData(array: input1, shape: [2, 2]) else {
            XCTAssertTrue(false, "Failed to convert array to data.")
            return
        }
        
        let output1 = pConv.dataToArray(data: data1)
        XCTAssertEqual(input1, output1)
        
        guard let data2 = pConv.arrayToData(array: input2, shape: [2, 2]) else {
            XCTAssertTrue(false, "Failed to convert array to data.")
            return
        }
        
        let output2 = pConv.dataToArray(data: data2)
        XCTAssertEqual(input2, output2)
    }
    
    // Serde
    func testParameterProtoConverter() throws {
        
        guard let data1 = pConv.arrayToData(array: input1, shape: [2, 2]) else {
            XCTAssertTrue(false, "Failed to convert array to data.")
            return
        }
        guard let data2 = pConv.arrayToData(array: input2, shape: [2, 2]) else {
            XCTAssertTrue(false, "Failed to convert array to data.")
            return
        }
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
        
        guard let data1 = pConv.arrayToData(array: input1, shape: [2, 2]) else {
            XCTAssertTrue(false, "Failed to convert array to data.")
            return
        }
        guard let data2 = pConv.arrayToData(array: input2, shape: [2, 2]) else {
            XCTAssertTrue(false, "Failed to convert array to data.")
            return
        }
        let parameters = Parameters(tensors: [data1, data2], tensorType: "testTensor")
        
        let parametersRes = GetParametersRes(parameters: parameters, status: status)
        
        // To Proto
        let proto = parametersResToProto(res: parametersRes)
        //let statusProto = statusToProto(status: parametersRes.status)
        
        // From Proto
        let result = parametersResFromProto(msg: proto)
        //result.status = statusFromProto(msg: statusProto)
        XCTExpectFailure("Working on a fix for this problem.")
        XCTAssertEqual(parametersRes, result)
    }
    
    
    func testFitResProtoConversion() throws {
        
        guard let data1 = pConv.arrayToData(array: input1, shape: [2, 2]) else {
            XCTAssertTrue(false, "Failed to convert array to data.")
            return
        }
        guard let data2 = pConv.arrayToData(array: input2, shape: [2, 2]) else {
            XCTAssertTrue(false, "Failed to convert array to data.")
            return
        }
        let parameters = Parameters(tensors: [data1, data2], tensorType: "testTensor")
        
        let fitRes = FitRes(parameters: parameters, numExamples: 3, status: status)
        
        // To Proto
        let proto = fitResToProto(res: fitRes)
        let statusProto = statusToProto(status: fitRes.status)
        
        // From Proto
        var result = fitResFromProto(msg: proto)
        result.status = statusFromProto(msg: statusProto)
        XCTExpectFailure("Working on a fix for this problem.")
        XCTAssertEqual(fitRes, result)
    }
    
    func testFitInsProtoConversion() throws {
        
        guard let data1 = pConv.arrayToData(array: input1, shape: [2, 2]) else {
            XCTAssertTrue(false, "Failed to convert array to data.")
            return
        }
        guard let data2 = pConv.arrayToData(array: input2, shape: [2, 2]) else {
            XCTAssertTrue(false, "Failed to convert array to data.")
            return
        }
        
        let parameters = Parameters(tensors: [data1, data2], tensorType: "testTensor")
        let fitIns = FitIns(parameters: parameters, config: ["config1": scalar1, "config2": scalar2])
        
        let proto = fitInsToProto(ins: fitIns)
        let result = fitInsFromProto(msg: proto)
        XCTExpectFailure("Working on a fix for this problem.")
        XCTAssertEqual(fitIns, result)
    }
    
    func testPropertiesInsProtoConversion() throws {
        
        let propertiesIns = GetPropertiesIns(config: properties)
        let proto = propertiesInsToProto(ins: propertiesIns)
        let result = propertiesInsFromProto(msg: proto)
        XCTExpectFailure("Working on a fix for this problem.")
        XCTAssertEqual(propertiesIns, result)
    }
    
    func testPropertiesResProtoConversion() throws {
        let propertiesRes = GetPropertiesRes(properties: properties, status: status)
        
        // To Proto
        let proto = propertiesResToProto(res: propertiesRes)
        let protoStatus = statusToProto(status: propertiesRes.status)
        
        // From Proto
        var result = propertiesResFromProto(msg: proto)
        result.status = statusFromProto(msg: protoStatus)
        XCTExpectFailure("Working on a fix for this problem.")
        XCTAssertEqual(propertiesRes, result)
    }
    
    func testEvaluateInsProtoConversion() throws {
        
        guard let data1 = pConv.arrayToData(array: input1, shape: [2, 2]) else {
            XCTAssertTrue(false, "Failed to convert array to data.")
            return
        }
        guard let data2 = pConv.arrayToData(array: input2, shape: [2, 2]) else {
            XCTAssertTrue(false, "Failed to convert array to data.")
            return
        }
        let parameters = Parameters(tensors: [data1, data2], tensorType: "testTensor")
        
        let evaluateIns = EvaluateIns(parameters: parameters, config: ["config1": scalar1, "config2": scalar2])
        let proto = evaluateInsToProto(ins: evaluateIns)
        let result = evaluateInsFromProto(msg: proto)
        XCTExpectFailure("Working on a fix for this problem.")
        XCTAssertEqual(evaluateIns, result)
    }
    
    func testEvaluateResProtoConversion() throws {
        let evaluateRes = EvaluateRes(loss: 0.3, numExamples: 3, metrics: metrics, status: status)
        let proto = evaluateResToProto(res: evaluateRes)
        let result = evaluateResFromProto(msg: proto)
        XCTExpectFailure("Working on a fix for this problem.")
        XCTAssertEqual(evaluateRes, result)
    }
    
    func testPropertiesProtoConversion() throws {
        let proto = propertiesToProto(properties: properties)
        let result = propertiesFromProto(proto: proto)
        XCTExpectFailure("Working on a fix for this problem.")
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
        XCTExpectFailure("Working on a fix for this problem.")
        XCTAssertEqual(scalar, result)
    }
    
    func testDisconnectReasonTransformation() throws {
        let reasonUnkownProto = Flwr_Proto_Reason.unknown.rawValue
        let reasonUnkown = ReasonDisconnect.unknown.rawValue
        XCTAssertEqual(reasonUnkownProto, reasonUnkown)
        let reasonPowerProto = Flwr_Proto_Reason.powerDisconnected.rawValue
        let reasonPower = ReasonDisconnect.powerDisconnected.rawValue
        XCTAssertEqual(reasonPowerProto, reasonPower)
        let reasonReconnectProto = Flwr_Proto_Reason.reconnect.rawValue
        let reasonReconnect = ReasonDisconnect.reconnect.rawValue
        XCTAssertEqual(reasonReconnectProto, reasonReconnect)
        let reasonAckProto = Flwr_Proto_Reason.ack.rawValue
        let reasonAck = ReasonDisconnect.ack.rawValue
        XCTAssertEqual(reasonAckProto, reasonAck)
        let reasonWifiUnavailableProto = Flwr_Proto_Reason.wifiUnavailable.rawValue
        let reasonWifiUnavailable = ReasonDisconnect.wifiUnavailable.rawValue
        XCTAssertEqual(reasonWifiUnavailableProto, reasonWifiUnavailable)
    }
}
