//
//  Client.swift
//  FlowerSDK
//
//  Created by Daniel Nugraha on 09.01.22.
//

import Foundation

/// The protocol class for the client implementation.
/// It contains abstract functions required for processing the server statements.
/// The expected return types are derived from the defined return structure.
public protocol Client {
    func getParameters() -> GetParametersRes
    func getProperties(ins: GetPropertiesIns) -> GetPropertiesRes
    func fit(ins: FitIns) -> FitRes
    func evaluate(ins: EvaluateIns) -> EvaluateRes
}

public extension Client {
    func getProperties(ins: GetPropertiesIns) -> GetPropertiesRes {
        return GetPropertiesRes(properties: [:], status: Status(code: .getPropertiesNotImplemented, message: String()))
    }
}
