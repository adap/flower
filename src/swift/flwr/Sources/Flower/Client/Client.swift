//
//  Client.swift
//  FlowerSDK
//
//  Created by Daniel Nugraha on 09.01.22.
//

import Foundation

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
