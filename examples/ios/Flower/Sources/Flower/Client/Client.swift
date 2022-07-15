//
//  Client.swift
//  FlowerSDK
//
//  Created by Daniel Nugraha on 09.01.22.
//

import Foundation

public protocol Client {
    func getParameters() -> ParametersRes
    func getProperties(ins: PropertiesIns) -> PropertiesRes
    func fit(ins: FitIns) -> FitRes
    func evaluate(ins: EvaluateIns) -> EvaluateRes
}

public extension Client {
    func getProperties(ins: PropertiesIns) -> PropertiesRes {
        return PropertiesRes(properties: [:])
    }
}
