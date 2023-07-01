//
//  Client.swift
//  FlowerSDK
//
//  Created by Daniel Nugraha on 09.01.22.
//

import Foundation

/// Protocol for Flower clients.
///
/// ## Topics
///
/// ### Functionalities
///
/// - ``getParameters()``
/// - ``getProperties(ins:)-4u0tf``
/// - ``fit(ins:)``
/// - ``evaluate(ins:)``
public protocol Client {
    
    /// Return the current local model parameters.
    func getParameters() -> GetParametersRes
    
    /// Return set of client properties.
    func getProperties(ins: GetPropertiesIns) -> GetPropertiesRes
    
    /// Refine the provided parameters using the locally held dataset.
    func fit(ins: FitIns) -> FitRes
    
    /// Evaluate the provided parameters using the locally held dataset.
    func evaluate(ins: EvaluateIns) -> EvaluateRes
}

public extension Client {
    /// Extension to Client since per default GetPropertiesIns is not implemented.
    func getProperties(ins: GetPropertiesIns) -> GetPropertiesRes {
        return GetPropertiesRes(properties: [:], status: Status(code: .getPropertiesNotImplemented, message: String()))
    }
}
