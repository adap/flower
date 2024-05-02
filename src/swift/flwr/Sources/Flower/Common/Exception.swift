//
//  Exception.swift
//  FlowerSDK
//
//  Created by Daniel Nugraha on 09.01.22.
//

import Foundation

/// Set of Flower client exceptions.
///
/// ## Topics
///
/// ### Exceptions
///
/// - ``TypeException(_:)``
/// - ``UnknownServerMessage``
public enum FlowerException: Error {
    case TypeError(String)
    case ValueError(String)
    case UnknownServerMessage
}
