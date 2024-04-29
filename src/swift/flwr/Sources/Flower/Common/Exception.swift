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
/// - ``ValueError(_:)``
/// - ``UnknownServerMessage``
public enum FlowerException: Error {
    case ValueError(String)
    case UnknownServerMessage
}
