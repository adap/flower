//
//  Exception.swift
//  FlowerSDK
//
//  Created by Daniel Nugraha on 09.01.22.
//

import Foundation

public enum FlowerException: Error {
    case TypeException(String)
    case UnknownServerMessage
}
