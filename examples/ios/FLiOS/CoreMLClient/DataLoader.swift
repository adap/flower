//
//  DataLoader.swift
//  FlowerIOS
//
//  Created by Daniel Nugraha on 14.06.22.
//

import Foundation
import CoreML

public struct DataLoader {
    let trainBatchProvider: MLBatchProvider
    let testBatchProvider: MLBatchProvider
}
