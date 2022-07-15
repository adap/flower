//
//  MLLayerWrapper.swift
//  FlowerSDK
//
//  Created by Daniel Nugraha on 30.03.22.
//

import Foundation

struct MLLayerWrapper {
    let shape: [Int16]
    let name: String
    var weights: [Float]
    let isUpdatable: Bool
}
