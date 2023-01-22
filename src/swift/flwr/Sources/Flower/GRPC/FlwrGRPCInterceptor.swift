//
//  FlwrGRPCInterceptor.swift
//  
//
//  Created by Christoph Weinhuber on 20.01.23.
//

import Foundation

public protocol FlwrGRPCInterceptor {
    func receive()
    func send()
}
