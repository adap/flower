//
//  FlwrGRPCInterceptor.swift
//  
//
//  Created by Christoph Weinhuber on 20.01.23.
//

import Foundation
import GRPC

public protocol InterceptorExtension {
    func receive(part: GRPCPartWrapper)
    func send(part: GRPCPartWrapper)
}

public enum GRPCPartWrapper {
    case metadata(header: String)
    case message(content: String)
    case end(status: GRPCStatus?, trailers: String?)

}
