//
//  Interceptors.swift
//  FlowerSDK
//
//  Created by Daniel Nugraha on 10.04.22.
//

import Foundation
import GRPC
import NIOCore
import os

@available(iOS 14.0, *)
class FlowerClientInterceptors: ClientInterceptor<Flwr_Proto_ClientMessage, Flwr_Proto_ServerMessage> {
    var extendedInterceptor: InterceptorExtension?
    private let log = Logger(subsystem: Bundle.main.bundleIdentifier ?? "flwr.Flower",
                                    category: String(describing: FlowerClientInterceptors.self))
    
    init(extendedInterceptor: InterceptorExtension?) {
        self.extendedInterceptor = extendedInterceptor
    }
    
    override func receive(_ part: GRPCClientResponsePart<Flwr_Proto_ServerMessage>, context: ClientInterceptorContext<Flwr_Proto_ClientMessage, Flwr_Proto_ServerMessage>) {
        
        // wrapper to expose message for additional interceptor
        let grpcPart: GRPCPartWrapper
        
        switch part {
            // The response headers received from the server. We expect to receive these once at the start
            // of a response stream, however, it is also valid to see no 'metadata' parts on the response
            // stream if the server rejects the RPC (in which case we expect the 'end' part).
        case let .metadata(headers):
            let formatted = prettify(headers)
            log.info("< Received headers: \(formatted)")
            
            grpcPart = .metadata(header: formatted)
            
            // A response message received from the server. For unary and client-streaming RPCs we expect
            // one message. For server-streaming and bidirectional-streaming we expect any number of
            // messages (including zero).
        case let .message(response):
            let msg = String(describing: response.msg)
            log.info("< Received response '\(decipherServerMessage(response.msg!))' with text size '\(msg.count)'")
            
            grpcPart = .message(content: msg)

            // The end of the response stream (and by extension, request stream). We expect one 'end' part,
            // after which no more response parts may be received and no more request parts will be sent.
        case let .end(status, trailers):
            let formatted = prettify(trailers)
            log.info("< Response stream closed with status: '\(status)' and trailers: \(formatted)")
            
            grpcPart = .end(status: status, trailers: formatted)
        }
        
        
        // Forward the response part to the next interceptor.
        context.receive(part)
        
        // Forward part to custom user Interceptor
        extendedInterceptor?.receive(part: grpcPart)
    }
    
    override func send(_ part: GRPCClientRequestPart<Flwr_Proto_ClientMessage>, promise: EventLoopPromise<Void>?, context: ClientInterceptorContext<Flwr_Proto_ClientMessage, Flwr_Proto_ServerMessage>) {
        
        // wrapper to expose message for additional interceptor
        let grpcPart: GRPCPartWrapper
        
        switch part {
            // The (user-provided) request headers, we send these at the start of each RPC. They will be
            // augmented with transport specific headers once the request part reaches the transport.
        case let .metadata(headers):
            let formatted = prettify(headers)
            log.info("> Starting '\(context.path)' RPC, headers: \(formatted)")
            
            grpcPart = .metadata(header: formatted)
            
            // The request message and metadata (ignored here). For unary and server-streaming RPCs we
            // expect exactly one message, for client-streaming and bidirectional streaming RPCs any number
            // of messages is permitted.
        case let .message(request, _):
            let msg = String(describing: request.msg)
            log.info("> Sending request \(decipherClientMessage(request.msg!)) with text size \(msg.count)")
            
            grpcPart = .message(content: msg)
            
            // The end of the request stream: must be sent exactly once, after which no more messages may
            // be sent.
        case .end:
            log.info("> Closing request stream")
            
            grpcPart = .end(status: nil, trailers: nil)
        }
        
        // Forward the request part to the next interceptor.
        context.send(part, promise: promise)
        
        // Forward part to custom user Interceptor
        extendedInterceptor?.send(part: grpcPart)
    }
}

import NIOHPACK

func prettify(_ headers: HPACKHeaders) -> String {
    return "[" + headers.map { name, value, _ in
        "'\(name)': '\(value)'"
    }.joined(separator: ", ") + "]"
}

func decipherServerMessage(_ msg: Flwr_Proto_ServerMessage.OneOf_Msg) -> String {
    switch msg {
    case .reconnectIns:
        return "Reconnect"
    case .getParametersIns:
        return "GetParameters"
    case .fitIns:
        return "FitIns"
    case .evaluateIns:
        return "EvaluateIns"
    case .getPropertiesIns:
        return "PropertiesIns"
    }
    
}

func decipherClientMessage(_ msg: Flwr_Proto_ClientMessage.OneOf_Msg) -> String {
    switch msg {
    case .disconnectRes:
        return "Disconnect"
    case .evaluateRes:
        return "EvaluateRes"
    case .fitRes:
        return "FitRes"
    case .getParametersRes:
        return "ParametersRes"
    case .getPropertiesRes:
        return "PropertiesRes"
    }
}

@available(iOS 14.0, *)
final class FlowerInterceptorsFactory: Flwr_Proto_FlowerServiceClientInterceptorFactoryProtocol {
    let extendedInterceptor: InterceptorExtension?
    
    init(extendedInterceptor: InterceptorExtension? = nil) {
        self.extendedInterceptor = extendedInterceptor
    }
    
    func makeJoinInterceptors() -> [ClientInterceptor<Flwr_Proto_ClientMessage, Flwr_Proto_ServerMessage>] {
        return [FlowerClientInterceptors(extendedInterceptor: self.extendedInterceptor)]
    }
    
}
