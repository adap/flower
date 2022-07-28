//
//  Interceptors.swift
//  FlowerSDK
//
//  Created by Daniel Nugraha on 10.04.22.
//

import Foundation
import GRPC
import NIOCore

class FlowerClientInterceptors: ClientInterceptor<Flower_Transport_ClientMessage, Flower_Transport_ServerMessage> {
    override func receive(_ part: GRPCClientResponsePart<Flower_Transport_ServerMessage>, context: ClientInterceptorContext<Flower_Transport_ClientMessage, Flower_Transport_ServerMessage>) {
        switch part {
            // The response headers received from the server. We expect to receive these once at the start
            // of a response stream, however, it is also valid to see no 'metadata' parts on the response
            // stream if the server rejects the RPC (in which case we expect the 'end' part).
            case let .metadata(headers):
              print("< Received headers:", prettify(headers))

            // A response message received from the server. For unary and client-streaming RPCs we expect
            // one message. For server-streaming and bidirectional-streaming we expect any number of
            // messages (including zero).
            case let .message(response):
            print("< Received response '\(decipherServerMessage(response.msg!))' with text size '\(String(describing: response.msg).count)'")
            

            // The end of the response stream (and by extension, request stream). We expect one 'end' part,
            // after which no more response parts may be received and no more request parts will be sent.
            case let .end(status, trailers):
              print("< Response stream closed with status: '\(status)' and trailers:", prettify(trailers))
            }

        // Forward the response part to the next interceptor.
        context.receive(part)
    }
    
    override func send(_ part: GRPCClientRequestPart<Flower_Transport_ClientMessage>, promise: EventLoopPromise<Void>?, context: ClientInterceptorContext<Flower_Transport_ClientMessage, Flower_Transport_ServerMessage>) {
        switch part {
            // The (user-provided) request headers, we send these at the start of each RPC. They will be
            // augmented with transport specific headers once the request part reaches the transport.
            case let .metadata(headers):
              print("> Starting '\(context.path)' RPC, headers:", prettify(headers))

            // The request message and metadata (ignored here). For unary and server-streaming RPCs we
            // expect exactly one message, for client-streaming and bidirectional streaming RPCs any number
            // of messages is permitted.
            case let .message(request, _):
            print("> Sending request \(decipherClientMessage(request.msg!)) with text size \(String(describing: request.msg).count)")

            // The end of the request stream: must be sent exactly once, after which no more messages may
            // be sent.
            case .end:
              print("> Closing request stream")
            }

            // Forward the request part to the next interceptor.
            context.send(part, promise: promise)
    }
}

import NIOHPACK

func prettify(_ headers: HPACKHeaders) -> String {
  return "[" + headers.map { name, value, _ in
    "'\(name)': '\(value)'"
  }.joined(separator: ", ") + "]"
}

func decipherServerMessage(_ msg: Flower_Transport_ServerMessage.OneOf_Msg) -> String {
    switch msg {
    case .reconnect:
        return "Reconnect"
    case .getParameters:
        return "GetParameters"
    case .fitIns:
        return "FitIns"
    case .evaluateIns:
        return "EvaluateIns"
    case .propertiesIns:
        return "PropertiesIns"
    }

}

func decipherClientMessage(_ msg: Flower_Transport_ClientMessage.OneOf_Msg) -> String {
    switch msg {
    case .disconnect:
        return "Disconnect"
    case .evaluateRes:
        return "EvaluateRes"
    case .fitRes:
        return "FitRes"
    case .parametersRes:
        return "ParametersRes"
    case .propertiesRes:
        return "PropertiesRes"
    }
}

class FlowerInterceptorsFactory: Flower_Transport_FlowerServiceClientInterceptorFactoryProtocol {
    func makeJoinInterceptors() -> [ClientInterceptor<Flower_Transport_ClientMessage, Flower_Transport_ServerMessage>] {
        return [FlowerClientInterceptors()]
    }
    
}
