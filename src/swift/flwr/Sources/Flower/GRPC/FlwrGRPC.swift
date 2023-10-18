//
//  FlwrGRPC.swift
//  
//
//  Created by Daniel Nugraha on 16.01.23.
//

import Foundation
import GRPC
import NIOCore
import os

/// A class that manages gRPC connection from client to the server.
///
/// ## Topics
///
/// ### Usage
///
/// - ``init(serverHost:serverPort:extendedInterceptor:)``
/// - ``startFlwrGRPC(client:)``
/// - ``startFlwrGRPC(client:completion:)``
/// - ``abortGRPCConnection(reasonDisconnect:completion:)``
/// - ``InterceptorExtension``
/// - ``GRPCPartWrapper``
///
/// ### GetParameters
///
/// - ``Parameters``
/// - ``GetParametersRes``
///
/// ### GetProperties
///
/// - ``GetPropertiesIns``
/// - ``GetPropertiesRes``
///
/// ### Fit
///
/// - ``FitIns``
/// - ``FitRes``
///
/// ### Evaluate
///
/// - ``EvaluateIns``
/// - ``EvaluateRes``
///
/// ### Reconnect
///
/// - ``Reconnect``
/// - ``Disconnect``
/// - ``ReasonDisconnect``
///
/// ### Supporting Messages
/// - ``Scalar``
/// - ``Status``
/// - ``Code``
///
/// ### Exceptions
/// - ``FlowerException``
@available(iOS 14.0, *)
public class FlwrGRPC {
    typealias GRPCResponse = (Flwr_Proto_ClientMessage, Int, Bool)
    
    private static let maxMessageLength: Int = 536870912
    private var bidirectionalStream: BidirectionalStreamingCall<Flwr_Proto_ClientMessage, Flwr_Proto_ServerMessage>? = nil
    
    private let eventLoopGroup: EventLoopGroup
    private let channel: GRPCChannel
    
    let extendedInterceptor: InterceptorExtension?
    
    private let log = Logger(subsystem: Bundle.main.bundleIdentifier ?? "flwr.Flower",
                                    category: String(describing: FlwrGRPC.self))
    /// Creates the client side communication class towards the server.
    ///
    /// - Parameters:
    ///   - serverHost: The address of the server.
    ///   - serverPort: The reserved server-side port.
    ///   - extendedInterceptor: A custom implementation of a communication interceptor.
    public init(serverHost: String, serverPort: Int, extendedInterceptor: InterceptorExtension? = nil) {
        self.extendedInterceptor = extendedInterceptor
        
        self.eventLoopGroup = PlatformSupport
            .makeEventLoopGroup(loopCount: 1, networkPreference: .best)
        
        let keepalive = ClientConnectionKeepalive(
          interval: .seconds(1000),
          timeout: .seconds(999),
          permitWithoutCalls: true,
          maximumPingsWithoutData: 0
        )
        
        self.channel = try! GRPCChannelPool.with(
            target: .host(serverHost, port: serverPort),
            transportSecurity: .plaintext,
            eventLoopGroup: eventLoopGroup
        ) {
            // Configure keepalive.
            $0.keepalive = keepalive
            $0.maximumReceiveMessageLength = FlwrGRPC.maxMessageLength
        }
    }
    
    /// Start a Flower client node which connects to a Flower server.
    ///
    /// - Parameters:
    ///   - client: The implementation of the Client which includes the machine learning routines and results.
    public func startFlwrGRPC(client: Client) {
        startFlwrGRPC(client: client) {}
    }
    
    /// Start a Flower client node which connects to a Flower server.
    ///
    /// - Parameters:
    ///   - client: The implementation of the Client which includes the machine learning routines and results.
    ///   - completion: A handler to define the action that will be executed after sending the response.
    public func startFlwrGRPC(client: Client, completion: @escaping () -> Void) {
        let grpcClient = Flwr_Proto_FlowerServiceNIOClient(channel: channel, interceptors: FlowerInterceptorsFactory(extendedInterceptor: self.extendedInterceptor))
        var callOptions = CallOptions()
        callOptions.customMetadata.add(name: "maxReceiveMessageLength", value: String(FlwrGRPC.maxMessageLength))
        callOptions.customMetadata.add(name: "maxSendMessageLength", value: String(FlwrGRPC.maxMessageLength))
        
        self.bidirectionalStream = grpcClient.join(callOptions: callOptions, handler: { sm in
            do {
                let promise = self.eventLoopGroup
                    .next()
                    .makePromise(of: GRPCResponse.self)
                let response = try handle(client: client, serverMsg: sm)
                promise.succeed(response)
                self.sendResponse(future: promise.futureResult, completion: completion)
            } catch {
                self.log.error("\(error)")
            }
        })
    }
    
    func sendResponse(future: EventLoopFuture<GRPCResponse>, completion: @escaping () -> Void) {
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                let response = try future.wait()
                _ = self.bidirectionalStream?.sendMessage(response.0)
                if !response.2 {
                    self.closeGRPCConnection(completion: completion)
                }
            } catch {
                self.log.error("\(error)")
            }
        }
    }

    func closeGRPCConnection(completion: @escaping () -> Void) {
        do {
            log.info("Closing gRPC bidirectional stream channel")
            try self.channel.close().wait()
            
            log.info("Closing gRPC event loop group")
            try self.eventLoopGroup.syncShutdownGracefully()
            
            completion()
            
        } catch {
            log.error("\(error)")
        }
    }
    
    /// Aborts the connection to the server on behalf of the client.
    ///
    /// - Parameters:
    ///   - completion: Handler function to define the action after closing the connection.
    public func abortGRPCConnection(reasonDisconnect: ReasonDisconnect, completion: @escaping () -> Void) {
        var disconnect = Flwr_Proto_ClientMessage.DisconnectRes()
        let reasonDisconnectProto = Flwr_Proto_Reason(rawValue: reasonDisconnect.rawValue)
        
        disconnect.reason = reasonDisconnectProto ?? .unknown
        var clientMessage = Flwr_Proto_ClientMessage()
        clientMessage.disconnectRes = disconnect
        
        _ = self.bidirectionalStream?.sendMessage(clientMessage)
        closeGRPCConnection(completion: completion)
    }
}
