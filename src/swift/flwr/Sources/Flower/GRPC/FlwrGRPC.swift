//
//  File.swift
//  
//
//  Created by Daniel Nugraha on 16.01.23.
//

import Foundation
import GRPC
import NIOCore
import NIOPosix

public class FlwrGRPC {
    typealias GRPCResponse = (Flwr_Proto_ClientMessage, Int, Bool)
    
    static let maxMessageLength: Int = 536870912
    var bidirectionalStream: BidirectionalStreamingCall<Flwr_Proto_ClientMessage, Flwr_Proto_ServerMessage>? = nil
    
    let eventLoopGroup: EventLoopGroup
    let channel: GRPCChannel
    
    public init(serverHost: String, serverPort: Int) {
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
    
    public func startFlwrGRPC(client: Client) {
        let grpcClient = Flwr_Proto_FlowerServiceNIOClient(channel: channel, interceptors: FlowerInterceptorsFactory())
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
                self.sendResponse(future: promise.futureResult)
            } catch let error {
                print(error)
            }
        })
    }
    
    func sendResponse(future: EventLoopFuture<GRPCResponse>) {
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                let response = try future.wait()
                _ = self.bidirectionalStream?.sendMessage(response.0)
                if !response.2 {
                    self.closeGRPCConnection()
                }
            } catch let error {
                print(error)
            }
        }
    }
    
    func closeGRPCConnection() {
        do {
            print("Closing gRPC bidirectional stream channel")
            try self.channel.close().wait()
            
            print("Closing gRPC event loop group")
            try self.eventLoopGroup.syncShutdownGracefully()
            
        } catch let error {
            print(error)
        }
    }
    
}
