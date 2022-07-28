//
//  Start.swift
//  FlowerSDK
//
//  Created by Daniel Nugraha on 13.01.22.
//

import Foundation
import GRPC
import NIOCore
import NIOPosix
import Combine

let GRPCMaxMessageLength: Int = 536870912

@available(iOS 13.0, *)
public func startClient(serverHost: String, serverPort: Int, client: Client) {
    let maxMessageLength: Int = GRPCMaxMessageLength
    var sleepDuration = 0;
    let messagePublisher = CurrentValueSubject<Flower_Transport_ServerMessage?, Never>(nil)
    var serverMessage: Flower_Transport_ServerMessage?
    
    // Setup an `EventLoopGroup` for the connection to run on.
    //
    // See: https://github.com/apple/swift-nio#eventloops-and-eventloopgroups
    let group = MultiThreadedEventLoopGroup(numberOfThreads: 1)
    let transport = PlatformSupport.makeEventLoopGroup(loopCount: 1, networkPreference: .best)

    // Make sure the group is shutdown when we're done with it.
    defer {
        try! group.syncShutdownGracefully()
    }
    
    let keepalive = ClientConnectionKeepalive(
      interval: .seconds(1000),
      timeout: .seconds(999),
      permitWithoutCalls: true,
      maximumPingsWithoutData: 0
    )


    // Configure the channel, we're not using TLS so the connection is `insecure`.
    let channel = try! GRPCChannelPool.with(
        target: .host(serverHost, port: serverPort),
        transportSecurity: .plaintext,
        eventLoopGroup: transport
    ) {
        // Configure keepalive.
        $0.keepalive = keepalive
        $0.maximumReceiveMessageLength = 536870912
    }
    
    // Close the connection when we're done with it.
    defer {
        print("Closing gRPC bidirectional stream channel")
        try! channel.close().wait()
    }
    
    let grpcClient = Flower_Transport_FlowerServiceClient(channel: channel, interceptors: FlowerInterceptorsFactory())
    var callOptions = CallOptions()
    callOptions.customMetadata.add(name: "maxReceiveMessageLength", value: String(maxMessageLength))
    callOptions.customMetadata.add(name: "maxSendMessageLength", value: String(maxMessageLength))
    
    let bidirectional = grpcClient.join(callOptions: callOptions, handler: { sm in
        messagePublisher.send(sm)
        serverMessage = sm
    })
    
    while true {
        if let msg = serverMessage {
            let receive = try! handle(client: client, serverMsg: msg)
            let result = bidirectional.sendMessage(receive.0)
            sleepDuration = receive.1
            if !receive.2 {
                break
            }
            serverMessage = nil
        }
    }
}
