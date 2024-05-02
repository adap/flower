//
//  Copyright 2024 Flower Labs GmbH. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================
//

import Foundation
import NIOCore
import NIOPosix
import GRPC

private struct GrpcRere {
    private let flwrLoop: EventLoopGroup
    private let grpcLoop: EventLoopGroup
    private let channel: GRPCChannel
    private let fleetStub: Flwr_Proto_FleetNIOClient
    private var node: Flwr_Proto_Node? = nil
    private var metadata: Metadata? = nil
    private var taskScheduler: RepeatedTask? = nil
    private var pingScheduler: RepeatedTask? = nil
    private var metadataDict: [String : Any] = [:]
    private var contextDict: [String : Context] = [:]
    
    init(serverHost: String, serverPort: Int) {
        self.flwrLoop = MultiThreadedEventLoopGroup(numberOfThreads: 1)
        self.grpcLoop = PlatformSupport
            .makeEventLoopGroup(loopCount: 1, networkPreference: .best)
        self.channel = startChannel(serverHost: serverHost, serverPort: serverPort, eventLoopGroup: self.grpcLoop)
        self.fleetStub = Flwr_Proto_FleetNIOClient(channel: self.channel)
    }
    
    fileprivate mutating func deleteNode() {
        if self.node != nil {
            return
        }
        // Stop ping
        
        let request = Flwr_Proto_DeleteNodeRequest()
        
        // Send request
        
        self.node = nil
    }
    
    fileprivate func sendWithRetry(request: Flwr_Proto_CreateNodeRequest, delay: TimeAmount = .zero) {
        let result = self.flwrLoop.next().scheduleTask(in: delay) {
            let call = self.fleetStub.createNode(request)
            do {
                let response = try call.response.wait()
            } catch {
                self.sendWithRetry(request: request, delay: exponentialDelay(delay))
            }
        }
    }
    
    fileprivate func exponentialDelay(_ timeAmount: TimeAmount) -> TimeAmount {
        let nanoseconds = timeAmount.nanoseconds
        if nanoseconds == 0 {
            return TimeAmount.seconds(1)
        }
        return TimeAmount.nanoseconds(nanoseconds * 2)
    }
}

public func runClientApp(serverHost: String, serverPort: Int) {
    let channel = startChannel(serverHost: serverHost, serverPort: serverPort, eventLoopGroup: PlatformSupport
        .makeEventLoopGroup(loopCount: 1, networkPreference: .best))
    var loop = MultiThreadedEventLoopGroup(numberOfThreads: 1)
    
    let grpcClient = Flwr_Proto_FleetNIOClient(channel: channel)
    
    
}


fileprivate func startChannel(serverHost: String, serverPort: Int, eventLoopGroup: EventLoopGroup) -> GRPCChannel {
    let maxMessageLength: Int = 536870912
    let channel = try! GRPCChannelPool.with(target: .host(serverHost, port: serverPort), transportSecurity: .plaintext, eventLoopGroup: eventLoopGroup) {
        $0.maximumReceiveMessageLength = maxMessageLength
    }
    return channel
}

fileprivate func schedulePullTaskIns(_ loop: EventLoopGroup, _ grpcClient: Flwr_Proto_FleetClientProtocol) {
    let executor = loop.next()
    executor.scheduleTask(in: .seconds(3)) {
        let response = grpcClient.pullTaskIns(Flwr_Proto_PullTaskInsRequest())
        schedulePullTaskIns(loop, grpcClient)
    }
}

fileprivate func createNode(_ loop: EventLoopGroup, _ grpcClient: Flwr_Proto_FleetClientProtocol) {
    let future = loop.next().submit {
        let response = grpcClient.createNode(Flwr_Proto_CreateNodeRequest())
        
    }
}

fileprivate func ping(_ loop: EventLoopGroup, _ grpcClient: Flwr_Proto_FleetClientProtocol) {
    
}
