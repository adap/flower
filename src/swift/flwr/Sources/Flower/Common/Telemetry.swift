//
//  Telemetry.swift
//  
//
//  Created by Daniel Nugraha on 18.09.23.
//

import Foundation
import Network
import UIKit

enum Event: String {
    case START_CLIENT_ENTER
    case START_CLIENT_LEAVE
}

struct Payload: Codable {
    let eventType: String
    let eventDetails: [String: String]
    let context: Context
}

struct Context: Codable {
    let source: String
    let cluster: String
    let date: String
    let flower: Flower
    let hw: HW
    let platform: Platform
}

struct Flower: Codable {
    let packageName: String
    let packageVersion: String
}

struct HW: Codable {
    let cpuCount: String
    let networkConnectionType: String
    let batteryLevel: String
}

struct Platform: Codable {
    let system: String
    let release: String
    let pythonImplementation: String
    let pythonVersion: String
    let androidSdkVersion: String
    let machine: String
    let architecture: String
    let version: String
}

func getSourceId() -> UUID {
    let flwrUrl = appDirectory.appendingPathComponent(".flwr")
    let fileManager = FileManager.default
    if fileManager.fileExists(atPath: flwrUrl.path) {
        if let uuidString = try? String(contentsOf: flwrUrl, encoding: .utf8) {
            return UUID(uuidString: uuidString) ?? UUID()
        }
        return UUID()
    } else {
        fileManager.createFile(atPath: flwrUrl.path, contents: nil)
        let uuid = UUID()
        let uuidString = uuid.uuidString
        try? uuidString.write(to: flwrUrl, atomically: true, encoding: .utf8)
        return uuid
    }
}

@available(iOS 15.0, *)
func createEventContext() -> Context {
    let date = Date.now.ISO8601Format()
    let version = "0.0.1"
    let pathMonitor = NWPathMonitor()
    
    let hw = HW(cpuCount: ProcessInfo.processInfo.processorCount.description,
                networkConnectionType: pathMonitor.currentPath.availableInterfaces[0].type.toString(),
                batteryLevel: UIDevice.current.batteryLevel.formatted())
    let platform = Platform(system: "iOS",
                            release: UIDevice.current.systemVersion,
                            pythonImplementation: "",
                            pythonVersion: "",
                            androidSdkVersion: "",
                            machine: "arm64e",
                            architecture: "",
                            version: "")
    let flower = Flower(packageName: "flwr", packageVersion: version)
    
    return Context(source: getSourceId().uuidString, cluster: UUID().uuidString, date: date, flower: flower, hw: hw, platform: platform)
}

@available(iOS 15.0, *)
func createEvent(event: Event, eventDetails: [String: String]) {
    let urlString = "https://telemetry.flower.dev/api/v1/event"
    guard let url = URL(string: urlString) else {
        return
    }
    Task {
        var request = URLRequest(url: url)
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpMethod = "POST"
        
        let payload = Payload(eventType: event.rawValue, eventDetails: eventDetails, context: createEventContext())
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        
        if let data = try? encoder.encode(payload) {
            request.httpBody = data
            _ = try? await URLSession.shared.data(for: request)
        }
    }
}

extension NWInterface.InterfaceType {
    public func toString() -> String {
        switch self {
        case .other:
            return "Other"
        case .wifi:
            return "Wifi"
        case .cellular:
            return "Cellular"
        case .wiredEthernet:
            return "WiredEthernet"
        case .loopback:
            return "Loopback"
        @unknown default:
            return "Unknown"
        }
    }
}
