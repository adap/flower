//
//  ContentView.swift
//  FI-Swift-Examples
//
//  Created by Daniel Nugraha on 13.03.25.
//

import SwiftUI
import FlowerIntelligence

struct ContentView: View {
    @State var answer: String = ""
    var body: some View {
        Text("Why is the sky blue?")
        Text(answer)
            .task {
                let fi = FlowerIntelligence.instance

                let messages = [
                  Message(role: "system", content: "You are a helpful assistant."),
                  Message(role: "user", content: "Why is the sky blue?")
                ]

                let result = await fi.chat("Why is the sky blue?")
                switch result {
                case .success(let success):
                    answer = success.content
                case .failure(let failure):
                    answer = failure.message
                }
            }
    }
}
