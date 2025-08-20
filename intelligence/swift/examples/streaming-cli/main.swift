//
//  main.swift
//  FI-Swift-Examples
//
//  Created by Daniel Nugraha on 12.03.25.
//

import FlowerIntelligence

let fi = FlowerIntelligence.instance
fi.remoteHandoff = true
fi.apiKey = "ENTER YOUR API_KEY HERE"

let messages = [
  Message(role: "system", content: "You are a helpful assistant."),
  Message(role: "user", content: "Why is the sky blue?"),
]

let options = ChatOptions(
  model: "meta/llama3.2-1b",
  stream: true,
  onStreamEvent: { streamEvent in
    print(streamEvent.chunk)
  }
)

let result = await fi.chat(options: (messages, options))

if case .failure(let error) = result {
  print("Error: \(error.message)")
}
