//
//  main.swift
//  FI-Swift-Examples
//
//  Created by Daniel Nugraha on 12.03.25.
//

import FlowerIntelligence

let fi = FlowerIntelligence.instance
fi.remoteHandoff = true
fi.apiKey = "fk_0_vPNdWiAlmvwuf4Pn_QyeUKPI1AmFU_6zPOj2mGtmO2s"

let messages = [
  Message(role: "system", content: "You are a helpful assistant."),
  Message(role: "user", content: "Why is the sky blue?")
]

let options = ChatOptions(
  model: "meta/llama3.2-1b"
)

let result = await fi.chat(options: (messages, options))

if case .failure(let error) = result {
  print("Error: \(error.message)")
}
