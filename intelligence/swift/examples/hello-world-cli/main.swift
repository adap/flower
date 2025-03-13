//
//  main.swift
//  FI-Swift-Examples
//
//  Created by Daniel Nugraha on 12.03.25.
//

import FlowerIntelligence

let fi = FlowerIntelligence.instance

let messages = [
  Message(role: "system", content: "You are a helpful assistant."),
  Message(role: "user", content: "Why is the sky blue?")
]

let result = await fi.chat("Why is the sky blue?")
switch result {
case .success(let success):
    print(success.content)
case .failure(let failure):
    print(failure.message)
}
