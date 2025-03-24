//
//  ChatViewModel.swift
//  FlowerIntelligenceExamples
//
//  Created by Daniel Nugraha on 24.03.25.
//

import SwiftUI

@MainActor
@Observable final class ChatViewModel {
  private var process: Process?
  private var inputPipe: Pipe?
  private var outputPipe: Pipe?
  var messages: [ChatMessage] = [
    ChatMessage(
      role: "system",
      content:
        "You are a helpful, respectful and honest assistant, except that you're currently drunk after having a few too many cocktails. You will try your very best to answer questions and respond to prompts, but you'll get sidetracked easily and have unrealistic, sometimes not-entirely-coherent ideas and you reply with a lot of emojis."
    )
  ]

  func start(_ modelName: String? = nil) {
    process = Process()
    inputPipe = Pipe()
    outputPipe = Pipe()

    guard let process = process, let inputPipe = inputPipe, let outputPipe = outputPipe else {
      return
    }

    process.executableURL = URL(fileURLWithPath: "/Users/danielnugraha/.pyenv/versions/flower-3.10.13/bin/python3")
    process.arguments = ["/Users/danielnugraha/Documents/work/flower/intelligence/swift/examples/llm-chat-macos/mlx_lm_chat.py"]
    
    process.standardInput = inputPipe
    process.standardOutput = outputPipe
    process.standardError = outputPipe

    let outHandle = outputPipe.fileHandleForReading
    outHandle.readabilityHandler = { [weak self] handle in
      guard let self = self else { return }
      let data = handle.availableData
      if let chunk = String(data: data, encoding: .utf8) {
        DispatchQueue.main.async {
          self.handleOutput(chunk)
        }
      }
    }

    do {
      try process.run()
    } catch {
      print("Error launching Python script: \(error)")
    }
  }

  private var buffer = ""
  private var isStreaming = false
  private var startOutputing = false

  func send(_ prompt: String) {
    guard let inputPipe = inputPipe else { return }
    if !prompt.hasSuffix("\n") {
      if let data = (prompt + "\n").data(using: .utf8) {
        inputPipe.fileHandleForWriting.write(data)
      }
    } else {
      if let data = prompt.data(using: .utf8) {
        inputPipe.fileHandleForWriting.write(data)
      }
    }
    isStreaming = true
  }

  private func handleOutput(_ chunk: String) {
    if !chunk.contains("[START]") && !startOutputing { return }
    
    if chunk.contains("[START]") {
      startOutputing = true
      return
    }
    
    let message = messages.removeLast()
    let newMessage: ChatMessage
    if chunk.contains("[END]") {
      isStreaming = false
      startOutputing = false
      newMessage = ChatMessage(
        role: "assistant", content: message.message.content + chunk.replacingOccurrences(of: "[END]", with: ""))
    } else {
      newMessage = ChatMessage(
        role: "assistant", content: message.message.content + chunk)
    }
    
    messages.append(newMessage)

  }

  func stop() {
    process?.terminate()
  }
}
