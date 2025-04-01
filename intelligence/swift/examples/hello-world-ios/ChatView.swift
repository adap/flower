// Copyright 2025 Flower Labs GmbH. All Rights Reserved.
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

import FlowerIntelligence
import SwiftUI

struct ChatMessage: Identifiable, Equatable {
  static func == (lhs: ChatMessage, rhs: ChatMessage) -> Bool {
    return lhs.id == rhs.id
  }
  
  let id = UUID()
  let message: Message

  init(role: String, content: String) {
    self.message = Message(role: role, content: content)
  }
}

struct ChatView: View {
  @Environment(\.colorScheme) var colorScheme
  @State private var messages: [ChatMessage] = [
    ChatMessage(
      role: "system",
      content:
        "You are a helpful, respectful and honest assistant, except that you answer in Python language."
    )
  ]
  @State private var userInput: String = ""
  @State private var isLoading: Bool = false
  @State private var isHovered: Bool = false
  
  @State private var showScrollToBottomButton = false
  @State private var userHasScrolled = false
  @State private var keyboardHeight: CGFloat = 0

  @State private var selectedModel: String = "meta/llama3.2-1b"
  let availableModels = [
    "meta/llama3.2-1b", "meta/llama3.2-3b", "deepseek/r1-distill-qwen-32b/4-bit",
    "deepseek/r1-distill-llama-8b/q4", "meta/llama3.1-405b/q4", "deepseek/r1-685b/q4"
  ]

  var body: some View {
    VStack {
      ChatListView(messages: $messages)
        .simultaneousGesture(
          DragGesture().onChanged { value in
            if value.translation.height > 10 {
              hideKeyboard()
            }
          }
        )
        .padding(.bottom, 16)
      .mask(
        LinearGradient(
          gradient: Gradient(stops: [
            .init(color: Color.clear, location: 0.0),
            .init(color: Color.black, location: 0.1),
            .init(color: Color.black, location: 0.9),
            .init(color: Color.clear, location: 1.0),
          ]),
          startPoint: .top,
          endPoint: .bottom
        )
      )

      ChatInput(
        userInput: $userInput,
        sendMessage: { Task { await sendMessage() } },
        selectedModel: $selectedModel,
        availableModels: availableModels
      )
    }
  }

  func sendMessage() async {
    guard !userInput.isEmpty else { return }

    let userMessage = ChatMessage(role: "user", content: userInput)
    messages.append(userMessage)

    userInput = ""
    let assistantAnswer = ChatMessage(role: "assistant", content: "")
    messages.append(assistantAnswer)

    let fi = FlowerIntelligence.instance
    let result = await fi.chat(
      options: (
        messages.map { $0.message },
        ChatOptions(
          model: selectedModel, stream: true,
          onStreamEvent: { stream in
            DispatchQueue.main.async {
              _ = messages.removeLast()
              let newMessage = ChatMessage(
                role: "assistant", content: stream.chunk)
              messages.append(newMessage)
            }
          })
      ))
    messages.removeLast()
    switch result {
    case .success(let success):
      messages.append(ChatMessage(role: success.role, content: success.content))
    case .failure(let failure):
      print(failure)
    }
  }
}

struct ChatBubble: View {
  @State private var isEditing: Bool = false
  @State private var editedMessage: String = ""

  let message: ChatMessage
  let onSave: (String) -> Void

  var body: some View {
    HStack {
      if message.message.role == "user" {
        Spacer()
        Text(message.message.content)
          .padding()
          .background(Color.gray.opacity(0.2))
          .cornerRadius(20)
          .frame(maxWidth: 300, alignment: .trailing)
      } else if message.message.role == "system" {
        VStack(alignment: .leading) {
          HStack {
            Text("System instructions")
              
            Button(action: {
              isEditing.toggle()
              if !isEditing {
                onSave(editedMessage)
              }
            }) {
              Image(systemName: isEditing ? "checkmark.circle.fill" : "pencil")
                .foregroundColor(.gray)
            }
          }
          .padding(.top, 20).padding(.bottom, 4)

          if isEditing {
            TextEditor(text: $editedMessage)
              .padding()
              .textEditorStyle(.plain)
              .lineSpacing(5)
              .font(.custom("HelveticaNeue", size: 14))
              .background(Color.white)
              .cornerRadius(10)
          } else {
            Text(message.message.content)
          }
        }
        .padding()
        .frame(alignment: .leading)
        .foregroundColor(.gray)
        .onAppear {
          editedMessage = message.message.content
        }

        Spacer()
      } else {
        Text(message.message.content)
          .padding()
          .frame(alignment: .leading)
        Spacer()
      }
    }.padding(.trailing)
  }
}

#Preview {
  ChatView()
}

extension View {
  func hideKeyboard() {
    UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
  }
}
