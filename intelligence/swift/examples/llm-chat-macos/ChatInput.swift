//
//  ChatInput.swift
//  llm-chat-macos
//
//  Created by Daniel Nugraha on 19.03.25.
//

import SwiftUI

struct ChatInput: View {
    @Binding var userInput: String
    var sendMessage: () -> Void
    @Binding var selectedModel: String
    let availableModels: [String]
    @State private var isHovered: Bool = false

    var body: some View {
        VStack() {
            HStack {
                Image("flwr")
                    .resizable()
                    .scaledToFit()
                    .frame(width: 30, height: 30)

                TextField("Type a message...", text: $userInput)
                    .textFieldStyle(PlainTextFieldStyle())

                Button(action: {
                    sendMessage()
                }) {
                    Image(systemName: "arrow.up.circle.fill")
                        .resizable()
                        .scaledToFit()
                        .frame(width: 25, height: 25)
                }
                .buttonStyle(BorderlessButtonStyle())
            }
            .padding()
            .background(Color.white)
            .cornerRadius(30)

            Menu {
                ForEach(availableModels, id: \.self) { model in
                    Button(action: {
                        selectedModel = model
                    }) {
                        Text(model)
                    }
                }
            } label: {
                Text(selectedModel)
                    .foregroundColor(.gray)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                    .background(isHovered ? Color.gray.opacity(0.2) : Color.clear)
                    .cornerRadius(20)
                    .onHover { hovering in
                        isHovered = hovering
                    }
            }
            .buttonStyle(.plain)
        }
        .padding()
    }
}

#Preview {
  @Previewable @State var userInput = ""
  @Previewable @State var selectedModel = "meta/llama3.2-1b"
    
    return ChatInput(
        userInput: .constant(userInput),
        sendMessage: {},
        selectedModel: .constant(selectedModel),
        availableModels: [
            "meta/llama3.2-1b",
            "meta/llama3.2-3b",
            "deepseek/r1-distill-qwen-32b/4-bit",
            "deepseek/r1-distill-llama-8b/q4"
        ]
    )
}
