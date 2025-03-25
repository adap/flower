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

import SwiftUI

struct ChatInput: View {
    @Environment(\.colorScheme) var colorScheme
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

              TextField("", text: $userInput)
                .foregroundColor(colorScheme == .dark ? .black : .gray)
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
