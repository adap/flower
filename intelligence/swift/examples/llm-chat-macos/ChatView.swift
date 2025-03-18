import FlowerIntelligence
import SwiftUI

struct ChatMessage: Identifiable {
  let id = UUID()
  let message: Message

  init(role: String, content: String) {
    self.message = Message(role: role, content: content)
  }
}

struct ChatView: View {
  @State private var messages: [ChatMessage] = [
    ChatMessage(
      role: "system",
      content:
        "You are a helpful, respectful and honest assistant, except that you're currently drunk after having a few too many cocktails. You will try your very best to answer questions and respond to prompts, but you'll get sidetracked easily and have unrealistic, sometimes not-entirely-coherent ideas."
    )
  ]
  @State private var userInput: String = ""
  @State private var isLoading: Bool = false
  @State private var isHovered: Bool = false

  @State private var selectedModel: String = "meta/llama3.2-1b"
  let availableModels = [
    "meta/llama3.2-1b", "meta/llama3.2-3b", "deepseek/r1-distill-qwen-32b/4-bit",
    "deepseek/r1-distill-llama-8b/q4",
  ]

  var body: some View {
    VStack {
      ScrollView {
        VStack(alignment: .leading, spacing: 10) {
          ForEach(messages) { message in
            ChatBubble(message: message)
          }
          if isLoading {
            ProgressView("Searching files...")
              .padding()
          }
        }
        .padding()
        
      }.mask(
        LinearGradient(
            gradient: Gradient(stops: [
                .init(color: Color.clear, location: 0.0),
                .init(color: Color.black, location: 0.1),
                .init(color: Color.black, location: 0.9),
                .init(color: Color.clear, location: 1.0)
            ]),
            startPoint: .top,
            endPoint: .bottom
        )
    )

      

      VStack(alignment: .leading) {
        HStack {
          Image("flwr")
            .resizable() // Allows resizing
              .scaledToFit() // Maintains aspect ratio
              .frame(width: 30, height: 30)

          TextField("Type a message...", text: $userInput)
            .textFieldStyle(PlainTextFieldStyle())

          Button(action: {
            Task {
              await sendMessage()
            }
          }) {
            Image(systemName: "arrow.up.circle.fill")
              .resizable() // Allows resizing
                .scaledToFit() // Maintains aspect ratio
                .frame(width: 25, height: 25)

          }
          .buttonStyle(BorderlessButtonStyle())
        }
        .padding()
          .overlay(
            RoundedRectangle(cornerRadius: 30)
              .stroke(.gray.opacity(0.4), lineWidth: 2)
          )
          .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 30))
        Menu {
          // Dropdown items
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

        }.buttonStyle(.plain)
        

      }.padding()
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
              let message = messages.removeLast()
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
  let message: ChatMessage

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
          Text("System instructions").padding(.bottom, 4)
          Text(message.message.content)
        }
        
          .padding()
          .frame(alignment: .leading)
          .foregroundColor(.gray)
        Spacer()
      } else {
        Text(message.message.content)
          .padding()
          .frame(alignment: .leading)
        Spacer()
      }
    }
  }
}

#Preview {
  ChatView()
}
