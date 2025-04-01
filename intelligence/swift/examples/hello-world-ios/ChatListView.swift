//
//  ChatListView.swift
//  FlowerIntelligenceExamples
//
//  Created by Daniel Nugraha on 25.03.25.
//
import SwiftUI

struct ChatListView: View {
  @Binding var messages: [ChatMessage]
  @State private var showScrollToBottomButton = false
  @State private var userHasScrolled = false
  @State private var keyboardHeight: CGFloat = 0
  
  var body: some View {
    ScrollViewReader { value in
      ScrollView {
        VStack {
          ForEach(messages) { message in
              ChatBubble(message: message) { updatedContent in
                  if let index = messages.firstIndex(where: { $0.id == message.id }) {
                      messages[index] = ChatMessage(role: "system", content: updatedContent)
                  }
              }
          }
          GeometryReader { geometry -> Color in
            DispatchQueue.main.async {
              let maxY = geometry.frame(in: .global).maxY
              print(maxY)
              let screenHeight = UIScreen.main.bounds.height - keyboardHeight
              print(screenHeight)
              let isBeyondBounds = maxY > screenHeight - 50
              if showScrollToBottomButton != isBeyondBounds {
                print(showScrollToBottomButton)
                print(userHasScrolled)
                showScrollToBottomButton = isBeyondBounds
                userHasScrolled = isBeyondBounds
              }
            }
            return Color.clear
          }
          .frame(height: 0)
        }
      }
      .onChange(of: messages) {
        if !userHasScrolled, let lastMessageId = messages.last?.id {
          withAnimation {
            value.scrollTo(lastMessageId, anchor: .bottom)
          }
        }
      }
      .overlay(
        Group {
          if showScrollToBottomButton {
            Button(action: {
              withAnimation {
                if let lastMessageId = messages.last?.id {
                  print(lastMessageId)
                  value.scrollTo(lastMessageId, anchor: .bottom)
                }
                userHasScrolled = false
              }
            }) {
              ZStack {
                Circle()
                  .fill(Color(UIColor.secondarySystemBackground).opacity(0.9))
                  .frame(height: 28)
                Image(systemName: "arrow.down.circle")
                  .resizable()
                  .aspectRatio(contentMode: .fit)
                  .frame(height: 28)
              }
            }
            .transition(AnyTransition.opacity.animation(.easeInOut(duration: 0.2)))
          }
        },
        alignment: .bottom
      )
    }
  }
}

#Preview {
  
  ChatListView(messages: .constant([
    ChatMessage(
      role: "system",
      content:
        "You are a helpful, respectful and honest assistant, except that you answer in Python language."
    ),ChatMessage(
      role: "user",
      content:
        "You are a helpful, respectful and honest assistant, except that you answer in Python language."
    ),ChatMessage(
      role: "system",
      content:
        "You are a helpful, respectful and honest assistant, except that you answer in Python language."
    ),ChatMessage(
      role: "system",
      content:
        "You are a helpful, respectful and honest assistant, except that you answer in Python language."
    ),ChatMessage(
      role: "system",
      content:
        "You are a helpful, respectful and honest assistant, except that you answer in Python language."
    ),ChatMessage(
      role: "system",
      content:
        "You are a helpful, respectful and honest assistant, except that you answer in Python language."
    ),ChatMessage(
      role: "system",
      content:
        "You are a helpful, respectful and honest assistant, except that you answer in Python language."
    ),ChatMessage(
      role: "system",
      content:
        "You are a helpful, respectful and honest assistant, except that you answer in Python language."
    ),ChatMessage(
      role: "system",
      content:
        "You are a helpful, respectful and honest assistant, except that you answer in Python language."
    ),ChatMessage(
      role: "system",
      content:
        "You are a helpful, respectful and honest assistant, except that you answer in Python language."
    ),ChatMessage(
      role: "system",
      content:
        "You are a helpful, respectful and honest assistant, except that you answer in Python language."
    ),ChatMessage(
      role: "system",
      content:
        "You are a helpful, respectful and honest assistant, except that you answer in Python language."
    )
  ]))
}
