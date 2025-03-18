//
//  ContentView.swift
//  FI-Swift-Examples
//
//  Created by Daniel Nugraha on 13.03.25.
//

import FlowerIntelligence
import SwiftUI

struct ContentView: View {
  @State var answer: String = ""
  var body: some View {
    ChatView()
      .task {
        let fi = FlowerIntelligence.instance
        await fi.fetchModel(model: "deepseek/r1-distill-qwen-32b/4-bit") {
          print("\($0.description): \($0.percentage), \($0.loadedBytes), \($0.totalBytes)")
          
        }
      }
  }
}

#Preview {
    ContentView()
}
