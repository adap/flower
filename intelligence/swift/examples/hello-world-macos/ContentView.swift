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

struct ContentView: View {
  @State var answer: String = ""
  var body: some View {
    Text("Why is the sky blue?")
    Text(answer)
      .task {
        let fi = FlowerIntelligence.instance

        let messages = [
          Message(role: "system", content: "You are a helpful assistant."),
          Message(role: "user", content: "Why is the sky blue?"),
        ]

        let result = await fi.chat("Why is the sky blue?")
        switch result {
        case .success(let success):
          answer = success.content
        case .failure(let failure):
          answer = failure.message
        }
      }
  }
}
