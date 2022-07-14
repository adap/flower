//
//  ContentView.swift
//  FlowerCoreML
//
//  Created by Daniel Nugraha on 24.06.22.
//

import SwiftUI

struct ContentView: View {
    @ObservedObject var clientModel = ClientModel()
    
    func isDataReady(for status: BatchPreparationStatus) -> Bool {
        switch status {
        case .ready: return true
        default: return false
        }
    }

    func isDataPreparing(for status: BatchPreparationStatus) -> Bool {
        switch status {
        case .preparing: return true
        default: return false
        }
    }
    
    var numberFormatter: NumberFormatter = {
        var nf = NumberFormatter()
        nf.usesGroupingSeparator = false
        nf.numberStyle = .none
        return nf
    }()
    
    var body: some View {
        GeometryReader { geometry in
            VStack(spacing: 0) {
                Spacer().frame(height: 50)
                Text("Flower iOS Prototype")
                    .font(.largeTitle)
                Spacer()
                Form {
                    Section(header: Text("Preparing Dataset")) {
                        HStack {
                            Text("Training Dataset: \(self.clientModel.trainingBatchStatus.description)")
                            /*if self.isDataReady(for: self.model.trainingBatchStatus) {
                                Text(" \(self.mnist.trainingBatchProvider!.count) samples")
                            }*/
                            Spacer()
                            Button(action: {
                                clientModel.prepareTrainDataset()
                            }) {
                                Text("Start")
                            }
                            //.disabled(self.isDataPreparing(for: self.model.trainingBatchStatus))
                        }
                        HStack {
                            Text("Test Dataset: \(self.clientModel.testBatchStatus.description)")
                            /*if self.isDataReady(for: self.mnist.predictionBatchStatus) {
                                Text(" \(self.mnist.predictionBatchProvider!.count) samples")
                            }*/
                            Spacer()
                            Button(action: {
                                clientModel.prepareTestDataset()
                            }) {
                                Text("Start")
                            }
                            //.disabled(self.isDataPreparing(for: self.mnist.predictionBatchStatus))
                        }
                        
                    }
                    Section(header: Text("Local Training")) {
                        HStack {
                            Text("Compile Model")
                            Spacer()
                            Button(action: {
                                clientModel.compileModel()
                            }) {
                                Text("Start")
                            }
                            //.disabled(!self.isDataReady(for: self.mnist.trainingBatchStatus))
                        }
                        
                        Stepper(value: self.$clientModel.epoch, in: 1...10, label: { Text("Epoch:  \(self.clientModel.epoch)")})
                        
                        HStack {
                            Text(self.clientModel.modelStatus)
                            Spacer()
                            Button(action: {
                                clientModel.startLocalTraining()
                            }) {
                                Text("Start")
                            }
                            //.disabled(!self.mnist.modelCompiled)
                        }
                        HStack {
                            Text("Predict Test data")
                            Spacer()
                            Button(action: {
                               //self.mnist.testModel()
                            }) {
                                Text("Start")
                            }
                            //.disabled(!self.isDataReady(for: self.mnist.predictionBatchStatus) || !self.mnist.modelTrained)
                        }
                        //Text(self.mnist.accuracy)
                    }
                    Section(header: Text("Federated Learning")) {
                        HStack {
                            Text("Server Hostname: ")
                            TextField("Server Hostname", text: $clientModel.hostname)
                                .multilineTextAlignment(.trailing)
                        }
                        HStack {
                            Text("Server Port: ")
                            TextField( "Server Port", value: $clientModel.port, formatter: numberFormatter)
                                .multilineTextAlignment(.trailing)
                        }
                        HStack {
                            Spacer()
                            Button(action: {
                                clientModel.startFederatedLearning()
                            }) {
                                Text("Start")
                            }
                        }
                    }
                }
            }
            .background(Color(UIColor.systemGray6))
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
