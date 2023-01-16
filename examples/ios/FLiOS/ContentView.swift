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
                                switch self.clientModel.trainingBatchStatus {
                                case BatchPreparationStatus.notPrepared:
                                    Text("Start")
                                case BatchPreparationStatus.ready:
                                    Image(systemName: "checkmark")
                                default:
                                    ProgressView()
                                }
                                
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
                                switch self.clientModel.testBatchStatus {
                                case BatchPreparationStatus.notPrepared:
                                    Text("Start")
                                case BatchPreparationStatus.ready:
                                    Image(systemName: "checkmark")
                                default:
                                    ProgressView()
                                }
                            }
                            .disabled(self.clientModel.trainingBatchStatus != BatchPreparationStatus.ready)
                        }
                        
                    }
                    Section(header: Text("Local Training")) {
                        HStack {
                            Text("Compile Model")
                            Spacer()
                            Button(action: {
                                clientModel.compileModel()
                            }) {
                                switch self.clientModel.modelCompilationStatus {
                                case BatchPreparationStatus.notPrepared:
                                    Text("Start")
                                case BatchPreparationStatus.ready:
                                    Image(systemName: "checkmark")
                                default:
                                    ProgressView()
                                }
                            }
                            .disabled(self.clientModel.testBatchStatus != BatchPreparationStatus.ready || self.clientModel.trainingBatchStatus != BatchPreparationStatus.ready)
                        }
                        
                        Stepper(value: self.$clientModel.epoch, in: 1...10, label: { Text("Epoch:  \(self.clientModel.epoch)")})
                            .disabled(self.clientModel.modelCompilationStatus != BatchPreparationStatus.ready)
                        
                        HStack {
                            switch self.clientModel.modelStatus {
                            case BatchPreparationModelStatus.notPrepared:
                                Text("Training")
                            default:
                                Text(self.clientModel.modelStatus.description)
                            }
                            Spacer()
                            Button(action: {
                                clientModel.startLocalTraining()
                            }) {
                                switch self.clientModel.modelStatus {
                                case BatchPreparationModelStatus.notPrepared:
                                    Text("Start")
                                case BatchPreparationModelStatus.ready:
                                    Image(systemName: "checkmark")
                                default:
                                    ProgressView()
                                }
                            }
                            .disabled(self.clientModel.modelCompilationStatus != BatchPreparationStatus.ready)
                        }
                        HStack {
                            Text("Predict Test data")
                            Spacer()
                            Button(action: {
                               //self.mnist.testModel()
                            }) {
                                Text("Start")
                            }
                            .disabled(!self.clientModel.modelStatus.description.hasPrefix("Training completed"))
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
                            if self.clientModel.federatedServerStatus == .run {
                                Button(action: {
                                    // TODO
                                }) {
                                    Text("Stop").foregroundColor(.red)
                                }
                            }
                            Spacer()
                            Button(action: {
                                clientModel.startFederatedLearning()
                            }) {
                                switch self.clientModel.federatedServerStatus {
                                case .stop:
                                    Text("Start")
                                default:
                                    ProgressView()
                                }
                                
                            }
                            .disabled(self.clientModel.modelCompilationStatus != BatchPreparationStatus.ready)
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
