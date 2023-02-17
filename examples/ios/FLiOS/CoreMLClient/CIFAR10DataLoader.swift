//
//  CIFAR10DataLoader.swift
//  FLiOS
//
//  Created by Daniel Nugraha on 07.02.23.
//

import Foundation

class CIFAR10DataLoader {
    private static let cifarUrl = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
    
    static func downloadCIFAR() {
        let downloadTask = URLSession.shared.downloadTask(with: URL(string: cifarUrl)!) {
            urlOrNil, responseOrNil, errorOrNil in
                // check for and handle errors:
                // * errorOrNil should be nil
                // * responseOrNil should be an HTTPURLResponse with statusCode in 200..<299
                
                guard let fileURL = urlOrNil else { return }
                do {
                    let documentsURL = try
                        FileManager.default.url(for: .documentDirectory,
                                                in: .userDomainMask,
                                                appropriateFor: nil,
                                                create: false)
                    let savedURL = documentsURL.appendingPathComponent(fileURL.lastPathComponent)
                    try FileManager.default.moveItem(at: fileURL, to: savedURL)
                } catch {
                    print ("file error: \(error)")
                }
        }
        downloadTask.resume()
        
        FileManager.default
    }
    
    
    
}
