/*
  Copyright (c) 2017-2019 M.I. Hollemans

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to
  deal in the Software without restriction, including without limitation the
  rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
  sell copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in
  all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
  IN THE SOFTWARE.
*/

#if canImport(UIKit)

  import UIKit

  extension CGImagePropertyOrientation {
    public init(_ orientation: UIImage.Orientation) {
      switch orientation {
      case .up: self = .up
      case .upMirrored: self = .upMirrored
      case .down: self = .down
      case .downMirrored: self = .downMirrored
      case .left: self = .left
      case .leftMirrored: self = .leftMirrored
      case .right: self = .right
      case .rightMirrored: self = .rightMirrored
      @unknown default: self = .up
      }
    }
  }

  #if !os(tvOS)

    extension CGImagePropertyOrientation {
      public init(_ orientation: UIDeviceOrientation) {
        switch orientation {
        case .portraitUpsideDown: self = .left
        case .landscapeLeft: self = .up
        case .landscapeRight: self = .down
        default: self = .right
        }
      }
    }

  #endif

  extension UIImage.Orientation {
    init(_ cgOrientation: UIImage.Orientation) {
      switch cgOrientation {
      case .up: self = .up
      case .upMirrored: self = .upMirrored
      case .down: self = .down
      case .downMirrored: self = .downMirrored
      case .left: self = .left
      case .leftMirrored: self = .leftMirrored
      case .right: self = .right
      case .rightMirrored: self = .rightMirrored
      @unknown default: self = .up
      }
    }
  }

#endif
