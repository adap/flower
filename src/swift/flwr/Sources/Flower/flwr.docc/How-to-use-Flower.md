# How to use Flower iOS SDK

This guide outlines how to incorporate Flower iOS SDK into a new project using Swift Package Manager (SPM).

## Overview

Using Flower iOS SDK is very easy, you only need to create a new iOS project in Xcode and import Flower as a package dependency. But before starting, ensure you have Xcode installed on your macOS system. Now, follow these steps:

#### Create new iOS project with Xcode

1. Open Xcode and navigate to File > New > Project or use the shortcut ⇧ (Shift) + ⌘ (Command) + N.
2. Select a template for your project, for this tutorial we choose iOS > App and proceed by clicking Next.
3. Enter your desired project name, choose Swift as the language, and click Next.
4. Define the directory to store your project and click Create.

#### Import Flower to your new iOS project

Flower iOS SDK is available through Swift Package Manager (SPM) only. To import Flower in your Xcode project using SPM:

1. Go to File > Add Package Dependencies…
2. Enter the Flower iOS SDK package URL: [https://github.com/adap/flower-swift.git](https://github.com/adap/flower-swift.git)
3. Follow the on-screen prompts to add Flower iOS SDK to your project.
4. Once added, you can import Flower iOS SDK into your Swift files and utilize it within your code.

This concludes the tutorial on setting up a new Swift project and integrating Flower as a package dependency, check out our other tutorial on how to utilize Flower Swift SDK.

### API reference
- Flower Client: <doc:Client>
- Flower gRPC <doc:FlwrGRPC>
- Serialization <doc:ParameterConverter>
