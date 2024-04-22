# Incorporate Flower iOS SDK in Your Project Tutorial 

In this tutorial, we will learn how to incorporate Flower iOS SDK into a new project using Swift Package Manager (SPM).

## Overview

Using Flower iOS SDK is very easy, and we will go through the steps one-by-one. First, ensure that you have Xcode installed on your macOS system. Then, follow these steps:

#### Create new iOS project with Xcode

1. Open Xcode and navigate to File > New > Project or use the shortcut ⇧ (Shift) + ⌘ (Command) + N.
2. Select a template for your project, for this tutorial we choose iOS > App and proceed by clicking Next.
3. Enter your desired project name, choose Swift as the language, and click Next.
4. Define the directory to store your project and click Create.

#### Import Flower to your new iOS project

Notice that Flower iOS SDK is available through Swift Package Manager (SPM) only. To import Flower in your Xcode project using SPM:

1. Go to File > Add Package Dependencies…
2. Enter the Flower iOS SDK package URL: [https://github.com/adap/flower-swift.git](https://github.com/adap/flower-swift.git)
3. Follow the on-screen prompts to add Flower iOS SDK to your project.
4. Once added, you can import Flower iOS SDK into your Swift files and utilize it within your code.

Now, you have successfully incorporated Flower iOS SDK in your own project. If you want to learn more about Flower iOS SDK, feel free to check out our other tutorial on how to utilize Flower Swift SDK.

### API reference
- Flower Client: <doc:Client>
- Flower gRPC <doc:FlwrGRPC>
- Serialization <doc:ParameterConverter>
