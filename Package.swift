// swift-tools-version: 6.0
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
  name: "flower",
  platforms: [.macOS(.v14), .iOS(.v16)],
  products: [
    // Products define the executables and libraries a package produces, making them visible to other packages.
    .library(
      name: "Flwr",
      targets: ["FlowerIntelligence"])
  ],
  dependencies: [
    // Dependencies declare other packages that this package depends on.
    // .package(url: /* package url */, from: "1.0.0"),
    .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.12.1"),
    .package(url: "https://github.com/huggingface/swift-transformers", from: "1.0.0"),
    .package(url: "https://github.com/apple/swift-async-algorithms", from: "1.0.0"),
    .package(url: "https://github.com/ml-explore/mlx-swift-lm/", .upToNextMinor(from: "2.29.1")),
    .package(url: "https://github.com/apple/swift-crypto.git", .upToNextMajor(from: "3.10.1")),
  ],
  targets: [
    // Targets are the basic building blocks of a package, defining a module or a test suite.
    // Targets can depend on other targets in this package and products from dependencies.
    .target(
      name: "FlowerIntelligence",
      dependencies: [
        .product(name: "MLX", package: "mlx-swift"),
        .product(name: "MLXFast", package: "mlx-swift"),
        .product(name: "MLXNN", package: "mlx-swift"),
        .product(name: "MLXOptimizers", package: "mlx-swift"),
        .product(name: "MLXRandom", package: "mlx-swift"),
        .product(name: "MLXLinalg", package: "mlx-swift"),
        .product(name: "Transformers", package: "swift-transformers"),
        .product(name: "AsyncAlgorithms", package: "swift-async-algorithms"),
        .product(name: "MLXLLM", package: "mlx-swift-lm"),
        .product(name: "Crypto", package: "swift-crypto"),
      ],
      path: "intelligence/swift/src"
    ),
    .testTarget(
      name: "FlowerIntelligenceTests",
      dependencies: ["FlowerIntelligence"],
      path: "intelligence/swift/tests"
    ),
  ]
)
