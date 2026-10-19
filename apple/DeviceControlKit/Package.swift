// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "DeviceControlKit",
    platforms: [.iOS(.v16), .macOS(.v13)],  // iPadOS ships within the iOS platform
    products: [
        .library(name: "DeviceControlKit", targets: ["DeviceControlKit"]),
    ],
    targets: [
        .target(name: "DeviceControlKit"),
        .testTarget(name: "DeviceControlKitTests", dependencies: ["DeviceControlKit"]),
    ]
)
