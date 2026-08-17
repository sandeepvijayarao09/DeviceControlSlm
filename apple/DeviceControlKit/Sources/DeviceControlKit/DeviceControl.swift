import Foundation

/// Errors surfaced by the device-control layer.
public enum DeviceControlError: Error, CustomStringConvertible {
    case unknownCommand(String)
    case unsupportedOnPlatform(command: String, platform: String)
    case invalidArgument(String)

    public var description: String {
        switch self {
        case .unknownCommand(let c): return "unknown command: \(c)"
        case .unsupportedOnPlatform(let c, let p): return "\(c) is not available on \(p)"
        case .invalidArgument(let a): return "invalid argument: \(a)"
        }
    }
}

/// Global entry point. `shared` is the concrete controller for this build;
/// inject a real per-OS controller on device (see DeviceController+iOS/macOS),
/// or leave the generated `StubDeviceController` for Simulator/Previews/tests.
public enum DeviceControl {
    public static var shared: DeviceControlling = StubDeviceController()
}
