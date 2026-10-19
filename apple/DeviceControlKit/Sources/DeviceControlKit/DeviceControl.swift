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

/// Global entry point. `shared` selects the real per-OS controller; unimplemented
/// commands fall through to the generated `unsupported` defaults (which makes the
/// recipe engine try Shortcuts / Accessibility next). Override with
/// `StubDeviceController()` in tests/Previews.
public enum DeviceControl {
    public static var shared: DeviceControlling = {
        #if os(macOS)
        return MacDeviceController()
        #elseif os(iOS)
        return IOSDeviceController()
        #else
        return StubDeviceController()
        #endif
    }()
}
