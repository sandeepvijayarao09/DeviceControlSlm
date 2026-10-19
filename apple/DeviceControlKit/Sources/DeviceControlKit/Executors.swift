import Foundation

/// A parsed device command from the model's DSL (tool + string args).
public struct DeviceCommand: Equatable {
    public let tool: String
    public let args: [String: String]
    public init(tool: String, args: [String: String]) {
        self.tool = tool
        self.args = args
    }
}

/// Something that can attempt to carry out a command via one mechanism.
/// Throws `DeviceControlError.unsupportedOnPlatform` to signal "try the next
/// mechanism in the recipe".
public protocol MechanismExecutor {
    func run(_ cmd: DeviceCommand) async throws
}

/// 1) Direct framework API — routes to the generated `CommandRouter`, which
///    calls the per-OS `DeviceControlling` implementation.
public struct DirectAPIExecutor: MechanismExecutor {
    public init() {}
    public func run(_ cmd: DeviceCommand) async throws {
        _ = try await CommandRouter.run(tool: cmd.tool, args: cmd.args)
    }
}

/// 2) Shortcuts / App Intents system action — the iOS unlock for system toggles
///    (Set Wi-Fi, Set Focus, Set Low Power Mode, …). On device this invokes the
///    system-provided intent / runs the named Shortcuts action with parameters.
public struct ShortcutsExecutor: MechanismExecutor {
    public let action: String
    public init(action: String) { self.action = action }
    public func run(_ cmd: DeviceCommand) async throws {
        // TODO(device): invoke the system App Intent / run the Shortcuts action
        // `action` with `cmd.args` (e.g. via AppIntent.perform() or an
        // x-callback `shortcuts://run-shortcut`). Stubbed to log for now.
        let argstr = cmd.args.map { "\($0)=\($1)" }.joined(separator: " ")
        DeviceLog.emit("shortcut:\(action) \(cmd.tool) \(argstr)")
    }
}

/// 3) Accessibility automation — drive System Settings UI via AXUIElement when
///    no API/Shortcut exists (primarily macOS).
public struct AccessibilityExecutor: MechanismExecutor {
    public init() {}
    public func run(_ cmd: DeviceCommand) async throws {
        // TODO(device, macOS): use AXUIElement to open the relevant Settings pane
        // and toggle the control. Requires the Accessibility permission. Stubbed.
        DeviceLog.emit("accessibility:\(cmd.tool)")
    }
}

extension Mechanism {
    /// The executor that carries out this mechanism.
    func executor() -> MechanismExecutor? {
        switch self {
        case .directAPI: return DirectAPIExecutor()
        case .shortcut(let a): return ShortcutsExecutor(action: a)
        case .accessibility: return AccessibilityExecutor()
        case .unsupported: return nil
        }
    }
}
