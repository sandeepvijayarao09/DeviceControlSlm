import Foundation

/// What the model decided for a request.
public enum DevicePlan: Equatable {
    case actions([DeviceCommand])
    case clarify
    case refuse
}

/// Outcome of attempting one command.
public struct ExecutionResult: Equatable {
    public let command: DeviceCommand
    public let mechanism: Mechanism
    public let ok: Bool
    public let message: String
}

/// Turns the SLM's DSL output into executed device actions, walking the per-OS
/// recipe and applying the confirmation gate. Mirrors the Python agent loop's
/// execution stage (dcslm/backend.py + agentcore.loop), so behavior matches
/// what DeviceControlBench scored.
public enum DeviceControlEngine {

    /// Parse the compact DSL:  "tool|k=v|k=v ; tool|k=v"  |  "ASK"  |  "REFUSE".
    public static func parse(_ dsl: String) -> DevicePlan {
        let trimmed = dsl.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed == "ASK" { return .clarify }
        if trimmed == "REFUSE" || trimmed.isEmpty { return .refuse }
        var cmds: [DeviceCommand] = []
        for part in trimmed.components(separatedBy: ";") {
            let fields = part.trimmingCharacters(in: .whitespaces).components(separatedBy: "|")
            guard let tool = fields.first?.trimmingCharacters(in: .whitespaces), !tool.isEmpty else { continue }
            var args: [String: String] = [:]
            for kv in fields.dropFirst() {
                let pair = kv.components(separatedBy: "=")
                if pair.count == 2 {
                    args[pair[0].trimmingCharacters(in: .whitespaces)] =
                        pair[1].trimmingCharacters(in: .whitespaces)
                }
            }
            cmds.append(DeviceCommand(tool: tool, args: args))
        }
        return cmds.isEmpty ? .refuse : .actions(cmds)
    }

    /// Execute the model's DSL. `confirm` is called for sensitive commands and
    /// must return true to proceed (the non-bypassable safety gate).
    @discardableResult
    public static func execute(
        dsl: String,
        platform: DCPlatform = .current,
        confirm: (DeviceCommand) async -> Bool = { _ in true }
    ) async -> [ExecutionResult] {
        guard case .actions(let cmds) = parse(dsl) else { return [] }
        var results: [ExecutionResult] = []
        for cmd in cmds {
            if Recipe.requiresConfirmation.contains(cmd.tool), await confirm(cmd) == false {
                results.append(.init(command: cmd, mechanism: .unsupported, ok: false,
                                     message: "declined at confirmation"))
                continue
            }
            results.append(await attempt(cmd, on: platform))
        }
        return results
    }

    /// Walk the recipe: try each mechanism until one succeeds.
    private static func attempt(_ cmd: DeviceCommand, on platform: DCPlatform) async -> ExecutionResult {
        let plan = Recipe.plan(cmd.tool, on: platform)
        for mech in plan {
            guard let executor = mech.executor() else { continue }
            do {
                try await executor.run(cmd)
                return .init(command: cmd, mechanism: mech, ok: true, message: "ok")
            } catch {
                continue   // this mechanism can't; fall back to the next
            }
        }
        return .init(command: cmd, mechanism: .unsupported, ok: false,
                     message: "no available mechanism on \(platform.rawValue)")
    }
}
