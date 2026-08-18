import Foundation

/// The on-device SLM, abstracted. A real implementation wraps the Core ML / MLX
/// model (P5); `StubLanguageModel` covers tests/Previews. Given a natural-language
/// request, it returns the device-control DSL:
///   "set_wifi|enabled=false ; set_brightness|level=30"  |  "ASK"  |  "REFUSE"
public protocol LanguageModel {
    func plan(_ request: String) async throws -> String
}

/// What the agent did with a request.
public enum AgentOutcome: Equatable {
    case executed([ExecutionResult])   // one entry per command attempted
    case clarify                       // under-specified — ask the user
    case refuse                        // out of scope — decline
}

/// The full on-device pipeline: request → model → DSL → parse → confirmation
/// gate → per-OS recipe execution. This is the Swift twin of the Python agent
/// loop (`dcslm/backend.py` + `agentcore.loop`) that DeviceControlBench scored,
/// so on-device behavior matches the benchmarked behavior.
public struct DeviceAgent {
    private let model: LanguageModel

    public init(model: LanguageModel) { self.model = model }

    @discardableResult
    public func handle(
        _ request: String,
        platform: DCPlatform = .current,
        confirm: @escaping (DeviceCommand) async -> Bool = { _ in true }
    ) async throws -> AgentOutcome {
        let dsl = try await model.plan(request)
        switch DeviceControlEngine.parse(dsl) {
        case .clarify:
            return .clarify
        case .refuse:
            return .refuse
        case .actions:
            let results = await DeviceControlEngine.execute(
                dsl: dsl, platform: platform, confirm: confirm)
            return .executed(results)
        }
    }
}

/// Deterministic model stub for tests/Previews (no ML). Wraps a closure.
public struct StubLanguageModel: LanguageModel {
    private let planner: (String) -> String
    public init(_ planner: @escaping (String) -> String) { self.planner = planner }
    public func plan(_ request: String) async throws -> String { planner(request) }
}
