import SwiftUI
import DeviceControlKit

@main
struct DeviceControlApp: App {
    var body: some Scene {
        WindowGroup { ContentView() }
    }
}

/// Selects the on-device model. Uses the Core ML SLM when the .mlmodelc is
/// bundled; otherwise falls back to a tiny heuristic so the app runs in the
/// Simulator / Previews before the model is exported.
enum AppModel {
    static let shared: LanguageModel = {
        if let m = try? CoreMLLanguageModel() { return m }
        return HeuristicLanguageModel()
    }()
}

/// Fallback planner (no ML): enough to demo the pipeline before the Core ML
/// model is added. Not a substitute for the SLM — just keyword rules.
struct HeuristicLanguageModel: LanguageModel {
    func plan(_ request: String) async throws -> String {
        let r = request.lowercased()
        var cmds: [String] = []
        if r.contains("wifi") { cmds.append("set_wifi|enabled=\(r.contains("off") ? "false" : "true")") }
        if r.contains("bluetooth") { cmds.append("set_bluetooth|enabled=\(r.contains("off") ? "false" : "true")") }
        if let m = r.range(of: #"brightness.*?(\d+)"#, options: .regularExpression) {
            let n = String(r[m]).filter(\.isNumber); cmds.append("set_brightness|level=\(n)")
        }
        if r.contains("flashlight") || r.contains("torch") {
            cmds.append("set_flashlight|enabled=\(r.contains("off") ? "false" : "true")")
        }
        if cmds.isEmpty { return r.count < 12 ? "ASK" : "REFUSE" }
        return cmds.joined(separator: " ; ")
    }
}
