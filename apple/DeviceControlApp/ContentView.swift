import SwiftUI
import DeviceControlKit

/// Minimal cross-platform (iOS / iPadOS / macOS) UI: type a request, the SLM
/// plans it, the engine executes it through the per-OS recipe, and sensitive
/// commands surface a confirmation alert (the safety gate).
struct ContentView: View {
    @State private var request = ""
    @State private var log: [String] = []
    @State private var busy = false

    // Confirmation-gate bridge: the engine's async `confirm` closure suspends
    // here until the user answers the alert.
    @State private var pending: DeviceCommand?
    @State private var continuation: CheckedContinuation<Bool, Never>?

    private let agent = DeviceAgent(model: AppModel.shared)

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Device Control").font(.title2).bold()
            HStack {
                TextField("e.g. turn off wifi and set brightness to 30", text: $request)
                    .textFieldStyle(.roundedBorder)
                    .onSubmit(run)
                Button("Run", action: run).disabled(busy || request.isEmpty)
            }
            List(Array(log.enumerated()), id: \.offset) { _, line in Text(line).font(.callout.monospaced()) }
        }
        .padding()
        .alert("Confirm action", isPresented: .constant(pending != nil)) {
            Button("Cancel", role: .cancel) { resolve(false) }
            Button("Allow", role: .destructive) { resolve(true) }
        } message: {
            Text(pending.map { "Allow \($0.tool.replacingOccurrences(of: "_", with: " "))?" } ?? "")
        }
    }

    private func run() {
        let text = request
        busy = true
        log.insert("▶︎ \(text)", at: 0)
        Task {
            let outcome = try? await agent.handle(text, confirm: { cmd in
                await withCheckedContinuation { cont in
                    pending = cmd; continuation = cont
                }
            })
            await MainActor.run {
                switch outcome {
                case .executed(let rs): for r in rs { log.insert("   \(r.ok ? "✓" : "✗") \(r.command.tool) [\(r.mechanism)]", at: 0) }
                case .clarify: log.insert("   ? needs clarification", at: 0)
                case .refuse: log.insert("   ⃠ declined (out of scope)", at: 0)
                case .none: log.insert("   ! error", at: 0)
                }
                busy = false; request = ""
            }
        }
    }

    private func resolve(_ allow: Bool) {
        let cont = continuation
        pending = nil; continuation = nil
        cont?.resume(returning: allow)
    }
}
