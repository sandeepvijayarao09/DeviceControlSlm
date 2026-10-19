#if os(macOS)
import Foundation

/// Real macOS controller. Uses system tools (osascript / networksetup / open /
/// pbcopy / screencapture / pmset) via `Process` — robust and dependency-free.
/// Commands not implemented here fall through to the generated `unsupported`
/// defaults, which makes the recipe engine try Accessibility next.
public final class MacDeviceController: DeviceControlling {
    public init() {}

    @discardableResult
    private func run(_ launchPath: String, _ args: [String], stdin: String? = nil) throws -> String {
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: launchPath)
        proc.arguments = args
        let out = Pipe()
        proc.standardOutput = out
        proc.standardError = out
        if let stdin { let inPipe = Pipe(); proc.standardInput = inPipe
            try proc.run()
            inPipe.fileHandleForWriting.write(stdin.data(using: .utf8) ?? Data())
            inPipe.fileHandleForWriting.closeFile()
        } else {
            try proc.run()
        }
        proc.waitUntilExit()
        let data = out.fileHandleForReading.readDataToEndOfFile()
        return String(data: data, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
    }

    @discardableResult private func osa(_ script: String) throws -> String {
        try run("/usr/bin/osascript", ["-e", script])
    }

    // Connectivity
    public func setWifi(enabled: Bool) async throws {
        _ = try run("/usr/sbin/networksetup", ["-setairportpower", "en0", enabled ? "on" : "off"])
    }
    public func getConnectivityStatus() async throws -> String {
        try run("/usr/sbin/networksetup", ["-getairportpower", "en0"])
    }

    // Audio
    public func setVolume(level: Int) async throws {
        try osa("set volume output volume \(max(0, min(100, level)))")
    }
    public func setMute(enabled: Bool) async throws {
        try osa("set volume output muted \(enabled ? "true" : "false")")
    }

    // Display / appearance
    public func setDarkMode(enabled: Bool) async throws {
        try osa("tell application \"System Events\" to tell appearance preferences "
                + "to set dark mode to \(enabled ? "true" : "false")")
    }
    public func setWallpaper(name: String) async throws {
        try osa("tell application \"System Events\" to set picture of every desktop to \"\(name)\"")
    }
    public func takeScreenshot() async throws {
        let path = NSTemporaryDirectory() + "dcslm-\(Int(Date().timeIntervalSince1970)).png"
        _ = try run("/usr/sbin/screencapture", ["-x", path])
    }

    // Apps
    public func openApp(name: String) async throws {
        _ = try run("/usr/bin/open", ["-a", name])
    }
    public func closeApp(name: String) async throws {
        try osa("tell application \"\(name)\" to quit")
    }

    // Media (Music/Spotify via AppleScript)
    public func mediaControl(action: MediaAction) async throws {
        let verb: String
        switch action {
        case .play: verb = "play"; case .pause: verb = "pause"
        case .next: verb = "next track"; case .previous: verb = "previous track"
        case .stop: verb = "pause"
        }
        try osa("tell application \"Music\" to \(verb)")
    }
    public func playMusic(query: String?) async throws {
        try osa("tell application \"Music\" to play")
    }

    // Power (gated by confirmation upstream)
    public func restartDevice() async throws {
        try osa("tell application \"System Events\" to restart")
    }
    public func shutdownDevice() async throws {
        try osa("tell application \"System Events\" to shut down")
    }

    // Utilities
    public func setClipboard(text: String) async throws {
        _ = try run("/usr/bin/pbcopy", [], stdin: text)
    }

    // Queries
    public func getBattery() async throws -> String {
        try run("/usr/bin/pmset", ["-g", "batt"])
    }
    public func getStorage() async throws -> String {
        try run("/bin/df", ["-h", "/"])
    }
}
#endif
