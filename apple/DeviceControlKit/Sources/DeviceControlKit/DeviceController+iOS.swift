#if os(iOS)
import UIKit
import AVFoundation
import MediaPlayer

/// Real iOS/iPadOS controller. Uses genuine public APIs for the commands Apple
/// permits a third-party app to perform directly. Radios/system modes with no
/// public API (Wi-Fi, Focus, Low Power, …) are intentionally NOT here — they
/// fall through to the `unsupported` defaults so the recipe engine routes them
/// to a Shortcuts system action instead (see Recipe.swift).
public final class IOSDeviceController: DeviceControlling {
    public init() {}

    // Display
    public func setBrightness(level: Int) async throws {
        await MainActor.run { UIScreen.main.brightness = CGFloat(max(0, min(100, level))) / 100.0 }
    }

    // Utilities
    public func setClipboard(text: String) async throws {
        await MainActor.run { UIPasteboard.general.string = text }
    }
    public func setFlashlight(enabled: Bool) async throws {
        guard let device = AVCaptureDevice.default(for: .video), device.hasTorch else {
            throw DeviceControlError.unsupportedOnPlatform(command: "set_flashlight", platform: "iOS")
        }
        try device.lockForConfiguration()
        try? device.setTorchModeOn(level: 1.0)
        device.torchMode = enabled ? .on : .off
        device.unlockForConfiguration()
    }

    // Apps (deep links / URL schemes)
    private static let appURLs = [
        "spotify": "spotify://", "safari": "https://apple.com", "maps": "maps://",
        "messages": "sms:", "mail": "mailto:", "music": "music://", "camera": "camera://",
    ]
    public func openApp(name: String) async throws {
        let key = name.lowercased()
        guard let scheme = Self.appURLs[key], let url = URL(string: scheme) else {
            throw DeviceControlError.unsupportedOnPlatform(command: "open_app:\(name)", platform: "iOS")
        }
        await MainActor.run { UIApplication.shared.open(url) }
    }

    // Comms (system UI / URL — user still confirms in the system sheet)
    public func startCall(to: String) async throws {
        let digits = to.filter { $0.isNumber }
        guard let url = URL(string: "tel://\(digits)") else {
            throw DeviceControlError.invalidArgument("phone: \(to)")
        }
        await MainActor.run { UIApplication.shared.open(url) }
    }

    // Media
    public func mediaControl(action: MediaAction) async throws {
        let player = MPMusicPlayerController.systemMusicPlayer
        switch action {
        case .play: player.play()
        case .pause, .stop: player.pause()
        case .next: player.skipToNextItem()
        case .previous: player.skipToPreviousItem()
        }
    }
    public func playMusic(query: String?) async throws {
        MPMusicPlayerController.systemMusicPlayer.play()
    }

    // Queries
    public func getBattery() async throws -> String {
        await MainActor.run {
            UIDevice.current.isBatteryMonitoringEnabled = true
            let pct = Int((UIDevice.current.batteryLevel * 100).rounded())
            return pct >= 0 ? "\(pct)%" : "unknown"
        }
    }
    public func getStorage() async throws -> String {
        let url = URL(fileURLWithPath: NSHomeDirectory())
        let values = try url.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey])
        let bytes = values.volumeAvailableCapacityForImportantUsage ?? 0
        return "\(bytes / 1_000_000_000) GB free"
    }
}
#endif
