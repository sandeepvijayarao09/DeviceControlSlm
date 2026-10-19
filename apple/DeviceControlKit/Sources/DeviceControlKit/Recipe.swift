import Foundation
#if canImport(UIKit)
import UIKit
#endif

/// A concrete way to carry out a command. The engine tries these in order until
/// one succeeds — this ordered fallback IS "the recipe".
public enum Mechanism: Equatable {
    case directAPI                 // public framework API (UIScreen, EventKit, UIPasteboard, …)
    case shortcut(String)          // a system Shortcuts action / App Intent (the iOS unlock)
    case accessibility             // drive Settings UI via AXUIElement (mostly macOS)
    case unsupported               // no legal path on this OS → decline gracefully
}

public enum DCPlatform: String {
    case iOS, iPadOS, macOS
    public static var current: DCPlatform {
        #if os(macOS)
        return .macOS
        #elseif canImport(UIKit)
        return UIDevice.current.userInterfaceIdiom == .pad ? .iPadOS : .iOS
        #else
        return .iOS
        #endif
    }
    var isMac: Bool { self == .macOS }
}

/// Per-command execution recipe. Encodes, honestly and per-OS, how each command
/// can actually be carried out today. This is the source of truth for both the
/// runtime (which mechanism to attempt) and the UX (what to gray out / deflect).
public enum Recipe {
    /// Commands that change something irreversible/outbound → confirm before running.
    public static let requiresConfirmation: Set<String> = [
        "restart_device", "shutdown_device", "send_message", "compose_mail", "start_call",
    ]

    // iOS/iPadOS: system toggles with no direct API but a Shortcuts action we can invoke.
    private static let iosShortcut: [String: String] = [
        "set_wifi": "Set Wi-Fi", "set_bluetooth": "Set Bluetooth",
        "set_airplane_mode": "Set Airplane Mode", "set_low_power_mode": "Set Low Power Mode",
        "set_do_not_disturb": "Set Focus", "set_focus_mode": "Set Focus",
        "set_volume": "Set Volume", "set_dark_mode": "Set Appearance",
        "start_timer": "Start Timer", "set_alarm": "Create Alarm", "cancel_timers": "Cancel Timer",
    ]
    // iOS/iPadOS: commands with a genuine public API.
    private static let iosDirect: Set<String> = [
        "set_brightness", "open_app", "create_reminder", "list_reminders",
        "create_calendar_event", "list_calendar_events", "send_message", "compose_mail",
        "start_call", "media_control", "play_music", "set_clipboard", "set_flashlight",
        "get_battery", "get_storage", "get_connectivity_status",
    ]
    // macOS: no meaningful path (hardware/absent concepts).
    private static let macUnsupported: Set<String> = [
        "set_flashlight", "set_orientation_lock", "set_ringer_mode",
        "set_cellular_data", "set_hotspot",
    ]

    public static func plan(_ command: String, on platform: DCPlatform) -> [Mechanism] {
        if platform.isMac {
            if macUnsupported.contains(command) { return [.unsupported] }
            // On macOS almost everything is reachable via a framework/script, with
            // Accessibility as the universal fallback for Settings toggles.
            return [.directAPI, .accessibility]
        }
        // iOS / iPadOS
        var chain: [Mechanism] = []
        if iosDirect.contains(command) { chain.append(.directAPI) }
        if let action = iosShortcut[command] { chain.append(.shortcut(action)) }
        return chain.isEmpty ? [.unsupported] : chain
    }
}
