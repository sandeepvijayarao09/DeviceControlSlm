import AppIntents

/// Exposes a curated set of intents to Siri / Spotlight / the Shortcuts app.
/// (The full command set is invocable via the model + `DeviceControlEngine`;
/// these are the high-value voice/quick-action entries.)
public struct DeviceControlShortcuts: AppShortcutsProvider {
    public static var appShortcuts: [AppShortcut] {
        AppShortcut(intent: SetWifiIntent(),
                    phrases: ["Turn Wi-Fi on with \(.applicationName)"],
                    shortTitle: "Set Wi-Fi", systemImageName: "wifi")
        AppShortcut(intent: SetBrightnessIntent(),
                    phrases: ["Set brightness with \(.applicationName)"],
                    shortTitle: "Set Brightness", systemImageName: "sun.max")
        AppShortcut(intent: SetLowPowerModeIntent(),
                    phrases: ["Turn on Low Power Mode with \(.applicationName)"],
                    shortTitle: "Low Power Mode", systemImageName: "battery.25")
        AppShortcut(intent: SetFocusModeIntent(),
                    phrases: ["Set my Focus with \(.applicationName)"],
                    shortTitle: "Set Focus", systemImageName: "moon")
        AppShortcut(intent: StartTimerIntent(),
                    phrases: ["Start a timer with \(.applicationName)"],
                    shortTitle: "Start Timer", systemImageName: "timer")
        AppShortcut(intent: OpenAppIntent(),
                    phrases: ["Open an app with \(.applicationName)"],
                    shortTitle: "Open App", systemImageName: "app")
    }
}
