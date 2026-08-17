# DeviceControlKit — on-device execution via App Intents + Shortcuts + Accessibility

The Swift package that turns the SLM's command output into **real device actions** on
iOS / iPadOS / macOS. It is layer 2–3 of the DCSLM architecture (`docs/paper/dcslm-architecture.md`).

## The recipe (how it actually succeeds)
Apple blocks many system toggles from third-party apps, so no single mechanism works for
everything. The engine walks a **per-command, per-OS fallback recipe** (`Recipe.swift`) and uses
the first mechanism that works:

1. **Direct framework API** — `UIScreen.brightness`, EventKit, `UIPasteboard`, `AVCaptureDevice`
   torch, `tel:`/URL opens, etc.
2. **Shortcuts / App Intents system actions** — the iOS unlock. Radios and system modes with no
   public API (**Set Wi-Fi, Set Bluetooth, Set Airplane Mode, Set Low Power Mode, Set Focus, Set
   Volume, Set Appearance, Start Timer**) are invoked through system-provided Shortcuts actions.
3. **Accessibility (AXUIElement)** — mainly macOS: open the relevant System Settings pane and
   toggle the control when nothing else exists.
4. **unsupported** — decline gracefully (the model is trained to, and the UX grays it out).

Example: `set_wifi` → iOS `[.shortcut("Set Wi-Fi")]`, macOS `[.directAPI, .accessibility]`.
`set_brightness` → iOS `[.directAPI, .shortcut("Set Brightness")]`.

## How the model connects
```
SLM output DSL ─► DeviceControlEngine.parse ─► for each command:
   confirmation gate (sensitive?) ─► Recipe.plan(cmd, platform) ─► try mechanisms in order
```
`DeviceControlEngine.execute(dsl:platform:confirm:)` mirrors the Python agent loop that
DeviceControlBench scored, so on-device behavior matches the benchmarked behavior. The
confirmation gate (`Recipe.requiresConfirmation`) is non-bypassable — the guarantee behind UER=0.

## App Intents surface
Every command is also a first-class **`AppIntent`** (`Generated/DeviceIntents.swift`), so the same
actions are available to **Siri, Spotlight, and the Shortcuts app** independently of the model.
`AppShortcuts.swift` exposes curated voice phrases. Enum parameters (Focus/Ringer/Media) are
generated as `AppEnum`s.

## Single source of truth
`Generated/*.swift` is produced from `bench/tools/catalog.json` by `tools/gen_appintents.py`
(run from the repo root). The **same catalog** drives the model's command set and the benchmark,
so the model, the eval, and the app can never drift. Do not edit generated files by hand.

```
python tools/gen_appintents.py        # regenerate intents + protocol + router + stub
swift test --package-path apple/DeviceControlKit   # parsing/recipe/gate tests (no device needed)
```

## What ships vs. what's scaffolded
- **Complete + compiling:** the App Intents, the recipe, the executor engine, the confirmation
  gate, the DSL parser, and a `StubDeviceController` (logs) so it builds and runs in Simulator/
  Previews and under `swift test`.
- **To fill in on device:** the real bodies of `DirectAPIExecutor` targets (per-OS
  `DeviceControlling` conformances), `ShortcutsExecutor` (invoke the named system action), and
  `AccessibilityExecutor` (AX automation). Each is marked `TODO(device)`.
