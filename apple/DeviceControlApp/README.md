# DeviceControlApp — minimal iOS / iPadOS / macOS app (P7)

A small SwiftUI app that demonstrates the full on-device pipeline end to end:

```
type a request → SLM (LanguageModel) → DSL → DeviceAgent
  → DeviceControlEngine (parse → confirmation gate → per-OS recipe) → device action
```

## Files
- `DeviceControlApp.swift` — `@main` app; selects the model (Core ML SLM if bundled,
  else a keyword `HeuristicLanguageModel` so it runs before the model is exported).
- `ContentView.swift` — request field, results log, and the **confirmation alert** that the
  safety gate drives (sensitive commands suspend until you Allow/Cancel).
- `CoreMLLanguageModel.swift` — loads `DCSLM.mlmodelc` + `dcslm_tokenizer.json` and greedy-decodes
  the DSL; conforms to `DeviceControlKit.LanguageModel`.

## Build (Xcode, on a Mac)
1. Create an app target and add the local package `apple/DeviceControlKit` as a dependency
   (File ▸ Add Package Dependencies ▸ Add Local…).
2. Add these three source files to the target.
3. (Optional, for the real SLM) run `python tools/export_coreml.py`, drag `DCSLM.mlpackage`
   (compiles to `.mlmodelc`) and `dcslm_tokenizer.json` into the target's Resources.
4. Add usage-permission keys as needed (e.g. `NSCameraUsageDescription` for the torch,
   Contacts if you resolve names) and enable App Intents.
5. Run on iOS/iPadOS Simulator, a device, or "My Mac".

Without the bundled model it uses the heuristic fallback — enough to see the gate, the recipe,
and the per-OS execution wiring work. With the model it runs the real from-scratch SLM on the
Neural Engine.

## Note
This app target is built in Xcode (not by the package CI job). The library it depends on
(`DeviceControlKit`) is fully CI-tested.
