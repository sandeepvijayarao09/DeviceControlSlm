# DeviceControlSlm — Architecture & Design

**Status:** Draft for review · **Scope:** on-device agentic SLM for iPhone / iPad / MacBook ·
**Runtime:** Apple MLX

---

## 1. Goal and non-goals

### Goal
Build a small language model *plus the runtime around it* that runs fully on-device and acts
as an **agent that controls the device**: it reads a natural-language request, decides on a
plan, calls tools, observes results, and either continues or returns a final answer — all
locally, privately, and offline-capable.

The differentiator is not raw model quality; it is **reliable tool-calling on a small model**
combined with **real, safe device actions**.

### Non-goals
- Not a general chatbot or a knowledge oracle. A 1B model is a poor encyclopedia; we do not
  optimize for open-domain Q&A.
- Not pretraining a model from scratch (wrong cost/benefit — see `decisions.md`, ADR-0001).
- Not cloud inference. Everything runs on the device. No prompt or user data leaves it.
- Not jailbreaking the sandbox. We only use capabilities Apple actually grants.

---

## 2. System overview

Five layers, from model up to UI. Everything below the app layer is portable Swift; the
tool adapters are the only place platform-specific code lives.

```
┌─────────────────────────────────────────────────────────────┐
│  UI layer — SwiftUI multiplatform app                         │
│  chat view · action-confirmation sheets · permission prompts  │
├─────────────────────────────────────────────────────────────┤
│  Agent orchestration — portable Swift ("AgentCore")           │
│  planner · tool-calling loop · memory · safety guardrails      │
├───────────────────────────────┬───────────────────────────────┤
│  Model layer (MLX)            │  Tool layer (protocol + registry)│
│  mlx-swift inference          │  Tool protocol, JSON schema,     │
│  tokenizer · sampler ·        │  dispatch, result formatting     │
│  constrained JSON decoding    │                                  │
│                               ├───────────────────────────────┤
│                               │  Platform tool adapters          │
│                               │  • In-app tools (all platforms)  │
│                               │  • App Intents / Shortcuts       │
│                               │  • macOS Accessibility / AppleEvts│
└───────────────────────────────┴───────────────────────────────┘
```

### Development vs on-device
We develop the agent logic and evaluate tool-calling in **Python with `mlx-lm`** because the
iterate/measure loop is 10× faster there, then port the validated core to **`mlx-swift`** for
the shipping app. Both use the *same* model weights and the *same* tool schemas, so behavior
transfers.

---

## 3. Model layer

### 3.1 Choosing the base model
We start from an existing open **instruct** model, quantized to 4-bit, and size it to the
device's memory budget. Candidates (all MLX-community quantized builds exist):

| Model | Params | 4-bit size (approx) | Notes |
| --- | --- | --- | --- |
| SmolLM2-1.7B-Instruct | 1.7B | ~1.0 GB | Tiny, fast; weaker tool-calling, needs LoRA. |
| Llama-3.2-1B-Instruct | 1B | ~0.7 GB | Good for iPhone; decent instruction following. |
| Llama-3.2-3B-Instruct | 3B | ~1.8 GB | Strong for iPad/Mac; native tool-calling. |
| **Qwen2.5-1.5B-Instruct** | 1.5B | ~1.0 GB | **Default for phone** — strong function-calling for its size. |
| **Qwen2.5-3B-Instruct** | 3B | ~1.9 GB | **Default for iPad/Mac** — best tool reliability at this size. |
| Phi-3.5-mini-instruct | 3.8B | ~2.2 GB | Capable; heavier, watch memory on phones. |

**Default choice:** Qwen2.5 family — its native tool-calling format and instruction
following are the strongest we've seen in the 1–3B range. We keep the model behind an
abstraction so swapping is a config change, not a rewrite.

### 3.2 Sizing per device
RAM is the binding constraint. iOS gives an app only a fraction of physical RAM before the
jetsam killer steps in.

| Device class | Physical RAM | Practical model | Quant |
| --- | --- | --- | --- |
| iPhone (A15–A17, 6–8 GB) | 6–8 GB | 0.5B–1.5B | 4-bit |
| iPad (M-series, 8–16 GB) | 8–16 GB | up to 3B | 4-bit |
| MacBook (M-series, 16 GB+) | 16 GB+ | 3B–8B | 4-bit / 8-bit |

The app detects device capability at launch and picks a model tier; it never loads a model
that won't fit the app's jetsam budget.

### 3.3 Inference
- **On-device:** `mlx-swift` + `mlx-swift-examples` LLM runner as the starting point. KV-cache
  reuse across turns; streaming token output to the UI.
- **Quantization:** 4-bit weights (group size 64) is the default; 8-bit only on Macs with
  headroom. Done via `mlx_lm.convert`.
- **Constrained decoding:** the single most important reliability lever (see §5.3).

---

## 4. Agent orchestration (`AgentCore`)

Portable Swift, no platform imports. Also mirrored in Python for development.

### 4.1 The loop (ReAct-style, tool-calling)
```
1. Build prompt = system + tool schemas + memory + user turn
2. Model generates → constrained to either a tool_call(JSON) or a final answer
3. If tool_call:
     a. Validate against the tool's JSON schema
     b. Safety gate: if the tool is "sensitive", pause for user confirmation
     c. Execute tool → observation
     d. Append observation to context, go to 2
4. If final answer: return to user, stop
```
A hard **max-steps** cap and a **wall-clock budget** prevent runaway loops on a battery
device. Every step is logged locally for debugging and for building the eval set.

### 4.2 Memory
- **Short-term:** the current task's message/observation history, trimmed to fit context.
- **Working scratchpad:** the model's plan/notes, kept separate from tool output.
- **Long-term (later):** an on-device store of user preferences and past actions (opt-in),
  retrieved by simple keyword/embedding match. Deferred to Phase 5.

### 4.3 Planning
For a 1–3B model, explicit multi-step planning is unreliable. We keep it minimal: a single
"think then act" step per turn, few tools in scope at once, and decomposition handled by the
loop rather than by asking the model to emit a whole plan up front.

---

## 5. Tool layer

### 5.1 The tool protocol
Every capability is a `Tool` with: a stable `name`, a human description, a **JSON Schema** for
its arguments, a `sensitivity` level, and an `invoke(args) -> ToolResult`. The registry
exposes the schemas to the model and dispatches calls. This is the contract that lets us add
capabilities without touching the agent loop.

```
protocol Tool {
    var name: String { get }                 // "create_reminder"
    var description: String { get }           // shown to the model
    var parameters: JSONSchema { get }         // argument schema
    var sensitivity: Sensitivity { get }       // .safe / .confirm / .blocked
    func invoke(_ args: JSONValue) async throws -> ToolResult
}
```

### 5.2 Tool-calling format
Use the model's **native** tool-call format where it has one (Qwen2.5, Llama-3.2 do). The
runtime parses the emitted call, not free-form prose. We normalize all models to one internal
representation so `AgentCore` is model-agnostic.

### 5.3 Reliability: constrained / grammar-guided decoding
Small models emit malformed JSON often enough to break a naive loop. Mitigations, in order:
1. **Constrained decoding** — restrict the sampler to tokens that keep the output valid
   against the active tool schema (grammar/regex-guided). This makes malformed calls
   structurally impossible, not just less likely.
2. **Schema validation + one repair retry** — if a call still fails validation, feed the
   error back once and let the model correct it.
3. **Few-shot exemplars** in the system prompt for the tools currently in scope.
4. **Keep tool count small per request** — scope tools to the task rather than exposing all.

### 5.4 Platform tool adapters
The same protocol, three families of implementations:

#### A. In-app tools — *all platforms, no entitlements*
Actions inside our own app: query app data, navigate screens, run computations, manage the
agent's own notes/reminders store. Always available, always safe to expose. This is where the
first working demo lives.

#### B. OS actions via App Intents / Shortcuts — *iOS, iPadOS, macOS*
Apple's sanctioned bridge from an app to system-level actions.
- **App Intents (iOS 16+ / macOS 13+):** we define intents; we can also *invoke* system and
  third-party intents that vendors expose. Enables: set alarms/timers, create calendar events
  and reminders (EventKit), compose messages/mail, open apps and deep links, control
  HomeKit scenes, start Maps navigation, toggle Focus, run user-authored Shortcuts.
- **What it can't do:** arbitrarily drive another app's UI, or do anything Apple hasn't
  surfaced as an intent/framework API. That's a hard sandbox boundary on iOS.
- Each capability is wrapped as a `Tool`; permission-gated ones (Contacts, Calendar,
  Photos, Health, Location) request authorization on first use.

#### C. macOS Accessibility & Apple Events — *macOS only*
The Mac is far less sandboxed and is where "control the device like a human would" is real.
- **Accessibility API (`AXUIElement`):** inspect the UI tree of other apps, read labels,
  click buttons, set field values, read window contents. Requires the user to grant
  **Accessibility** permission in System Settings.
- **Apple Events / AppleScript / JXA:** script scriptable apps (Finder, Mail, Safari, etc.).
  Requires **Automation** permission per target app.
- **CGEvent:** synthesize keyboard/mouse input as a last resort when no AX action exists.
- These are powerful and therefore all classified `.confirm` or higher, and are only
  compiled into the macOS target.

### 5.5 Capability matrix

| Capability | iPhone / iPad | macOS |
| --- | --- | --- |
| In-app tools | ✅ | ✅ |
| Reminders / Calendar (EventKit) | ✅ (permission) | ✅ (permission) |
| Messages / Mail compose | ✅ (user sends) | ✅ |
| Open app / deep link | ✅ | ✅ |
| Run user Shortcuts | ✅ | ✅ |
| HomeKit / Maps / Focus | ✅ (permission) | ✅ |
| Drive another app's UI | ❌ (sandbox) | ✅ (Accessibility) |
| Script other apps | ❌ | ✅ (Apple Events) |
| Synthesize keyboard/mouse | ❌ | ✅ (CGEvent) |

---

## 6. Safety, permissions, and privacy

- **Confirmation gate.** Every `.confirm` tool call surfaces a sheet showing exactly what
  will happen (tool, arguments, target) before it runs. Destructive/irreversible actions are
  never auto-executed. `.blocked` tools are compiled out or hard-refused.
- **Least privilege.** Only the tools relevant to the current task are exposed to the model.
- **OS permissions.** We request Apple's authorization (Calendar, Contacts, Accessibility,
  Automation, …) lazily, with clear purpose strings, and degrade gracefully if denied.
- **Privacy by construction.** Inference is 100% on-device. No network calls for the model.
  Logs and memory are local, and any long-term memory is opt-in and user-clearable.
- **Loop safety.** Max-step and time budgets; a visible "stop" control; no background action
  execution without foreground consent.

---

## 7. Repository shape (proposed)

```
DeviceControlSlm/
├── docs/                     # design, roadmap, decisions (this set)
├── python/                   # dev + eval harness (mlx-lm)
│   ├── agent/                # portable agent loop, tool schemas, prompts
│   ├── tools/                # mock/host tools for iteration
│   └── eval/                 # tool-calling eval suite + datasets
├── swift/
│   ├── AgentCore/            # SwiftPM package: portable agent + tool protocol
│   ├── MLXRuntime/           # mlx-swift inference wrapper
│   └── Tools/
│       ├── InApp/            # in-app tools (all platforms)
│       ├── AppIntentsBridge/ # iOS/iPadOS/macOS system actions
│       └── MacAccessibility/ # macOS-only control
└── app/                      # multiplatform SwiftUI app (iOS/iPadOS/macOS target)
```
Rationale: the Python and Swift agents share prompt/schema files (checked in as data) so the
eval work in Python directly hardens the shipping Swift agent.

---

## 8. Evaluation

Reliability is the whole game, so we measure it from Phase 1:
- **Tool-call validity rate** — % of emitted calls that parse and pass schema validation.
- **Task success rate** — on a fixed suite of scripted tasks with mock tools.
- **Grounding** — did it call the right tool with the right args, vs. hallucinate?
- **Latency / throughput** — tokens/sec and end-to-end task time per device tier.
- **Memory** — peak RSS vs. the device's jetsam budget.
The eval suite is version-controlled datasets + a runner in `python/eval/`, run on every
prompt/model change.

---

## 9. Key risks and mitigations

| Risk | Mitigation |
| --- | --- |
| Small model produces unreliable tool calls | Constrained decoding + validation/repair + optional LoRA fine-tune (ADR-0004). |
| iPhone RAM/jetsam kills the model | Device-tiered model selection; 4-bit; cap context; measure peak RSS. |
| Apple sandbox blocks desired control on iOS | Design around App Intents/frameworks; put "drive any app" only on macOS. |
| App Store review of Accessibility/Automation use | macOS-only, user-granted, clearly disclosed; consider notarized direct distribution. |
| Battery / thermals during long loops | Step + time budgets; stream and stop early; prefer smallest capable model. |
| Prompt injection via tool output / app content | Treat tool results as untrusted; keep the safety gate between the model and any action. |

---

## 10. Open questions for review
1. Primary device to demo first — iPhone, iPad, or Mac? (Affects model tier and which tool
   adapter we build first.)
2. Which concrete tools matter most to you (reminders? messages? Mac app control?) — that
   sets the Phase 3 order.
3. Are you open to a later LoRA fine-tune to boost tool reliability, or do we stay
   prompt-only?
4. Distribution target: App Store, TestFlight, or direct/notarized (matters most on macOS)?
