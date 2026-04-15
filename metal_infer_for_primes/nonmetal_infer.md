nonmetal_infer is the Windows/native CPU backend.

Current scope:
- Reads manifest config from exported artifacts.
- Validates the canonical tensor contract used by infer.m.
- Computes packed-expert layout from manifest dimensions.
- Resolves embedding and lm_head tensor offsets from model_weights.json.
- Streams non-expert rows from model_weights.bin for:
	- embedding lookup by token id
	- lm_head argmax/top-5 over an embedded token
- Streams per-layer gate tensors from model_weights.bin for:
	- top-k expert routing from an embedded token
	- optional HDGL blend on top of gate softmax
- Executes the coupled one-layer MoE step for --route-token by:
	- running the routed packed experts selected by the gate
	- streaming shared expert gate/up/down tensors from model_weights.bin
	- applying the shared expert sigmoid gate and combining residual + routed + shared outputs
- Can carry the hidden state across multiple consecutive routed layers and optionally probe lm_head afterward.
- Executes a portable CPU 4-bit packed-expert slice for:
	- single-expert forward on one layer
	- simple K-expert MoE combination on one layer
- Uses plain C stdio and math only, with no Metal/Foundation dependency.

Current command surface:
- --check-only validates manifest, weights, and packed expert presence.
- --embed-token ID reads one embedding row from model_weights.bin.
- --lm-head-token ID embeds the token then streams lm_head for greedy next-token inspection.
- --route-token ID with --route-layer N runs gate routing and a coupled one-layer MoE step on CPU.
- --route-layers N extends --route-token across N consecutive layers starting at --route-layer.
- --route-lm-head runs a final streamed lm_head argmax/top-5 after the routed layer stack.
- --hdgl, --hdgl-alpha, and --hdgl-load enable optional HDGL routing blend for --route-token.
- --hdgl-semantic enables deterministic phi-octave semantic biasing of expert scores before top-k.
- --layer / --expert run a single expert from packed_experts/layer_XX.bin.
- --moe --k N runs a CPU MoE slice with deterministic routed experts.
- --benchmark N repeats the chosen CPU path and prints average timings.

Windows build helper:
- `metal_infer/build_nonmetal_windows.ps1` compiles `nonmetal_infer.exe` with clang and can run a smoke pass.
- `metal_infer/build_nonmetal_windows.bat` is a one-click wrapper for the PowerShell build script.
- Smoke mode always runs `--help`; it runs `--check-only` automatically when `model_weights.json`, `model_weights.bin`, and `packed_experts/layer_00.bin` are found under either the workspace root or `workspace_root/metal_infer`.

Windows HDGL preseed helper:
- `metal_infer/build_hdgl_lattice_windows.ps1` compiles `generate_hdgl_lattice.exe` and can run lattice generation.
- `metal_infer/build_hdgl_lattice_windows.bat` compiles and generates `metal_infer/hdgl_lattice.bin` in one step.
- The helper auto-resolves `model_weights.json` from workspace root first, then `workspace_root/metal_infer`.
- The generator is manifest-aware (`--manifest`) and derives `--instances` from `hidden_size` when `--instances` is not provided.

---

## Artifact Location Contract

All required files must live under a single **model root** directory.
`nonmetal_infer.exe` searches for the model root in this order:

1. The path passed via `--model PATH`
2. Workspace root (parent of `metal_infer/`)
3. `metal_infer/` itself

| File | Required | Description |
|------|----------|-------------|
| `model_weights.json` | Yes | Manifest — tensor offsets and model config |
| `model_weights.bin` | Yes | Non-expert weights (embeddings, norms, lm_head, attention, gate tensors) |
| `packed_experts/layer_NN.bin` | Yes (per layer) | 4-bit packed expert weights, one file per layer |
| `hdgl_lattice.bin` | No | Pre-seeded APA lattice (speeds up HDGL init) |

The build scripts accept the same search order via `Find-ModelRoot`.
Run `--check-only` to validate the artifact contract before any inference run.

---

## Windows A/B Parity Procedure

**Goal:** confirm that HDGL routing produces observable expert selection deltas
relative to pure MoE gating, with no crashes or NaN outputs at any alpha level.

### Quick start (auto-detect model)

```bat
cd metal_infer

REM One-time: build binaries
build_nonmetal_windows.bat
build_hdgl_lattice_windows.bat   REM generates hdgl_lattice.bin

REM Run A/B comparison (4 configs, logs side-by-side)
run_ab_parity.bat
```

### With explicit model path

```bat
run_ab_parity.bat -Model D:\models\qwen3-397b
```

### Benchmark mode (timing envelope)

```bat
run_ab_parity.bat -Model D:\models\qwen3-397b -Benchmark -BenchmarkN 5 -RouteLayers 10
```

### Manual equivalent commands

```bat
REM OFF (pure MoE)
nonmetal_infer.exe --model <root> --route-token 9707 --route-layers 5 --route-lm-head

REM HDGL alpha=0.20
nonmetal_infer.exe --model <root> --route-token 9707 --route-layers 5 --route-lm-head ^
    --hdgl --hdgl-alpha 0.20 --hdgl-load hdgl_lattice.bin

REM HDGL alpha=0.35
nonmetal_infer.exe --model <root> --route-token 9707 --route-layers 5 --route-lm-head ^
    --hdgl --hdgl-alpha 0.35 --hdgl-load hdgl_lattice.bin

REM Hybrid semantic preselection + HDGL
nonmetal_infer.exe --model <root> --route-token 9707 --route-layers 5 --route-lm-head ^
	--hdgl --hdgl-alpha 0.20 --hdgl-semantic --hdgl-load hdgl_lattice.bin
```

Logs are saved to `metal_infer/parity_logs/<timestamp>/OFF.log`, `HDGL020.log`, `HDGL035.log`,
`HDGL020SEM.log`, and a `run_manifest.json` recording the exact arguments and timestamps.

---

## Acceptance Thresholds (Finish-Line Gate)

A configuration is considered **passing** when all of the following hold:

| Gate | Criterion | Notes |
|------|-----------|-------|
| **Build** | `nonmetal_infer.exe --help` exits 0 | Binaries must exist and be runnable |
| **Artifact** | `--check-only` exits 0 | All required files present and manifest valid |
| **No crash** | All four A/B configs exit 0 | No segfault, assertion, or OOM on any config |
| **No NaN / Inf** | No `nan` or `inf` in lm_head top-5 logits | HDGL routing must not corrupt hidden state |
| **Routing delta** | OFF vs HDGL020 differ in ≥1 chosen expert in ≥1 of 5 layers | Confirms lattice signal is non-trivial |
| **Semantic delta** | OFF vs HDGL020SEM differ in ≥1 chosen expert in ≥1 of 5 layers | Confirms semantic preselection is active and observable |
| **Alpha ordering** | HDGL035 diverges ≥ HDGL020 from OFF | Higher alpha should produce stronger lattice influence |
| **Timing stability** | σ/mean < 15% across benchmark repeats per config | Confirms no pathological variance |
| **Stability** | Repeated identical runs produce identical expert selection | HDGL routing must be deterministic given same lattice seed |

HDGL is **expected to change** expert selection — exact OFF/HDGL match is not a pass criterion.
Exact parity to the macOS Metal path is **out of scope** for the Windows CPU backend.

---

## Next Scope

- Extend the non-Metal backend from one-layer execution to full token forward execution.
- Add more of the transformer path around the routed MoE stack so parity is not limited to the MLP portion.