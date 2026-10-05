# RX 6800 / RDNA2 performance experiments

This is an experimental branch based on `6634719`, separate from
`codex/rdna2-krea-focused` and its existing pull request. All runtime changes are
opt-in and reachable only through the existing gfx103* model-load selection.
Defaults retain the PR's FP32 NF4 GEMMs and eight-head checkpointed attention.
The shared attention, NF4, SDPA, GPU backend, capabilities and family trainer
files are unchanged.

## Full-step measurements

Measured on an AMD Radeon RX 6800 with Windows ROCm,
`torch 2.12.0+rocm7.15.0a20260728`. The full RAW Krea 2 checkpoint was streamed
into NF4; all 28 main blocks and 264 LoRA modules were present. Each variant
started from identical adapter weights and a new AdamW8bit optimizer. Inputs
were fixed synthetic latents, noise, time and conditioning: 704x704, 64 valid
text tokens, rank 16 / alpha 16, batch 1, whole-block gradient checkpointing.
The driver's usual BF16 autocast was active. There was one warm-up update then
two timed updates per variant, synchronized around forward, backward and the
optimizer step. Finiteness checks were outside the timer.

| Variant | Median seconds/step | Peak allocated GiB | Finding |
| --- | ---: | ---: | --- |
| PR defaults | 24.71 | 10.52 | Baseline |
| Two heads per attention chunk | 22.30 | 10.52 | 9.7% less step time; recorded losses identical to baseline |
| Two heads + analytical attention backward | 22.19 | 10.52 | Only 0.5% beyond smaller chunks; rounding differs |
| Above + selected FP16 GEMMs and output tiling | 18.22 | 10.52 | 26.3% less time than baseline; finite gradients |
| Two heads + selected FP16 GEMMs/output tiling, standard SDPA | 18.09 | 10.52 | Simpler option; no analytical attention required |
| Two heads + analytical attention + additional reduction tiling | 17.36 | 10.39 | Best measured: 29.7% less time than baseline |
| MLP-only checkpointing + recomputed RMSNorm | 114.81 warm-up | See below | Rejected: Windows GPU paging |

Raw results are in [full_step_results.json](full_step_results.json).
Follow-up tiling and standard-SDPA measurements are recorded separately in
[additional_step_results.json](additional_step_results.json), from a fresh
process with the same model, seeds, adapter initialization and input settings.
The baseline was not repeated in that second process; two samples per variant
are enough to identify candidates, not characterize timing variance.

These are short, synthetic full-model training benchmarks. They exclude data
loading, EMA, saving and previews, and do not establish long-run convergence or
image quality. This is not a reproduction of the user's exact rank-8 dataset
run. The experiments neither resumed nor wrote the user's LoRA or state files.

## What improved and why

1. **Smaller attention groups.** At 2,000 tokens, the isolated forward/backward
   test improved from 129.31 ms for eight heads to 90.59 ms for two heads. A
   smaller score matrix gives this GPU/runtime a better working set. The full
   model also improved, so this is stronger evidence than a component result.
   This keeps the existing SDPA implementation and checkpoint strategy.

2. **Selected FP16 GEMMs.** BF16 weight and activation storage stays unchanged;
   only selected frozen NF4 matrix products convert to FP16 locally. Square
   6144-wide projections improved from 18.63 to 9.71 ms forward and 13.99 to
   8.55 ms for the input gradient. Blanket FP16 was unsuitable: some MLP
   shapes got slower. The opt-in policy selects only measured shapes and keeps
   FP32 elsewhere. It does not change trainable adapter storage or the optimizer.

3. **Shape-specific matrix tiling.** For the MLP expansion, splitting the output
   dimension into 4,096-column products reduced the component time from 39.18
   ms FP32 to 27.45 ms FP16. Further splitting the reduction dimension into
   4,096-element products, with an FP32 sum of partial outputs, improved MLP
   expansion input gradients from 35.36 to 28.09 ms and MLP-down forward from
   42.96 to 38.86 ms. Square projections were faster without reduction tiling.
   The combined reduction-tiled policy reached 17.36 seconds in full steps.
   Component improvements cannot simply be added to predict full-step time.

4. **Analytical attention backward.** This recomputes probabilities once and
   explicitly evaluates the exact attention derivatives rather than replaying
   an inner SDPA checkpoint through autograd. It retains FP32 arithmetic and
   BF16 input/output storage. It supports the existing Krea validity masks and
   expanded GQA, but has no dropout or causal interface. Its full-step gain
   over two-head SDPA was small. It is a research path, not a FlashAttention
   kernel or a recommendation to replace the established attention path.

The component results are in [gemm_results.json](gemm_results.json) and
[attention_results.json](attention_results.json). Matrix tests include NF4
dequantization. Attention component inputs were contiguous BHSD; full-model
tests used the model's real layouts and masking path.

## Rejected memory tradeoff

Replacing whole-block checkpointing with MLP-only checkpointing retained too
many attention activations. The first update took 114.81 seconds. Windows
reported 13.61 GiB dedicated and 6.83 GiB shared GPU memory; that process was
stopped after the warm-up rather than letting paging distort the timed results.
PyTorch allocated memory alone does not reveal all Windows paging.

The prototype is preserved in `rejected_checkpoint.py` for reproduction, but
is absent from the model-load driver and the benchmark's default variants.
It can only be invoked deliberately with `--variants mlp-only` in the benchmark.

## Accuracy and architecture isolation

Seven CPU checks cover attention outputs and derivatives with boolean/additive
masks, fully masked rows and nested checkpointing; tiling axes and dtype;
the default FP32 matmul; the rejected prototype's derivatives; and ignoring
all experiment flags on NVIDIA, gfx1100, gfx1200 and gfx90a. The NVIDIA check
also verifies that no GPU property query occurs.
The existing 12 private loader, architecture-isolation and preview-lifetime
regression checks also passed against this checkout, including byte equality
of shared GPU files against the PR's upstream base.

[accuracy_results.json](accuracy_results.json) contains 36 checks on the actual
RX 6800. NF4 products were checked in both directions at normal and small
input amplitudes. All results were finite; the largest relative L2 difference
from the FP32/BF16 reference was 0.324%. Analytical attention output and gradient
differences were below 0.009% on a noncontiguous layout with per-head masks.
These small checks cannot rule out FP16 overflow, underflow or training-quality
differences on other inputs. Full-step losses for FP16/analytical variants
differed from baseline; no claim of bitwise equivalence is made for those paths.

## Opt-in controls

Set these in the process that launches Fizgig from this experimental checkout.
Leave them unset to retain the PR's behavior. Other GPU architectures ignore
them because the model-load hook exits before installing RDNA2 forwards.

| Environment variable | Values | Default |
| --- | --- | --- |
| `FIZGIG_RDNA2_HEADS` | `1`, `2`, `4`, `8`, `12`, `16` | `8` |
| `FIZGIG_RDNA2_GEMM` | `tuned-fp16`, `tiled-fp16` | Existing FP32 path |
| `FIZGIG_RDNA2_ATTN_IMPL` | `analytical` | Existing checkpointed SDPA |

For the measured 17.36-second variant, set heads to `2`, GEMM to `tiled-fp16`,
and attention implementation to `analytical`. For the simpler 18.09-second
variant, set heads to `2`, GEMM to `tuned-fp16`, and leave attention
implementation unset. Smaller chunks alone are the
first candidate for real-dataset validation. FP16 and analytical paths require
quality comparisons before promotion to defaults or submission upstream.
All flags must be set before training; changing them between a forward and
its checkpointed backward would invalidate the computation.

## Reproduction

Use an idle GPU. The scripts require a free-memory check, but it cannot prevent
another GPU application starting during a benchmark. The `--runtime` directory
supplies the existing Windows ROCm environment; imports use this checkout's
`src`. No model weights, datasets or private configuration are published here.

```powershell
# From this experimental checkout; use a Python environment with its dependencies.
& ..\Fizgig\venv\Scripts\python.exe benchmarks\test_experiments.py
& ..\Fizgig\venv\Scripts\python.exe benchmarks\rdna2_components.py --runtime ..\Fizgig --kind gemm --output gemm.json
& ..\Fizgig\venv\Scripts\python.exe benchmarks\rdna2_components.py --runtime ..\Fizgig --kind attention --output attention.json
& ..\Fizgig\venv\Scripts\python.exe benchmarks\rdna2_accuracy.py --runtime ..\Fizgig --output accuracy.json
& ..\Fizgig\venv\Scripts\python.exe benchmarks\rdna2_full_step.py --runtime ..\Fizgig --dit 'PATH\TO\krea2_raw_bf16.safetensors' --output steps.json
```

## Could sub-10-second steps be possible?

There is a plausible compute route, but this branch has not achieved it.
AMD specifies 16.17 TFLOPS FP32 and 32.33 TFLOPS FP16 vector performance for
the [RX 6800](https://www.amd.com/en/products/graphics/desktops/radeon/6000-series/amd-radeon-rx-6800.html).
RDNA 3 introduced [WMMA matrix instructions](https://gpuopen.com/learn/wmma_on_rdna3/);
the RX 6800 needs kernels suited to its RDNA2 vector hardware.

An optimistic calculation from the main block dimensions gives about 12.16
billion frozen matrix weights. At 2,000 tokens, two forwards (including
checkpoint recomputation) and one input-gradient pass cost about 146 TFLOP.
That is roughly 9.0 seconds at peak FP32 or 4.5 seconds at peak FP16, before
attention, text fusion, LoRA derivatives, casts, memory traffic and launches.
Those are ideal compute lower bounds, not predicted training times.

The next substantial research targets are:

- **An RDNA2-specific fused attention forward/backward.** Avoid materializing
  the full per-chunk score/probability matrices, fuse softmax and reuse shared
  memory. The analytical prototype establishes derivative/mask checks, but
  still materializes those matrices and therefore does not provide the main
  benefit of a tiled fused implementation.
- **Checkpointing with a measured activation budget.** Save selected expensive
  outputs or checkpoint fewer blocks only while staying in dedicated memory.
  The failed MLP-only experiment shows why broadly removing recomputation is
  unsuitable on this 16 GB Windows setup. A budget must include previews and
  Windows memory counters, not only PyTorch's allocation counter.
- **Fuse casts, dequantization and matrix/MLP operations.** This could reduce
  repeated conversion and intermediate traffic. Dequantization alone measured
  only 0.18-1.46 ms per matrix here, so caching every BF16/FP32 weight would
  consume too much VRAM for a relatively small potential gain.

These are unimplemented research candidates, not promised speed-ups. Each
needs full-model timing, numerical checks and a real dataset quality run.
