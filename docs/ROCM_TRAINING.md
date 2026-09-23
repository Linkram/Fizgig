# ROCm training and performance

Start Fizgig with `run_fizgig_rocm.bat`. Krea 2 uses hardware detection to select its training strategy when Base Precision and Blocks Swap are set to Auto. On RDNA2 GPUs (ROCm architecture identifiers `gfx103*`), Auto prefers NF4 without block swapping when the estimated memory requirement fits the available VRAM. The estimate accounts for resolution, batch size, and adapter rank.

The optimized NF4 operations and streamed loading are selected inside the trainer, including CLI training. Explicit compatible precision settings remain available. INT8 requires zero block swap; incompatible INT8-plus-swap settings produce a validation error instead of falling back to FP8.

Auto precision is resolved even if a manual block swap count was saved. If it selects NF4 or INT8, training uses zero swap and logs the adjustment. If it selects FP8, the manual swap count is retained. Selecting an explicit precision bypasses Auto precision selection.

The implementation targets the RDNA2 architecture family. Measurements below cover one GPU and software stack; performance and stability on other RDNA2 devices require validation. Other architectures retain their existing matrix and loading paths.

## Implementation

- Krea 2's PyTorch attention path processes long sequences on RDNA2 in groups of eight heads. During training, each group is checkpointed so backward reconstructs one group's attention scores at a time. All heads and tokens are retained. This bounds the math backend's workspace without reducing image resolution or changing base precision. Short sequences, dropout, other devices, and other attention backends retain their existing paths. `FIZGIG_RDNA2_ATTENTION=0` disables grouping for comparison.

- Automatically on ROCm gfx103* GPUs, frozen NF4 linear layers use FP32 matrix multiplication and return BF16 activations. This avoids a slow BF16 GEMM path measured on the RX 6800 with torch 2.12.0+rocm7.15.0a20260728. LoRA weights and gradients retain the existing training dtype. Autocast is disabled only around these GEMMs. There is no persistent FP32 model copy.
- The NF4 backward pass reconstructs each frozen weight from its packed representation instead of retaining a dequantized weight for every layer. This limits memory retention, including with gradient checkpointing.
- Automatically on RDNA2, Krea 2 NF4 loading starts on the meta device and reads, transfers, and quantizes one checkpoint tensor at a time. The standard loader first stages the entire approximately 26 GB BF16 checkpoint in host RAM. Streaming is particularly useful on a 32 GB host. It validates checkpoint keys, shapes, and dtypes before GPU allocations. It currently requires a plain Krea 2 checkpoint, not a prequantized FP8 file.
- Auto planning recognizes the detected RDNA2 path. It does not assume that an available INT8 kernel is the fastest choice merely because that choice performed well on NVIDIA.
- The upstream v6.3.0 VRAM monitor avoids initializing HIP in the GUI. Preview cleanup restores model placement. The existing ROCm launcher supplies the library paths; this change does not override them.
- Windows ROCm Krea 2 startup validates Auto without resolving it on Tk's thread. Device metadata is queried by a short-lived helper process, with kernel probes disabled for Auto/NF4/FP8. The GUI polls completion without waiting synchronously, supports cancelling the pending launch, and reuses the memory snapshot for planning. Explicit INT8 selection can request its capability probe in the helper. The invalid INT8 `_scaled_mm` probe has been removed entirely.

For debugging, set `FIZGIG_RDNA2_LINEAR=0` before calling any launcher to restore the original math while retaining streamed loading. Set `FIZGIG_STREAM_NF4=0` to restore the old loader too. Explicit environment overrides are respected in GUI and CLI training.

The optional `scaled_fp16` matrix path is experimental and was slower than FP32 in the realistic block benchmark. It is retained to reproduce that comparison, not selected automatically.

## Measurements and interpretation

Attention memory validation: at 48 heads, 2,048 tokens and head dimension 128, checkpointed BF16 attention peaked at 3.320 GiB with the original path versus 0.773 GiB with eight-head groups. Warm forward/backward took 0.163 versus 0.113 seconds. All three input gradients were identical in this test. On one real NF4 Krea 2 block with FP32 frozen GEMMs, grouping reduced peak allocated memory from 3.677 to 1.685 GiB; median forward/backward time was 0.545 versus 0.568 seconds (about 4% slower in isolation), with identical output and input gradient. The expected benefit at model scale is reduced memory pressure; isolated attention speed does not establish full-model speed. Reproduce with `benchmarks/probe_attention_backward.py` and `benchmarks/probe_krea_block.py --dit PATH_TO_KREA_BF16 --compare-attention`; their generated JSON output is intentionally excluded from the repository.

Full-model attention comparison, with FP32 frozen GEMMs in both variants: **37.900 → 18.831 seconds (2.01x)** per timed step, including AdamW8bit updates and gradient clipping. Peak allocated memory fell from **12.152 → 10.251 GiB**. Settings: 704x704 synthetic latents, 64 valid text tokens, batch 1, rank 16, LR 0.0004, gradient checkpointing. Each variant ran one warm-up and one timed step; adapters and optimizer state were reset between variants. Both recorded losses matched exactly and all four steps had finite gradients. This short synthetic comparison excludes previews and EMA and does not establish dataset throughput or long-run quality/stability. Reproduce with `benchmarks/probe_krea_full.py --dit PATH_TO_KREA_BF16 --compare-attention --steps 2 --optimizer --output benchmarks/krea_full_attention.json`; the generated result file is intentionally excluded from the repository.

Measured locally on an RX 6800 (gfx1030), Windows, Python 3.12.10, torch 2.12.0+rocm7.15.0a20260728, bitsandbytes 0.50.2.dev0. The benchmark scripts use the existing environment; no driver or dependency upgrade was performed.

Real Krea 2 block weights, NF4, 2,048 tokens, checkpoint recomputation plus backward, median of three timed iterations after warmup:

| Frozen matrix path | Seconds | Peak allocated VRAM | Relative L2 output / input-gradient difference from baseline |
| --- | ---: | ---: | ---: |
| Original BF16 | 1.3504 | 3.853 GiB | reference |
| FP32 | 0.5933 | 3.677 GiB | 0.131% / 0.126% |
| Range-scaled FP16 | 0.6954 | 3.677 GiB | 0.303% / 0.368% |

The FP32 block speedup is 2.28x. It is a block-level measurement, not a claim of identical full training speedup. Outputs are not bit-identical: changing GEMM precision changes rounding. Short finite-gradient checks do not establish long-run LoRA quality or driver stability.

The completed full-model forward/backward comparison measured **80.141 seconds → 36.875 seconds (2.17x)**, median of two timed passes per mode after one warmup. Peak allocated VRAM was **12.217 → 12.041 GiB**. All six passes had finite LoRA gradients. This is a controlled synthetic-input comparison, not a guarantee of the same speedup on a real dataset. Optimizer updates were excluded from this recorded run.

The full-model benchmark is `benchmarks/probe_krea_full.py`. New runs default to `benchmarks/krea_full.json` or the path supplied with `--output`; generated results are local artifacts and should not be committed. The benchmark uses the real model and rank-16 LoRA network with synthetic latents/text features, approximately 0.5 MP (704x704), batch 1, 64 text tokens, no previews or file saving. The original recorded run excludes optimizer updates. The script offers `--optimizer` to include AdamW8bit, gradient clipping, and LR 0.0004 updates; adapters and optimizer state are reset between variants. Neither is a dataset quality evaluation.

Run from the repository with the installed ROCm environment:

```powershell
.\venv\Scripts\python.exe benchmarks\probe_krea_full.py --dit "YOUR_KREA2_RAW_BF16.safetensors" --resolution 704 --rank 16 --steps 6 --optimizer
```

The script loads the model once and compares original vs optimized math. Its first iteration per variant is warmup. It writes measured timings, loss, configuration, and peak allocated memory. It does not save or overwrite model weights. Use an idle GPU for a clean comparison; ComfyUI or other GPU work can contaminate timings.

## Comparing GPUs

Compare identical resolution, caption length, batch size, rank, precision, swapping, checkpointing, and optimizer settings after warmup. Doubling image tokens increases linear-layer work roughly proportionally and the image-image attention term roughly quadratically. Include previews and checkpoint saving consistently when measuring end-to-end throughput.

PyTorch documents that its math SDPA backend normally upcasts FP16/BF16 intermediates to FP32. Enabling reduced-precision BF16 math attention in the local microbenchmark was approximately four times slower, so the application does not enable that flag. See [PyTorch numerical accuracy](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html#reduced-precision-reduction-for-fp16-and-bf16-in-scaled-dot-product-attention-sdpa) and [SDPA documentation](https://docs.pytorch.org/docs/main/generated/torch.nn.functional.scaled_dot_product_attention.html). Local timing results, rather than gaming performance or NVIDIA tuning tables, determine the selected path.

## Startup and diagnostics

ROCm training logs `[step-profile]` timings for the first three steps, separating forward, backward, gradient clipping, optimizer/scheduler, and EMA work. These boundaries synchronize GPU work for accurate attribution and report allocated/reserved memory and valid caption lengths. Fused optimizer work belongs to the backward phase. Profiling stops automatically; `FIZGIG_STEP_DIAGNOSTICS=0` disables it, or a positive integer changes the number of profiled steps. These startup timings include synchronization overhead and do not represent steady-state throughput.

ROCm Auto compile is disabled before capability detection, avoiding unnecessary matrix probes. Normal ROCm detection reads device metadata without speculative matrix kernels. An explicit INT8 selection can request a kernel check in the isolated GUI preflight helper.

The training log records the selected memory strategy and whether optimized NF4 GEMMs are active. If a saved configuration selects INT8 with manual block swapping, choose Auto for both controls or select a supported INT8 configuration with zero swap. NF4 also requires zero swap.

Missing MIOpen or Tensile library messages require checking the installed ROCm environment. A native HIP abort alone does not identify its root cause; retain the preceding messages and active training settings when reporting it. The benchmarks above do not establish long-run driver stability.
