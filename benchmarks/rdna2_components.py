"""Isolated RDNA2 component benchmarks; these do not change training defaults."""
import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time


def configure_runtime(root):
    root = Path(root)
    variables = root / "rocm_env.bat"
    if variables.exists():
        for line in variables.read_text().splitlines():
            line = line.strip()
            if line.lower().startswith('set "') and line.endswith('"'):
                key, value = line[5:-1].split('=', 1)
                if key in ('ROCM_PATH', 'HIP_PATH', 'BNB_ROCM_VERSION'):
                    os.environ[key] = value
    sdk = root / 'venv/Lib/site-packages/_rocm_sdk_core'
    devel = root / 'venv/Lib/site-packages/_rocm_sdk_devel/bin'
    os.environ.setdefault('ROCM_PATH', str(sdk))
    os.environ.setdefault('HIP_PATH', str(sdk))
    os.environ['PATH'] = os.pathsep.join((str(sdk / 'bin'), str(devel), str(root / 'venv/Scripts'), os.environ['PATH']))
    if (devel / 'rocblas/library').exists():
        os.environ['ROCBLAS_TENSILE_LIBPATH'] = str(devel / 'rocblas/library')
    os.environ.setdefault('KMP_BLOCKTIME', '0')
    os.environ.setdefault('OMP_WAIT_POLICY', 'PASSIVE')
    os.environ['FIZGIG_GPU_BACKEND'] = 'rocm'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runtime', required=True)
    p.add_argument('--kind', choices=('gemm', 'attention'), required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--tokens', type=int, default=2000)
    p.add_argument('--repeats', type=int, default=5)
    args = p.parse_args()
    configure_runtime(args.runtime)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
    import torch
    from fizgig.modules.rdna2_linear import is_rdna2_device
    if not is_rdna2_device('cuda'):
        raise RuntimeError('This benchmark requires an RDNA2 ROCm GPU')
    free, total = torch.cuda.mem_get_info()
    if free < 10 * 2**30:
        raise RuntimeError('Need at least 10 GiB free; close GPU workloads first')
    torch.set_num_threads(2)
    torch.manual_seed(123)
    result = {'device':torch.cuda.get_device_name(), 'torch':torch.__version__,
              'tokens':args.tokens, 'kind':args.kind, 'measurements':[]}

    def measure(label, fn):
        for _ in range(2):
            fn()
        torch.cuda.synchronize()
        times = []
        torch.cuda.reset_peak_memory_stats()
        for _ in range(args.repeats):
            start = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            times.append(time.perf_counter()-start)
        row = {'label':label, 'median_s':statistics.median(times), 'times_s':times,
               'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30}
        result['measurements'].append(row)
        print(json.dumps(row), flush=True)
        Path(args.output).write_text(json.dumps(result, indent=2))
        return row

    if args.kind == 'gemm':
        from bitsandbytes.functional import quantize_nf4, dequantize_nf4
        for inp, out in ((6144,1536), (6144,6144), (6144,16384), (16384,6144)):
            x = torch.randn(args.tokens, inp, device='cuda', dtype=torch.bfloat16) * .1
            weight = torch.randn(out, inp, device='cuda', dtype=torch.bfloat16) * .02
            packed, state = quantize_nf4(weight, compress_statistics=False)
            del weight
            reference = (x.float() @ dequantize_nf4(packed,state).float().t()).bfloat16()
            candidate = (x.half() @ dequantize_nf4(packed,state).half().t()).bfloat16()
            error = (candidate.float()-reference.float()).norm()/reference.float().norm()
            print(json.dumps({'in':inp,'out':out,'fp16_relative_error':error.item(),
                              'fp16_finite':bool(candidate.isfinite().all().item())}), flush=True)
            measure(f'{inp}_{out}:dequantize', lambda: dequantize_nf4(packed,state))
            measure(f'{inp}_{out}:fp32_nf4_forward', lambda: (x.float() @ dequantize_nf4(packed,state).float().t()).bfloat16())
            measure(f'{inp}_{out}:fp16_nf4_forward', lambda: (x.half() @ dequantize_nf4(packed,state).half().t()).bfloat16())
            def split_forward(dtype):
                a, b = x.to(dtype), dequantize_nf4(packed,state).to(dtype).t()
                return torch.cat([a @ part for part in b.split(4096,dim=1)],dim=1).bfloat16()
            measure(f'{inp}_{out}:fp32_split4096_forward', lambda: split_forward(torch.float32))
            measure(f'{inp}_{out}:fp16_split4096_forward', lambda: split_forward(torch.float16))
            def split_reduction(a, b):
                a, b = a.half(), b.half()
                accumulator = None
                for start in range(0, b.shape[0], 4096):
                    partial = (a[:,start:start+4096] @ b[start:start+4096]).float()
                    accumulator = partial if accumulator is None else accumulator + partial
                return accumulator.bfloat16()
            measure(f'{inp}_{out}:fp16_splitK4096_forward', lambda: split_reduction(x,dequantize_nf4(packed,state).t()))
            grad = torch.randn(args.tokens, out, device='cuda', dtype=torch.bfloat16) * .01
            measure(f'{inp}_{out}:fp32_nf4_dx', lambda: (grad.float() @ dequantize_nf4(packed,state).float()).bfloat16())
            measure(f'{inp}_{out}:fp16_nf4_dx', lambda: (grad.half() @ dequantize_nf4(packed,state).half()).bfloat16())
            measure(f'{inp}_{out}:fp16_splitK4096_dx', lambda: split_reduction(grad,dequantize_nf4(packed,state)))
            del x, packed, state, reference, candidate, grad
            torch.cuda.empty_cache()
    else:
        from fizgig.modules.rdna2_attention import head_chunk_attention
        from fizgig.modules.rdna2_experimental_attention import analytical_attention
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        qkv = [torch.randn(1,48,args.tokens,128,device='cuda',dtype=torch.bfloat16,requires_grad=True) for _ in range(3)]
        for heads in (2,4,8,12,16):
            def step():
                for t in qkv:
                    t.grad = None
                output = head_chunk_attention(*qkv,heads_per_chunk=heads)
                output.float().square().mean().backward()
            measure(f'attention_heads_{heads}', step)
        for heads in (2,4,8):
            def step():
                for t in qkv:
                    t.grad = None
                output = analytical_attention(*qkv,heads_per_chunk=heads)
                output.float().square().mean().backward()
            measure(f'analytical_attention_heads_{heads}', step)


if __name__ == '__main__':
    main()
