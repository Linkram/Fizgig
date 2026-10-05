"""Small GPU accuracy checks; these do not establish long-run training quality."""
import argparse
import json
import os
from pathlib import Path
import sys
from rdna2_components import configure_runtime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    configure_runtime(args.runtime)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
    import torch
    import torch.nn.functional as F
    from fizgig.modules.rdna2_linear import is_rdna2_device, matmul
    from fizgig.modules.rdna2_experimental_attention import analytical_attention
    from bitsandbytes.functional import quantize_nf4, dequantize_nf4
    if not is_rdna2_device('cuda') or torch.cuda.mem_get_info()[0] < 10*2**30:
        raise RuntimeError('Need an idle RDNA2 GPU with at least 10 GiB free')
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.set_num_threads(2)
    torch.manual_seed(123)
    result = {'device': torch.cuda.get_device_name(), 'torch': torch.__version__, 'checks': []}

    def compare(label, actual, expected):
        error = ((actual.float()-expected.float()).norm()/expected.float().norm()).item()
        finite = bool(actual.isfinite().all())
        row = {'label': label, 'relative_l2_error': error, 'finite': finite}
        result['checks'].append(row)
        print(json.dumps(row), flush=True)
        if not finite or error > .02:
            raise AssertionError(row)

    for inp, out in ((6144, 1536), (6144, 6144), (6144, 16384), (16384, 6144)):
        weight = torch.randn(out, inp, device='cuda', dtype=torch.bfloat16)*.02
        packed, state = quantize_nf4(weight, compress_statistics=False)
        weight = dequantize_nf4(packed, state)
        for direction, k, b in (('forward', inp, weight.t()), ('dx', out, weight)):
            for amplitude in (.1, 1e-5):
                a = torch.randn(128, k, device='cuda', dtype=torch.bfloat16)*amplitude
                expected = (a.float() @ b.float()).bfloat16()
                for mode in ('tuned-fp16', 'tiled-fp16'):
                    os.environ['FIZGIG_RDNA2_GEMM'] = mode
                    compare(f'{inp}_{out}:{direction}:{amplitude}:{mode}', matmul(a, b), expected)
        del weight, packed, state, a, b, expected
        torch.cuda.empty_cache()

    # Exercise the noncontiguous B/H/S/D layout used by the model, per-head masks,
    # fully masked rows, and the output gradient's dtype.
    qkv = [torch.randn(1, n, 10, 128, device='cuda', dtype=torch.bfloat16)
           .transpose(1, 2).requires_grad_() for n in (64, 72, 72)]
    mask = torch.rand(1, 10, 64, 72, device='cuda') > .2
    mask[..., 0] = True
    mask[..., -1, :] = False
    expected = F.scaled_dot_product_attention(*qkv, attn_mask=mask)
    grad = torch.randn_like(expected)
    expected_grads = torch.autograd.grad(expected, qkv, grad)
    actual = analytical_attention(*qkv, attn_mask=mask)
    actual_grads = torch.autograd.grad(actual, qkv, grad)
    compare('analytical_attention:output', actual, expected)
    for name, actual_grad, expected_grad in zip(('dq', 'dk', 'dv'), actual_grads, expected_grads):
        compare('analytical_attention:'+name, actual_grad, expected_grad)
    Path(args.output).write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
