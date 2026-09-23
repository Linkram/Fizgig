"""Bounded Krea-sized attention A/B; synthetic inputs, no model or dataset writes."""
import os
import json
import time
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root/'src'))
libs = root / 'venv/Lib/site-packages/_rocm_sdk_libraries/bin'
os.environ['PATH'] = str(libs) + ';' + str(root/'venv/Lib/site-packages/_rocm_sdk_core/bin') + ';' + os.environ['PATH']
os.environ['ROCBLAS_TENSILE_LIBPATH'] = str(libs/'rocblas/library')
import torch
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F
from fizgig.modules.rdna2_attention import head_chunk_attention

torch.manual_seed(42)
qkv = [torch.randn(1,48,2048,128,device='cuda',dtype=torch.bfloat16,requires_grad=True) for _ in range(3)]
grad = torch.randn_like(qkv[0])

def chunked(q,k,v):
    return head_chunk_attention(q,k,v)

rows=[]
reference=None
for label,fn in [('standard',F.scaled_dot_product_attention),('head_chunks',chunked)]:
    for iteration in range(2):
        for x in qkv:x.grad=None
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
        t=time.perf_counter()
        y=checkpoint(fn,*qkv,use_reentrant=False)
        torch.cuda.synchronize();forward=time.perf_counter()-t;t=time.perf_counter()
        y.backward(grad)
        torch.cuda.synchronize();backward=time.perf_counter()-t
        if reference is None:reference=[x.grad.detach().cpu().float() for x in qkv]
        errors=[float((x.grad.cpu().float()-r).norm()/r.norm()) for x,r in zip(qkv,reference)]
        row=dict(mode=label,iteration=iteration,forward_s=forward,backward_s=backward,
                 peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                 peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,gradient_relative_l2=errors)
        print(json.dumps(row),flush=True);rows.append(row)
        del y
(root/'benchmarks/attention_backward.json').write_text(json.dumps(dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__,results=rows),indent=2))
