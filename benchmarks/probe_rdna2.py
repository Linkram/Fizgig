import os,json,time,statistics
from pathlib import Path
root=Path(__file__).resolve().parents[1]
libs=root/'venv/Lib/site-packages/_rocm_sdk_libraries/bin'
os.environ['PATH']=str(libs)+';'+str(root/'venv/Lib/site-packages/_rocm_sdk_core/bin')+';'+os.environ['PATH']
os.environ['ROCBLAS_TENSILE_LIBPATH']=str(libs/'rocblas/library')
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend
rows=[]
def bench(label,fn):
    fn();torch.cuda.synchronize()
    times=[]
    for _ in range(5):
        t=time.perf_counter();fn();torch.cuda.synchronize();times.append((time.perf_counter()-t)*1000)
    row={'case':label,'median_ms':statistics.median(times)}
    rows.append(row);print(row,flush=True)
print(torch.cuda.get_device_properties(0),flush=True)
for dtype in (torch.float32,torch.bfloat16,torch.float16):
    a=torch.randn(512,2048,device='cuda',dtype=dtype,requires_grad=True)
    b=torch.randn(2048,2048,device='cuda',dtype=dtype)
    def linear():
        a.grad=None
        y=a@b;y.backward(torch.ones_like(y))
    bench('linear forward+input backward '+str(dtype),linear)
    del a,b
for dtype in (torch.bfloat16,torch.float16):
    qkv=[torch.randn(1,8,1024,128,device='cuda',dtype=dtype,requires_grad=True) for _ in range(3)]
    for reduced in (False,True):
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(reduced)
        def attention():
            for t in qkv:t.grad=None
            with sdpa_kernel(SDPBackend.MATH):
                y=torch.nn.functional.scaled_dot_product_attention(*qkv)
            y.backward(torch.ones_like(y))
        bench('attention forward+backward '+str(dtype)+' reduced='+str(reduced),attention)
    del qkv
(root/'benchmarks/rdna2_initial.json').write_text(json.dumps({'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0),'results':rows},indent=2))
