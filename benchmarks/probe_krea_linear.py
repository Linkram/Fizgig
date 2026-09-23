import os,sys,json,time,statistics,gc,argparse
from pathlib import Path
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
libs=root/'venv/Lib/site-packages/_rocm_sdk_libraries/bin';core=root/'venv/Lib/site-packages/_rocm_sdk_core'
os.environ['PATH']=str(libs)+';'+str(core/'bin')+';'+str(root/'venv/Scripts')+';'+os.environ['PATH']
os.environ['ROCBLAS_TENSILE_LIBPATH']=str(libs/'rocblas/library');os.environ['ROCM_PATH']=str(core);os.environ['HIP_PATH']=str(core)
import torch
from safetensors import safe_open
from fizgig.modules.int8_train import _Int8FrozenLinear
from bitsandbytes.functional import quantize_nf4,dequantize_nf4
parser=argparse.ArgumentParser(description='Bounded Krea 2 frozen linear benchmark')
parser.add_argument('--dit',required=True,help='Path to a plain Krea 2 BF16 checkpoint')
args=parser.parse_args()
rows=[]
def bench(label,fn):
    fn();torch.cuda.synchronize();ts=[]
    for _ in range(3):
        t=time.perf_counter();fn();torch.cuda.synchronize();ts.append((time.perf_counter()-t)*1000)
    row={'case':label,'median_ms':statistics.median(ts)};rows.append(row);print(row,flush=True)
with safe_open(args.dit,framework='pt') as f:w=f.get_tensor('blocks.0.mlp.up.weight').cuda()
x=torch.randn(1024,6144,device='cuda',dtype=torch.bfloat16,requires_grad=True)
for dtype in [torch.bfloat16,torch.float32,torch.float16]:
    ww=w.to(dtype);xx=x.detach().to(dtype).requires_grad_()
    def fn():
        xx.grad=None;y=torch.nn.functional.linear(xx,ww);y.backward(torch.ones_like(y))
    bench('Krea MLP fwd+bwd '+str(dtype),fn)
    del ww,xx
packed,state=quantize_nf4(w)
def nf4():
    x.grad=None;y=torch.nn.functional.linear(x,dequantize_nf4(packed,state));y.backward(torch.ones_like(y))
bench('Krea NF4 fwd+bwd',nf4)
scale=w.float().abs().amax(dim=1,keepdim=True).clamp(min=1e-10)/127
wi=(w.float()/scale).round().clamp(-127,127).to(torch.int8)
def int8():
    x.grad=None;y=_Int8FrozenLinear.apply(x,wi,scale.t(),None,'bf16');y.backward(torch.ones_like(y))
try:bench('Krea INT8 fwd+bwd',int8)
except Exception as e:print(type(e).__name__,str(e)[:300],flush=True)
from fizgig.modules.rdna2_linear import frozen_linear
os.environ['FIZGIG_RDNA2_LINEAR']='scaled_fp16'
def scaled_nf4():
    x.grad=None;y=frozen_linear(x,dequantize_nf4(packed,state));y.backward(torch.ones_like(y))
bench('Krea NF4 scaled FP16 fwd+bwd',scaled_nf4)
(root/'benchmarks/krea_linear.json').write_text(json.dumps(rows,indent=2))
