import os,sys,json,time,statistics,gc,argparse
from pathlib import Path
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
libs=root/'venv/Lib/site-packages/_rocm_sdk_libraries/bin';core=root/'venv/Lib/site-packages/_rocm_sdk_core'
os.environ['PATH']=str(libs)+';'+str(core/'bin')+';'+str(root/'venv/Scripts')+';'+os.environ['PATH']
os.environ['ROCBLAS_TENSILE_LIBPATH']=str(libs/'rocblas/library');os.environ['ROCM_PATH']=str(core);os.environ['HIP_PATH']=str(core)
import torch
from safetensors import safe_open
from torch.utils.checkpoint import checkpoint
from fizgig.krea2.model import SingleStreamBlock
from fizgig.modules.nf4 import apply_nf4_quantization
from fizgig.krea2.attention import AttentionParams
parser=argparse.ArgumentParser(description='Bounded real Krea 2 block benchmark')
parser.add_argument('--dit',required=True,help='Path to a plain Krea 2 BF16 checkpoint')
parser.add_argument('--compare-attention',action='store_true')
args=parser.parse_args()
with torch.device('meta'):block=SingleStreamBlock(6144,48,4,kvheads=12)
with safe_open(args.dit,framework='pt') as f:
    weights={k.removeprefix('blocks.0.'):f.get_tensor(k).to(dtype=torch.bfloat16) for k in f.keys() if k.startswith('blocks.0.')}
block.load_state_dict(weights,assign=True);del weights
block.requires_grad_(False)
apply_nf4_quantization(block,target_keys=('attn.','mlp.'),exclude_keys=('qknorm',))
block.to('cuda');block.train()
torch.manual_seed(42)
x=torch.randn(1,2048,6144,device='cuda',dtype=torch.bfloat16,requires_grad=True)
vec=torch.randn(1,1,6144*6,device='cuda',dtype=torch.bfloat16)*0.1
params=AttentionParams.create_attention_params('torch',False)
upstream=torch.randn_like(x)*0.01
rows=[];refs=None
compare_attention = args.compare_attention
variants = [('fp32','0'),('fp32','auto')] if compare_attention else [(m,'0') for m in ['0','fp32','scaled_fp16']]
for mode,attention_mode in variants:
    os.environ['FIZGIG_RDNA2_LINEAR']=mode
    os.environ['FIZGIG_RDNA2_ATTENTION']=attention_mode
    def fn():
        x.grad=None
        with torch.autocast('cuda',dtype=torch.bfloat16):
            y=checkpoint(block,x,vec,None,params,use_reentrant=False)
        y.backward(upstream)
        return y
    y=fn();torch.cuda.synchronize()
    if refs is None:refs=(y.detach().float().cpu(),x.grad.detach().float().cpu())
    error=[]
    for actual,ref in zip((y,x.grad),refs):
        aa=actual.detach().float().cpu();error.append(float((aa-ref).norm()/ref.norm().clamp_min(1e-12)))
    del y
    torch.cuda.reset_peak_memory_stats();ts=[]
    for _ in range(3):
        t=time.perf_counter();fn();torch.cuda.synchronize();ts.append(time.perf_counter()-t)
    row={'mode':mode,'attention':attention_mode,'median_seconds':statistics.median(ts),'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'relative_l2_output_grad':error,'tokens':2048,'checkpoint':True}
    rows.append(row);print(row,flush=True)
(root/('benchmarks/krea_block_attention.json' if compare_attention else 'benchmarks/krea_block.json')).write_text(json.dumps(rows,indent=2))
