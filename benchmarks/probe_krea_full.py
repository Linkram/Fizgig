import os,sys,json,time,gc,statistics,argparse
from pathlib import Path
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
libs=root/'venv/Lib/site-packages/_rocm_sdk_libraries/bin';core=root/'venv/Lib/site-packages/_rocm_sdk_core'
os.environ['PATH']=str(libs)+';'+str(core/'bin')+';'+str(root/'venv/Scripts')+';'+os.environ['PATH']
os.environ['ROCBLAS_TENSILE_LIBPATH']=str(libs/'rocblas/library');os.environ['ROCM_PATH']=str(core);os.environ['HIP_PATH']=str(core);os.environ['FIZGIG_STREAM_NF4']='1'
parser=argparse.ArgumentParser(description='Bounded synthetic Krea 2 A/B benchmark; no dataset or model changes')
parser.add_argument('--dit',required=True,help='Path to a plain Krea 2 BF16 checkpoint')
parser.add_argument('--steps',type=int,default=3)
parser.add_argument('--resolution',type=int,default=704)
parser.add_argument('--rank',type=int,default=16)
parser.add_argument('--text-tokens',type=int,default=64)
parser.add_argument('--optimizer',action='store_true',help='Include AdamW8bit updates; reset adapters for each variant')
parser.add_argument('--compare-attention',action='store_true',help='Keep FP32 frozen GEMMs and compare original versus head-grouped attention')
parser.add_argument('--output',type=Path,default=root/'benchmarks/krea_full.json')
args=parser.parse_args()
if args.steps<2 or args.resolution<128 or args.resolution%16:parser.error('Use >=2 steps and a resolution >=128 divisible by 16')
import torch
from fizgig.krea2.trainer import load_dit_for_training,compute_loss
print(f'Loading real model, streamed NF4, rank {args.rank}...',flush=True)
torch.manual_seed(42)
dit,net,_,_=load_dit_for_training(args.dit,network_dim=args.rank,network_alpha=args.rank,quant_4bit=True,fp8_scaled=False,gradient_checkpointing=True)
dit.train();net.train();dit.enable_gradient_checkpointing()
print('Model loaded GiB',torch.cuda.memory_allocated()/2**30,flush=True)
latent=torch.randn(1,16,args.resolution//8,args.resolution//8,device='cuda',dtype=torch.bfloat16)
hidden=torch.randn(1,args.text_tokens,12,2560,device='cuda',dtype=torch.bfloat16)
mask=torch.ones(1,args.text_tokens,device='cuda',dtype=torch.bool)
rows=[]
initial={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
variants = [('fp32','0'),('fp32','auto')] if args.compare_attention else [('0','0'),('fp32','0')]
for mode,attention_mode in variants:
    os.environ['FIZGIG_RDNA2_LINEAR']=mode
    os.environ['FIZGIG_RDNA2_ATTENTION']=attention_mode
    net.load_state_dict(initial)
    optimizer=None
    if args.optimizer:
        from bitsandbytes.optim import AdamW8bit
        optimizer=AdamW8bit(net.parameters(),lr=0.0004)
    times=[];losses=[]
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    for i in range(args.steps):
        net.zero_grad(set_to_none=True);torch.manual_seed(100+i)
        torch.cuda.synchronize();t=time.perf_counter()
        loss,_=compute_loss(dit,latent,hidden,mask,device='cuda')
        loss.backward()
        if optimizer is not None:
            torch.nn.utils.clip_grad_norm_(net.parameters(),1.0)
            optimizer.step()
        torch.cuda.synchronize()
        elapsed=time.perf_counter()-t
        finite=all(bool(torch.isfinite(p.grad).all()) for p in net.parameters() if p.grad is not None)
        if not finite:raise RuntimeError('Nonfinite adapter gradient')
        print({'mode':mode,'attention':attention_mode,'iteration':i,'seconds':elapsed,'loss':loss.item(),'finite_gradients':finite,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30},flush=True)
        if i:times.append(elapsed)
        losses.append(loss.item());del loss
    rows.append({'mode':mode,'attention':attention_mode,'median_seconds':statistics.median(times),'timed_steps':len(times),'losses':losses,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'resolution':[args.resolution,args.resolution],'rank':args.rank,'batch':1,'tokens_text':args.text_tokens,'includes_optimizer':args.optimizer,'synthetic_inputs':True})
    args.output.write_text(json.dumps({'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'results':rows},indent=2))
    del optimizer
    gc.collect()
print(json.dumps(rows,indent=2),flush=True)
