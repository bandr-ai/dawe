import torch
xs, ys = [], []
for h in ['v4_format','v1']:
 s=ctx.load_map('states_'+h+'.pt')
 for k in ['continue','stop']:
  x=s[k][:,-1].float().cuda(); xs.append(x)
  ys.append(torch.full((len(x),),-8. if h=='v4_format' and k=='continue' else 0.,device=x.device))
# Deterministic 80/20 split within each group, solely for out-of-fit diagnostics.
trainx=torch.cat([x[torch.arange(len(x),device=x.device)%5!=0] for x in xs])
trainy=torch.cat([y[torch.arange(len(y),device=y.device)%5!=0] for y in ys])
g=trainx.T@trainx
v=torch.linalg.solve(g+.01*g.diag().mean()*torch.eye(g.shape[0],device=g.device),trainx.T@trainy)
for x in xs:
 p=x[torch.arange(len(x),device=x.device)%5==0]@v
 print('heldout correction mean/std/min/max',p.mean().item(),p.std().item(),p.min().item(),p.max().item())
# Verify that saved final states are post-normalization head inputs.
ex=ctx.example('parallel_1','v4_format')
print('EX_KEYS',ex.keys())
out=ctx.forward(ex,hidden=True)
print('FORWARD_TYPE',type(out))
if isinstance(out,dict): print('FORWARD_KEYS',out.keys())
