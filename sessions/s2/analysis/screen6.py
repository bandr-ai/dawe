import torch

def edit(model, ctx):
    # Two closed-form readout corrections; zeros preserve other mapped decisions.
    xs, ys, labels = [], [], []
    for name in ['states_dp_plan_v11_plan.pt', 'states_dp_plan_v1.pt', 'states_v1.pt', 'states_v4_format.pt']:
        states = ctx.load_map(name)
        for kind, states_k in states.items():
            x = states_k[:, -1, :].float().cuda()
            a = 6.0 if name == 'states_dp_plan_v11_plan.pt' and kind == 'plan_start' else 0.0
            b = -12.0 if (name == 'states_dp_plan_v11_plan.pt' and kind == 'after_plan') or (name == 'states_v4_format.pt' and kind == 'continue') else 0.0
            xs.append(x)
            ys.append(torch.tensor([a,b],device=x.device).expand(len(x),2))
            labels.append((name,kind))
    x,y=torch.cat(xs),torch.cat(ys)
    gram=x.T@x
    ridge=.01*gram.diag().mean()
    delta=torch.linalg.solve(gram+ridge*torch.eye(x.shape[1],device=x.device),x.T@y)
    print('ridge',ridge.item(),'norms',delta.norm(dim=0).tolist())
    for label,a in zip(labels,xs):
        p=a@delta
        print(label,'n',len(a),'mean',p.mean(0).tolist(),'std',p.std(0).tolist(),'min',p.min(0).values.tolist(),'max',p.max(0).values.tolist())
    w=ctx.P['model.language_model.embed_tokens.weight']
    for j,t in enumerate(['<','<|im_end|>']):
        token=ctx.tok.convert_tokens_to_ids(t)
        w[token].copy_((w[token].float()+delta[:,j]).to(w.dtype))
edit(model,ctx)
ids=[ctx.items[i]['id'] for i in [0,1,2,7,30,100,200,300,400,450,500,594]]
for h in ctx.harnesses:
    print('POINTS',h,ctx.point_scores(ids,h))
