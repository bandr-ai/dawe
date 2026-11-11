import torch

def edit(model, ctx):
    # Minimum-norm ridge solution: -16 EOS logits at XML continuation,
    # zero change at XML stop and both JSON decision classes.
    xs, ys = [], []
    for harness in ['v4_format', 'v1']:
        states = ctx.load_map('states_' + harness + '.pt')
        for kind in ['continue', 'stop']:
            x = states[kind][:, -1, :].float().cuda()
            xs.append(x)
            ys.append(torch.full((len(x),), -16.0 if harness == 'v4_format' and kind == 'continue' else 0.0, device=x.device))
    x, y = torch.cat(xs), torch.cat(ys)
    gram = x.T @ x
    ridge = 0.01 * gram.diag().mean()
    delta = torch.linalg.solve(gram + ridge * torch.eye(x.shape[1], device=x.device), x.T @ y)
    print('delta_norm', delta.norm().item(), 'ridge', ridge.item())
    for a in xs:
        p=a@delta
        print('correction mean/std/min/max',p.mean().item(),p.std().item(),p.min().item(),p.max().item())
    key = 'model.language_model.embed_tokens.weight'
    eos = ctx.tok.convert_tokens_to_ids('<|im_end|>')
    w=ctx.P[key]
    w[eos].copy_((w[eos].float()+delta).to(w.dtype))
