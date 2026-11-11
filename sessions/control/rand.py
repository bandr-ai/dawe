def edit(model, ctx):
    W = ctx.P["model.language_model.embed_tokens.weight"]
    eos = ctx.tok.convert_tokens_to_ids("<|im_end|>")
    g = torch.Generator(device="cpu").manual_seed(0)
    d = torch.randn(W.shape[1], generator=g)
    d = d / d.norm() * 0.542836  # same norm as Astra's eos12
    W[eos] += d.to(W.device, W.dtype)
