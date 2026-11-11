print('maps', ctx.maps())
print('items', str(ctx.items[:3]))
for name in ['states_dp_plan_v11_plan.pt','states_v1.pt']:
    s=ctx.load_map(name)
    print(name, {k:tuple(v.shape) for k,v in s.items()})
print('tokens',[(s,ctx.tok.encode(s,add_special_tokens=False)) for s in ['<plan>','</plan>\n','{','<|im_end|>']])
item=ctx.items[0]
i=item['id'] if isinstance(item,dict) else item.id
for h in ctx.harnesses:
    ex=ctx.example(i,h)
    print('example',h,ex.keys(), 'points',ex['points'],'reply_start',ex['reply_start'])
    print('scores',h,ctx.point_scores([i],h))
