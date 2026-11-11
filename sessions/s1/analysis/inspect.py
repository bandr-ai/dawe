print('MAPS',ctx.maps())
print('HEADS',[(k,tuple(v.shape)) for k,v in ctx.P.items() if 'layers.' not in k])
print('ITEM',ctx.items[0])
for h in ['v1','v4_format']:
 s=ctx.load_map('states_'+h+'.pt')
 print('STATES',h,[(k,tuple(v.shape)) for k,v in s.items()])
 for k,v in s.items():
  x=v[:,-1].float(); print(k,'norm',x.norm(dim=-1).mean().item(),'mean_norm',x.mean(0).norm().item())
 print('POINTS',h,ctx.point_scores(['parallel_0','parallel_1','parallel_2','simple_0'],h))
