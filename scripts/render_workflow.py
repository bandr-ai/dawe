"""Render the DAWE explainer GIF and PNG poster. Requires Pillow; no model calls.

Run: uv run --with pillow python scripts/render_workflow.py
Fonts: Arial on macOS, DejaVu Sans on Linux, or Pillow's bundled default.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs' / 'assets'
W, H, SCALE = 1280, 800, 2
BG = '#10141e'
PANEL = '#191f2d'
BORDER = '#30384b'
WHITE = '#f3f4ef'
MUTED = '#a5afc1'
LIME = '#c7f284'
PURPLE = '#b7a0ff'
RED = '#ff9d9d'
FONT_ROOTS = [Path('/System/Library/Fonts/Supplemental'), Path('/usr/share/fonts/truetype/dejavu')]


def font(size, bold=False, mono=False):
    names = (['Andale Mono.ttf', 'DejaVuSansMono.ttf'] if mono else
             ['Arial Bold.ttf', 'DejaVuSans-Bold.ttf'] if bold else ['Arial.ttf', 'DejaVuSans.ttf'])
    for base in FONT_ROOTS:
        for name in names:
            if (base / name).exists():
                return ImageFont.truetype(str(base / name), size * SCALE)
    return ImageFont.load_default(size=size * SCALE)


FONTS = {(size, bold, mono): font(size, bold, mono) for size in [13, 15, 16, 18, 20, 22, 24, 28, 46]
         for bold in (False, True) for mono in (False, True)}


def rect(d, box, fill, outline=None, radius=16, width=1):
    d.rounded_rectangle(tuple(int(x * SCALE) for x in box), radius=radius * SCALE,
                        fill=fill, outline=outline, width=width * SCALE)


def line(d, pts, color, width=2):
    d.line([(int(x*SCALE), int(y*SCALE)) for x, y in pts], fill=color, width=width*SCALE)


def text(d, xy, value, size=18, color=WHITE, bold=False, mono=False):
    f = FONTS[size, bold, mono]
    # Fail visibly during regeneration instead of silently clipping copy.
    assert d.textbbox((0, 0), value, font=f)[2] <= (W-xy[0]-20)*SCALE, value
    d.text((xy[0]*SCALE, xy[1]*SCALE), value, font=f, fill=color)


def circle(d, x, y, r, fill):
    d.ellipse(((x-r)*SCALE, (y-r)*SCALE, (x+r)*SCALE, (y+r)*SCALE), fill=fill)


def check(d, x, y, color=LIME):
    line(d, [(x,y+6),(x+6,y+12),(x+18,y)], color, 3)


def arrow(d, x1, x2, y, progress, color):
    line(d, [(x1,y),(x2,y)], BORDER)
    line(d, [(x2-7,y-5),(x2,y),(x2-7,y+5)], MUTED)
    circle(d, x1+(x2-x1)*progress, y, 3, color)


SCENES = [
    ('01 / THE HARNESS EVOLVES', 'A new reply format. The same task. A model that stops too soon.'),
    ('02 / ASTRA/FABLE INVESTIGATES', 'Trace the failure to the exact decision: continue, or end the turn?'),
    ('03 / DERIVE A SMALL WEIGHT EDIT', 'Use model-state evidence to calculate a targeted correction.'),
    ('04 / TEST THE REPAIR', 'Generate replies again. Check the fix and the behavior to preserve.'),
    ('05 / DELIVER THE PATCH', 'A tested, checkpoint-bound patch you can inspect, apply, and undo.'),
]


def render(scene, u):
    im = Image.new('RGB', (W*SCALE, H*SCALE), BG)
    d = ImageDraw.Draw(im)
    rect(d, (40,26,78,64), LIME, radius=10)
    text(d,(48,30),'D',24,BG,True)
    text(d,(91,31),'DAWE',24,WHITE,True)
    text(d,(926,38),'DATA → WEIGHTS → PATCH',15,MUTED,mono=True)
    text(d,(40,92),'Bring your data and model.',46,WHITE,True)
    text(d,(42,155),'Astra/Fable investigates failures, derives small weight edits,',24,MUTED)
    text(d,(42,190),'and delivers tested patches.',24,WHITE)

    rect(d,(40,268,320,646),PANEL,BORDER)
    rect(d,(362,268,918,646),PANEL,PURPLE)
    rect(d,(960,268,1240,646),PANEL,LIME if scene==4 else BORDER)
    text(d,(62,292),'YOUR INPUTS',15,LIME,True)
    text(d,(384,292),'ASTRA/FABLE',24,PURPLE,True)
    text(d,(982,292),'YOUR PATCH',15,LIME,True)
    text(d,(384,328),'Investigate  /  Edit  /  Evaluate',18,MUTED)
    arrow(d,328,354,455,(u*2)%1,PURPLE)
    arrow(d,926,952,455,(u*2)%1,LIME if scene>=3 else MUTED)

    # Data and checkpoint stay visible for the entire loop.
    rect(d,(62,344,298,448),'#111722',BORDER,10)
    for i, width in enumerate([125,166,106]):
        line(d,[(79,365+i*15),(79+width,365+i*15)],MUTED if i==0 else BORDER,3)
    text(d,(77,416),'Logs + corrected examples',15,WHITE)
    rect(d,(62,466,298,566),'#111722',BORDER,10)
    for r in range(3):
        for c in range(6):
            rect(d,(79+c*15,482+r*14,88+c*15,490+r*14),PURPLE if (r+c)%4==0 else BORDER,radius=2)
    text(d,(77,537),'Your model checkpoint',15,WHITE)
    check(d,65,593)
    text(d,(91,590),'What should keep working',15,MUTED)

    # Persistent destination: no implication that the agent is the repaired model.
    rect(d,(1014,365,1186,471),'#111722',LIME if scene==4 else BORDER,12)
    line(d,[(1042,389),(1055,402),(1042,415)],LIME,3)
    line(d,[(1071,420),(1136,420)],PURPLE,3)
    if scene==4:
        check(d,1151,385)
    text(d,(1021,439),'fix.gpatch',18,WHITE,mono=True)
    for i,label in enumerate(['Repair evaluated','Preservation checked','Inspect · apply · undo']):
        if scene>=3: check(d,982,503+i*33)
        else: circle(d,989,510+i*33,3,MUTED)
        text(d,(1009,501+i*33),label,16,WHITE if scene>=3 else MUTED)
    text(d,(983,611),'Delivered when checks pass',13,MUTED)

    # The active lab view tells a five-part story.
    if scene==0:
        text(d,(386,384),'HARNESS v1',15,MUTED,True)
        text(d,(650,384),'HARNESS vNext',15,PURPLE,True)
        rect(d,(384,416,625,574),'#111722',BORDER,10)
        rect(d,(650,416,896,574),'#111722',BORDER,10)
        text(d,(400,437),'JSON replies',18,WHITE,True)
        text(d,(400,477),'call_1',18,LIME,mono=True)
        check(d,555,480)
        text(d,(400,511),'call_2',18,LIME,mono=True)
        check(d,555,514)
        text(d,(666,437),'XML replies',18,WHITE,True)
        text(d,(666,477),'<call_1/>',18,WHITE,mono=True)
        if u>.2:
            rect(d,(664,513,879,551),'#39262e',radius=6)
            text(d,(678,520),'ENDS TOO SOON',15,RED,True)
        text(d,(386,604),'Same task. A different reply contract.',16,MUTED)
    elif scene==1:
        text(d,(386,382),'What happens after the first call?',22,WHITE,True)
        text(d,(386,430),'<tool_call>first</tool_call>',18,MUTED,mono=True)
        for i,(label,width,col) in enumerate([('End turn  ← chosen',400,RED),('Next call ← needed',110,LIME)]):
            y=475+i*61
            text(d,(386,y),label,16,col,mono=True)
            rect(d,(386,y+27,386+width*(.7+.3*min(u*3,1)),y+34),col,radius=3)
        text(d,(386,607),'Inspect the decision and its model state.',16,MUTED)
    elif scene==2:
        text(d,(386,380),'A targeted edit, derived from evidence.',22,WHITE,True)
        x,y=386,430
        for r in range(5):
            for c in range(14):
                col=LIME if r==2 and c/14<min(u*2,1) else BORDER
                rect(d,(x+c*22,y+r*24,x+c*22+15,y+r*24+15),col,radius=3)
        text(d,(722,437),'W → W + Δ',18,PURPLE,True,mono=True)
        text(d,(722,479),'Small change.',16,WHITE)
        text(d,(722,505),'Fresh source',16,MUTED)
        text(d,(722,529),'each try.',16,MUTED)
        text(d,(386,607),'Closed-form correction. No fine-tuning loop.',16,MUTED)
    elif scene==3:
        text(d,(386,380),'A better token score is not enough.',22,WHITE,True)
        rect(d,(384,428,896,581),'#111722',BORDER,10)
        text(d,(402,447),'GENERATED REPLIES',15,MUTED,True)
        for i,label in enumerate(['Required calls completed','Existing behavior preserved']):
            y=491+i*43
            text(d,(402,y),label,18,WHITE)
            if u>i*.25: check(d,858,y)
        text(d,(386,607),'Select on development. Evaluate held-out.',16,MUTED)
    else:
        text(d,(386,380),'Your model + a reviewed weight patch.',22,WHITE,True)
        rect(d,(384,430,896,578),'#111722',LIME,10)
        text(d,(404,449),'<tool_call>first</tool_call>',18,WHITE,mono=True)
        text(d,(404,486),'<tool_call>second</tool_call>',18,LIME,mono=True)
        text(d,(404,538),'Both calls. Complete turn.',16,LIME,True)
        check(d,850,533)
        text(d,(386,607),'Exact checkpoint. Reversible changes.',16,MUTED)

    # Timing and copy are explicit; first frame already explains the complete product.
    title,desc=SCENES[scene]
    text(d,(42,681),title,15,LIME,True)
    text(d,(42,710),desc,20,WHITE)
    for i in range(5):
        start=40+i*243
        rect(d,(start,763,start+228,766),BORDER,radius=1)
        if i<scene: rect(d,(start,763,start+228,766),LIME,radius=1)
        elif i==scene: rect(d,(start,763,start+max(1,228*u),766),LIME,radius=1)
    text(d,(42,779),'Illustrated workflow · Experimental; small edits do not fix every failure.',13,MUTED)
    return im.resize((W,H),Image.Resampling.LANCZOS)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    # 5 scenes × 3.2 seconds, plus a readable hold on the delivered patch.
    frames=[]
    for scene in range(5):
        for i in range(32):
            rgb=render(scene,i/31)
            frames.append(rgb.quantize(colors=96,method=Image.Quantize.MEDIANCUT,dither=Image.Dither.NONE))
    durations=[100]*len(frames)
    durations[-1]=2100
    frames[0].save(OUT/'dawe-workflow.gif',save_all=True,append_images=frames[1:],
                   duration=durations,loop=0,optimize=True,disposal=1)
    render(4,1).save(OUT/'dawe-workflow.png',optimize=True)
    # Contact sheet for local review; generated outside the repository.
    sheet=Image.new('RGB',(W*2,H*3),BG)
    for i in range(5): sheet.paste(render(i,.75),(i%2*W,i//2*H))
    sheet.resize((1280,1200)).save('/tmp/dawe-storyboard.png')
    print(f"Saved GIF ({(OUT/'dawe-workflow.gif').stat().st_size:,} bytes) and PNG poster to {OUT}")


if __name__=='__main__':
    main()
