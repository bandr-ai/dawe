"""Render the compact DAWE workflow illustration; no model calls.

Run: uv run --with pillow python scripts/render_workflow.py
Fonts: Arial on macOS, DejaVu Sans on Linux, or Pillow's default.
"""
from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs' / 'assets'
W, H, SCALE = 900, 360, 3
EXPORT_SIZE = (1800, 720)
BG, INK, MUTED = '#ffffff', '#202b32', '#6c7780'
LINE, PALE = '#dce2e6', '#f0f3f5'
BLUE, ORANGE = '#246bd6', '#ef9638'
TEAL, PURPLE = '#119a92', '#8763ce'
FONT_ROOTS = [Path('/System/Library/Fonts/Supplemental'), Path('/usr/share/fonts/truetype/dejavu')]


def font(size, bold=False):
    names = ['Arial Bold.ttf', 'DejaVuSans-Bold.ttf'] if bold else ['Arial.ttf', 'DejaVuSans.ttf']
    for base in FONT_ROOTS:
        for name in names:
            if (base / name).exists():
                return ImageFont.truetype(str(base / name), size * SCALE)
    return ImageFont.load_default(size=size * SCALE)


def text(d, xy, value, size=16, color=INK, bold=False):
    d.text((xy[0]*SCALE, xy[1]*SCALE), value, font=font(size, bold), fill=color)


def line(d, points, color=LINE, width=1):
    d.line([(round(x*SCALE), round(y*SCALE)) for x, y in points], fill=color, width=width*SCALE)


def rect(d, box, fill, outline=None, radius=0):
    d.rounded_rectangle(tuple(round(v*SCALE) for v in box), radius=radius*SCALE,
                        fill=fill, outline=outline, width=SCALE)


def mix(a, b, t):
    aa, bb = (tuple(bytes.fromhex(c[1:])) for c in (a, b))
    return tuple(round(x+(y-x)*t) for x, y in zip(aa, bb))


def check(d, x, y, active):
    color = TEAL if active else LINE
    line(d, [(x,y+5), (x+5,y+10), (x+15,y)], color, 2)


def grid(d, x, y, edited=0, sparse=False, scan=None):
    for r in range(7):
        for c in range(16):
            changed = r == 3 and c < round(16*edited)
            value = (math.sin(r*19+c*7)+1)/2
            color = PALE if sparse else mix('#e8effa', '#92b8e5', value*.65)
            if changed:
                color = mix('#ffffff', [BLUE, TEAL, PURPLE, ORANGE][c % 4], .45+value*.5)
            rect(d, (x+c*11,y+r*11,x+c*11+10,y+r*11+10), color)
    if scan is not None:
        sy = y + min(6, int(scan*7))*11
        rect(d, (x-4,sy-3,x+179,sy+13), None, BLUE, 2)


def render(stage, u):
    im = Image.new('RGB', (W*SCALE,H*SCALE), BG)
    d = ImageDraw.Draw(im)
    text(d,(30,22),'DAWE',20,BLUE,True)
    text(d,(112,22),'Harness changed? Patch your model.',24,INK,True)
    line(d,[(30,68),(870,68)])

    for x, letter, label, color in [(30,'a','Your data + model',BLUE),(333,'b','Astra/Fable',PURPLE),(638,'c','Tested patch',TEAL)]:
        rect(d,(x,85,x+20,108),color,radius=5)
        text(d,(x+5,87),letter,15,BG,True)
        text(d,(x+29,86),label,18,INK,True)

    # Data sits above the user's checkpoint; the editing agent has its own panel.
    rect(d,(53,133,119,201),'#eef4ff','#bad0f1',3)
    for i, end in enumerate([105,97,108,90]):
        line(d,[(64,148+i*11),(end,148+i*11)],MUTED,2)
    text(d,(132,148),'Failures',15,MUTED)
    for r in range(3):
        for c in range(5):
            rect(d,(54+c*12,220+r*12,64+c*12,230+r*12),mix(BG,[BLUE,PURPLE,TEAL][(r+c)%3],.3+((r+c)%3)*.15))
    text(d,(132,225),'Model',15,MUTED)

    edited = 0 if stage == 0 else min(1,u*1.5) if stage == 1 else 1
    grid(d,356,151,edited,scan=u if stage==0 else None)
    text(d,(356,245),['Investigate','Edit a few weights','Test the change'][stage],15,PURPLE,True)

    grid(d,661,151,edited if stage==2 else 0,sparse=True)
    for y, label, active in [(244,'New harness',stage==2 and u>.15),
                              (270,'Existing behavior',stage==2 and u>.45)]:
        check(d,661,y+2,active)
        text(d,(684,y),label,15,INK if active else MUTED)

    for start,end in [(266,309),(571,614)]:
        line(d,[(start,192),(end,192)],MUTED,1)
        line(d,[(end-6,187),(end,192),(end-6,197)],MUTED,1)
    line(d,[(30,310),(870,310)])
    text(d,(30,326),'Bring the data. Astra/Fable delivers the patch.',15,INK)
    text(d,(775,328),'Illustration',12,MUTED)
    return im.resize(EXPORT_SIZE,Image.Resampling.LANCZOS)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    frames = [render(stage,i/19).quantize(colors=192,method=Image.Quantize.MEDIANCUT,
              dither=Image.Dither.NONE) for stage in range(3) for i in range(20)]
    durations = [100]*len(frames)
    durations[-1] = 1800
    frames[0].save(OUT/'dawe-workflow.gif',save_all=True,append_images=frames[1:],
                   duration=durations,loop=0,optimize=True,disposal=1)
    render(2,1).save(OUT/'dawe-workflow.png',optimize=True)
    sheet = Image.new('RGB',(EXPORT_SIZE[0],EXPORT_SIZE[1]*3),BG)
    for stage in range(3):
        sheet.paste(render(stage,.8),(0,stage*EXPORT_SIZE[1]))
    sheet.resize((900,1080),Image.Resampling.LANCZOS).save('/tmp/dawe-storyboard.png')
    print(f"Saved {EXPORT_SIZE[0]}×{EXPORT_SIZE[1]} GIF ({(OUT/'dawe-workflow.gif').stat().st_size:,} bytes) and PNG")


if __name__ == '__main__':
    main()
