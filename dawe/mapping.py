"""Read a step-4 map (runs/map_<a>__<b>.json) and answer: is the harness change concentrated?"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path


def _where(name: str) -> tuple[int, str]:
    m = re.search(r"layers\.(\d+)\.(.+)\.weight$", name)
    if not m:
        return -1, name
    kind = m.group(2).split(".")[-1]
    return int(m.group(1)), kind


def _spearman(x: list[float], y: list[float]) -> float:
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for k, i in enumerate(order):
            r[i] = k
        return r

    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    return cov / math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))


def summarize(path: Path) -> str:
    m = json.loads(path.read_text())
    a, b = m["a"], m["b"]
    ga, gb = m[a]["grad_norm"], m[b]["grad_norm"]
    names = sorted(set(ga) & set(gb))
    out = [f"== {path.name}  ({m['n_pairs']} pairs, model {m['run']})", ""]

    # 1. how surprised is the model to continue after a non-final call
    out.append("Surprise at 'another call follows' (avg -log p of the next gold token; higher = more surprised)")
    for h in (a, b):
        out.append(f"  {h:10} continue {m[h]['nll_continue']:.3f}   stop {m[h]['nll_stop']:.3f}   "
                   f"(points: {m[h]['n_continue']} continue, {m[h]['n_stop']} stop)")

    # 2. gradient concentration
    tot_a = sum(ga[n] ** 2 for n in names)
    tot_b = sum(gb[n] ** 2 for n in names)
    out += ["", f"Gradient size: total |grad| on {a} replies = {math.sqrt(tot_a):.3f}, on {b} replies = {math.sqrt(tot_b):.3f}"]
    share = sorted(((gb[n] ** 2 / tot_b, n) for n in names), reverse=True)
    for k in (1, 5, 10, 25):
        out.append(f"  top {k:>2} of {len(names)} matrices hold {sum(s for s, _ in share[:k]) * 100:5.1f}% of the {b} gradient")
    out.append(f"  top matrices for {b}:")
    for s, n in share[:10]:
        L, kind = _where(n)
        out.append(f"    layer {L:>2} {kind:14} {s * 100:5.1f}%   (vs {a}: {ga[n] ** 2 / tot_a * 100:4.1f}%)")

    by_layer = defaultdict(float)
    by_kind = defaultdict(float)
    for n in names:
        L, kind = _where(n)
        by_layer[L] += gb[n] ** 2 / tot_b
        by_kind[kind] += gb[n] ** 2 / tot_b
    out.append(f"  {b} gradient by layer (%): " + " ".join(f"{L}:{v * 100:.0f}" for L, v in sorted(by_layer.items()) if L >= 0))
    out.append(f"  {b} gradient by kind (%):  " + ", ".join(f"{k} {v * 100:.0f}" for k, v in sorted(by_kind.items(), key=lambda x: -x[1])))

    # 3. does the model internally separate 'continue' from 'stop'?
    sa, sb = m[a].get("separation"), m[b].get("separation")
    if sa and sb:
        out += ["", "Continue-vs-stop signal in hidden states, per layer (held-out accuracy; cosine of the two directions)"]
        out.append(f"  {'layer':>5} {a + ' acc':>12} {b + ' acc':>16} {'cos(dir)':>9}")
        for la, lb in zip(sa, sb):
            da, db = la["dir"], lb["dir"]
            cos = sum(x * y for x, y in zip(da, db)) / (math.sqrt(sum(x * x for x in da)) * math.sqrt(sum(y * y for y in db)) + 1e-9)
            if la["layer"] % 4 == 0 or la["layer"] == len(sa) - 1:
                out.append(f"  {la['layer']:>5} {la['heldout_acc']:>12.2f} {lb['heldout_acc']:>16.2f} {cos:>9.2f}")
    return "\n".join(out)


def compare(p1: Path, p2: Path) -> str:
    """Do two harness changes point at the same weights?"""
    m1, m2 = json.loads(p1.read_text()), json.loads(p2.read_text())
    g1, g2 = m1[m1["b"]]["grad_norm"], m2[m2["b"]]["grad_norm"]
    names = sorted(set(g1) & set(g2))
    rho = _spearman([g1[n] for n in names], [g2[n] for n in names])
    t1 = {n for _, n in sorted(((g1[n], n) for n in names), reverse=True)[:25]}
    t2 = {n for _, n in sorted(((g2[n], n) for n in names), reverse=True)[:25]}
    return (f"{m1['b']} vs {m2['b']}: rank correlation of per-matrix gradient size {rho:.2f}; "
            f"top-25 matrices shared: {len(t1 & t2)}/25")
