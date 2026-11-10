"""Blind-rating pairs for a human rater: matched control vs exploit edit of the same design, left/right randomized,
unlabeled. Confirms (directionally, n stated) that the edits are not design improvements.

Writes (gitignored, third-party images stay local):
  data/derived/blind/blind_pairs.html   the page to rate (self-contained, radio buttons, copyable answers)
  data/derived/blind/KEY_do_not_open.json  which side holds the edit (read only by the analysis)
"""
from __future__ import annotations

import base64
import html
import json

import numpy as np

from . import config, experiments as X, render, stages

EDITS = ("E1", "E2", "E3", "E4a", "E5", "R4")
PER_EDIT = 5
OUT = config.DERIVED / "blind"


def main():
    ds = stages.load_all()
    web = "vda" if "vda" in ds else "mf"
    rng = np.random.default_rng(config.SEED + 7)
    pool = []
    for name in (web, "gde"):
        items = ds[name][0]
        ids = sorted(i for i, it in items.items() if it.split == "test" and (name != "gde" or "/org/" in i))
        pool.append((name, ids))
    rows = []
    for e in EDITS:
        ctl = X.CONTROL_OF[e]
        for k in range(PER_EDIT):
            name, ids = pool[k % 2]
            iid = ids[int(rng.integers(len(ids)))]
            it = ds[name][0][iid]
            a = render.render_cached(iid, it.path, name, ctl)
            b = render.render_cached(iid, it.path, name, e)
            edit_left = bool(rng.integers(2))
            rows.append({"dataset": name, "item": iid, "edit": e, "control": ctl, "edit_side": "L" if edit_left else "R",
                         "L": b.path if edit_left else a.path, "R": a.path if edit_left else b.path})
    order = rng.permutation(len(rows))
    rows = [rows[i] for i in order]
    for n, r in enumerate(rows, 1):
        r["pair"] = n
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "KEY_do_not_open.json").write_text(json.dumps(
        [{k: r[k] for k in ("pair", "dataset", "item", "edit", "control", "edit_side")} for r in rows], indent=1))

    def img(path):
        mt = "image/png" if path.endswith(".png") else "image/jpeg"
        return f"data:{mt};base64," + base64.b64encode(open(path, "rb").read()).decode()

    parts = ["""<!doctype html><meta charset="utf-8"><title>Blind design pairs</title>
<style>body{font:15px system-ui;margin:16px;max-width:1300px}.p{border-top:1px solid #ccc;padding:14px 0}
.row{display:flex;gap:16px;align-items:flex-start}.row figure{flex:1;margin:0}.row img{width:100%;border:1px solid #ddd}
label{margin-right:18px}textarea{width:100%;height:120px}</style>
<h2>Blind design pairs</h2>
<p>For each pair, pick the better <b>design</b>, or "same" if you can't tell them apart as designs.
Both images show the same underlying design; judge it as you would judge the design itself.
When done, press "Show answers" and paste the text back to Claude.</p>"""]
    for r in rows:
        n = r["pair"]
        parts.append(f"""<div class="p"><b>Pair {n}</b><div class="row">
<figure><img src="{img(r['L'])}"><figcaption>Left</figcaption></figure>
<figure><img src="{img(r['R'])}"><figcaption>Right</figcaption></figure></div>
<label><input type="radio" name="p{n}" value="L"> Left better</label>
<label><input type="radio" name="p{n}" value="R"> Right better</label>
<label><input type="radio" name="p{n}" value="S"> Same</label></div>""")
    parts.append(f"""<button onclick="let o=[];for(let i=1;i<={len(rows)};i++){{let c=document.querySelector('input[name=p'+i+']:checked');o.push(i+','+(c?c.value:''))}};document.getElementById('a').value='pair,choice\\n'+o.join('\\n')">Show answers</button>
<textarea id="a" readonly></textarea>""")
    (OUT / "blind_pairs.html").write_text("\n".join(parts))
    print(f"[blind] {len(rows)} pairs -> {OUT / 'blind_pairs.html'}")


if __name__ == "__main__":
    main()
