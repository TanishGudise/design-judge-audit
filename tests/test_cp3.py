"""CP3 (plan A7): new exploit renders keep the canvas and match their controls outside the edit."""
import io

import numpy as np
import pytest
from PIL import Image

from dja import cli, render


@pytest.fixture(scope="module")
def src(tmp_path_factory):
    rng = np.random.default_rng(1)
    arr = rng.integers(0, 255, (900, 1440, 3), dtype=np.uint8)     # VDA screenshot size
    p = tmp_path_factory.mktemp("img") / "x.png"
    Image.fromarray(arr).save(p)
    return p


def _img(r):
    return np.array(Image.open(io.BytesIO(r.data)).convert("RGB")).astype(int)


@pytest.mark.parametrize("cond", render.CP3_CONDITIONS)
def test_cp3_conditions_keep_canonical_size(src, cond):
    base = _img(render.render(src, "vda", "base"))
    assert base.shape == (700, 1120, 3)
    assert _img(render.render(src, "vda", cond)).shape == base.shape


def test_badge_and_control_differ_only_inside_the_footprint(src):
    base = _img(render.render(src, "vda", "base"))
    x1, k1 = _img(render.render(src, "vda", "X1")), _img(render.render(src, "vda", "K1"))
    H, W = base.shape[:2]
    cx, cy, R = render.badge_geometry(W, H)
    yy, xx = np.mgrid[0:H, 0:W]
    outside = (xx - cx) ** 2 + (yy - cy) ** 2 > (R + 2) ** 2
    for im in (x1, k1):
        assert np.array_equal(im[outside], base[outside])
    assert not np.array_equal(x1, k1)


def test_laptop_matches_plain_control_in_design_region(src):
    x2, k2 = _img(render.render(src, "vda", "X2")), _img(render.render(src, "vda", "K2"))
    H, W = x2.shape[:2]
    dw, dh, b, base_h, x, top = render.laptop_geometry(W, H)
    y0 = top + b
    assert np.array_equal(x2[y0:y0 + dh, x:x + dw], k2[y0:y0 + dh, x:x + dw])
    assert (k2[:y0] == 255).all() and (k2[y0 + dh:] == 255).all(), "control is plain white outside the design"
    assert x2[:y0 - b].mean() < 100, "exploit backdrop is dark"


@pytest.mark.parametrize("bg", [255, 0])
def test_hero_text_colour_rule_and_band(tmp_path, bg):
    p = tmp_path / "flat.png"
    Image.new("RGB", (1440, 900), (bg, bg, bg)).save(p)
    base = _img(render.render(p, "vda", "base"))
    x3, k3 = _img(render.render(p, "vda", "X3")), _img(render.render(p, "vda", "K3"))
    H, W = base.shape[:2]
    font, cx, cy, th, fill, stroke, box = render.hero_layout(Image.fromarray(base.astype(np.uint8)))
    assert fill == ((17, 17, 17) if bg == 255 else (255, 255, 255))
    rows = np.where((x3 != base).any(axis=(1, 2)) | (k3 != base).any(axis=(1, 2)))[0]
    assert rows.min() >= cy - th and rows.max() <= cy + th, "text stays in one band at the pre-registered centre"
    cols_x = np.where((x3 != base).any(axis=(0, 2)))[0]
    cols_k = np.where((k3 != base).any(axis=(0, 2)))[0]
    assert abs((cols_x.min() + cols_x.max()) / 2 - cx) < 4 and abs((cols_k.min() + cols_k.max()) / 2 - cx) < 4
    assert abs((cols_x.max() - cols_x.min()) - (cols_k.max() - cols_k.min())) < 0.08 * W, "similar text width"


def test_watermark_same_grid_and_opacity(tmp_path):
    p = tmp_path / "white.png"
    Image.new("RGB", (1440, 900), (255, 255, 255)).save(p)
    x4, k4 = _img(render.render(p, "vda", "X4")), _img(render.render(p, "vda", "K4"))
    for im in (x4, k4):
        assert im.min() >= 255 - round(0.16 * 127) - 1, "never darker than grey at 16% opacity on white"
    ink_x, ink_k = (x4 < 250).mean(), (k4 < 250).mean()
    assert 0.5 < ink_x / ink_k < 2.0, "similar ink coverage"
    assert not np.array_equal(x4, k4)


def test_cp3_manifest_shape_and_leak_tokens():
    from dja import stages, stages_cp3
    ds = stages.load_all() if __import__("dja.data", fromlist=["x"]).vda_available() else None
    banned = cli.banned_tokens(ds) if ds else set()
    for tok in ("X1", "X4", "K1", "K4", "F3"):
        assert not ds or tok in banned
    if ds is None:
        pytest.skip("VDA not on disk")
    rows = stages_cp3.cp3_rows(ds)
    assert len(rows) == 24 * 135
    for r in rows:
        e = r["test"].split(".")[-1]
        if r["test"].split(".")[-2] == "ctl":
            assert r["cond_a"] == r["cond_b"] == e
        else:
            x, y = r["pair"]
            ctl = render.CP3_CONTROL[e]
            cond = {r["a"]: r["cond_a"], r["b"]: r["cond_b"]}
            assert cond[x] == ctl and cond[y] == e, "the human loser gets the exploit, the winner the control"
        assert r["note_a"] is None and r["note_b"] is None
