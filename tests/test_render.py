import io

import numpy as np
import pytest
from PIL import Image

from dja import render


@pytest.fixture(scope="module")
def src(tmp_path_factory):
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 255, (450, 720, 3), dtype=np.uint8)
    p = tmp_path_factory.mktemp("img") / "x.png"
    Image.fromarray(arr).save(p)
    return p


def _img(r):
    return np.array(Image.open(io.BytesIO(r.data)).convert("RGB"))


@pytest.mark.parametrize("ds", ["gde", "vda", "mf"])
@pytest.mark.parametrize("cond", ["E1", "E3", "E4a", "E5", "R4"])
def test_strip_edits_match_control_geometry(src, ds, cond):
    c, e = render.render(src, ds, "C1"), render.render(src, ds, cond)
    a, b = _img(c), _img(e)
    assert a.shape == b.shape
    base = _img(render.render(src, ds, "base"))
    H = base.shape[0]
    assert np.array_equal(a[:H], b[:H]), "design region must be identical; only the strip differs"
    assert np.array_equal(a[:H], base)
    assert not np.array_equal(a[H:], b[H:]), "edit must put text in the strip"
    strip = a[H + 1:]
    assert (strip == 255).all(), "control strip is blank"


@pytest.mark.parametrize("ds", ["gde", "vda", "mf"])
def test_mockup_matches_plain_control_in_design_region(src, ds):
    c, e = _img(render.render(src, ds, "C2")), _img(render.render(src, ds, "E2"))
    assert c.shape == e.shape
    W, H = c.shape[1], c.shape[0]
    dw, dh, bar, x, top = render._mockup_geometry(W, H)
    assert np.array_equal(c[top + bar: top + bar + dh, x: x + dw], e[top + bar: top + bar + dh, x: x + dw])
    assert not np.array_equal(c, e)


def test_formats_and_determinism(src):
    for cond, mt in [("jpeg40", "image/jpeg"), ("webp", "image/webp"), ("half", "image/png"), ("pad10", "image/png")]:
        r1, r2 = render.render(src, "vda", cond), render.render(src, "vda", cond)
        assert r1.media_type == mt
        assert r1.data == r2.data
    base = render.render(src, "vda", "base")
    assert base.size == (1120, 700)
    assert render.render(src, "vda", "half").size == (560, 350)


def test_render_cached_reverifies(tmp_path, src, monkeypatch):
    from dja import config
    monkeypatch.setattr(config, "DERIVED", tmp_path)
    render._MEMO.clear()
    r = render.render_cached("t/x", str(src), "vda", "E1")
    assert r.path and r.sha256 and r.data == b""
    assert render.render_cached("t/x", str(src), "vda", "E1").sha256 == r.sha256
