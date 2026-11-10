"""Non-LLM arms on frozen UIClip image embeddings ($0, CPU).

J0  UIClip zero-shot quality score (the model card's normalized "well-designed" vs "poor design").
J4  a small probe: a linear classifier on frozen embeddings trained on the BUILD
    split's human judgments and scored on the held-out TEST split.
    - vote data (VDA): Bradley-Terry logistic regression on embedding differences, soft labels =
      vote share (each pair enters as two weighted rows, both orientations), no intercept.
    - rating data (GDE, MF): ridge regression on the human mean score.
    Regularization is picked by grouped cross-validation inside the build split only.

Weights: biglab/uiclip_jitteredwebsites-2-224-paraphrased_webpairs_humanpairs (MIT, 605,156,676 bytes);
processor: openai/clip-vit-base-patch32 (preprocessing + tokenizer files only).
"""
from __future__ import annotations

import io
from functools import lru_cache

import numpy as np

from . import config

UICLIP = "biglab/uiclip_jitteredwebsites-2-224-paraphrased_webpairs_humanpairs"
UICLIP_REV = "505d4731b09263c07437a9022d23c800274da56b"
PROCESSOR = "openai/clip-vit-base-patch32"
PROCESSOR_REV = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
IMG = 224
LOGIT_SCALE = 100
DESCRIPTION = {"vda": "landing page of a web app", "mf": "homepage of a commercial bank",
               "gde": "promotional banner design"}


@lru_cache(maxsize=1)
def _model():
    import torch
    from transformers import CLIPModel, CLIPProcessor
    torch.set_num_threads(4)
    m = CLIPModel.from_pretrained(UICLIP, revision=UICLIP_REV).eval()
    p = CLIPProcessor.from_pretrained(PROCESSOR, revision=PROCESSOR_REV)
    return m, p


def _feat(out):
    """transformers 5 returns a ModelOutput whose pooler_output is the projected embedding;
    older versions return the tensor itself."""
    return getattr(out, "pooler_output", out)


def _preresize(im, size):
    ar = im.width / im.height
    return im.resize((int(ar * size), size)) if ar > 1 else im.resize((size, int(size / ar)))


def _windows(im, size=IMG):
    """Model card's sliding window: resize short side to 224, square crops along the long side."""
    im = _preresize(im.convert("RGB"), size)
    w, h = im.size
    sq = min(w, h)
    longer = max(w, h)
    steps = (longer + sq - 1) // sq
    step = (longer - sq) // (steps - 1) if steps > 1 else sq
    crops = []
    for y in range(0, h - sq + 1, step if h > w else sq):
        for x in range(0, w - sq + 1, step if w > h else sq):
            crops.append(im.crop((x, y, x + sq, y + sq)))
    return crops


def embed(image_bytes_list: list[bytes]) -> np.ndarray:
    import torch
    from PIL import Image
    m, p = _model()
    out = []
    for b in image_bytes_list:
        crops = _windows(Image.open(io.BytesIO(b)))
        with torch.no_grad():
            f = _feat(m.get_image_features(**p(images=crops, return_tensors="pt")))
        e = f.mean(0)
        out.append((e / e.norm()).numpy())
    return np.stack(out)


def zero_shot(embs: np.ndarray, dataset: str) -> np.ndarray:
    import torch
    m, p = _model()
    desc = DESCRIPTION[dataset]
    texts = [f"ui screenshot. well-designed. {desc}", f"ui screenshot. poor design. {desc}"]
    with torch.no_grad():
        t = _feat(m.get_text_features(**p(text=texts, return_tensors="pt", padding=True)))
    t = (t / t.norm(dim=-1, keepdim=True)).numpy()
    logits = LOGIT_SCALE * embs @ t.T
    logits -= logits.max(1, keepdims=True)
    pr = np.exp(logits)
    return pr[:, 0] / pr.sum(1)


# ---------------------------------------------------------------------------------------------
# J4 training
# ---------------------------------------------------------------------------------------------
def fit_bt(emb: dict, pairs, groups: dict, Cs=(0.01, 0.03, 0.1, 0.3, 1, 3, 10)):
    """Bradley-Terry logistic on embedding differences with soft labels; C by leave-one-group-out
    log loss over build pairs. Returns weight vector w (score = w . e)."""
    from sklearn.linear_model import LogisticRegression

    def design(ps):
        X, y, sw = [], [], []
        for p in ps:
            d = emb[p.a] - emb[p.b]
            for sign, share in ((1, p.share_a), (-1, 1 - p.share_a)):
                X += [sign * d, sign * d]
                y += [1, 0]
                sw += [share, 1 - share]
        return np.array(X), np.array(y), np.array(sw)

    gs = sorted({groups[p.a] for p in pairs} | {groups[p.b] for p in pairs})
    best, best_c = np.inf, Cs[0]
    for C in Cs:
        loss = 0.0
        for g in gs:
            tr = [p for p in pairs if groups[p.a] != g and groups[p.b] != g]
            te = [p for p in pairs if groups[p.a] == g or groups[p.b] == g]
            if not tr or not te:
                continue
            X, y, sw = design(tr)
            clf = LogisticRegression(C=C, fit_intercept=False, max_iter=5000).fit(X, y, sample_weight=sw)
            Xt, yt, swt = design(te)
            pr = np.clip(clf.predict_proba(Xt)[:, 1], 1e-6, 1 - 1e-6)
            loss += -(swt * (yt * np.log(pr) + (1 - yt) * np.log(1 - pr))).sum()
        if loss < best:
            best, best_c = loss, C
    X, y, sw = design(pairs)
    clf = LogisticRegression(C=best_c, fit_intercept=False, max_iter=5000).fit(X, y, sample_weight=sw)
    return clf.coef_[0], {"C": best_c, "cv_logloss": best}


def fit_ridge(emb: dict, target: dict, groups: dict, alphas=(0.1, 0.3, 1, 3, 10, 30, 100)):
    from sklearn.linear_model import Ridge
    items = [i for i in target if i in emb]
    gs = sorted({groups[i] for i in items})
    rng = np.random.default_rng(config.SEED)
    folds = {g: k % 5 for k, g in enumerate(rng.permutation(gs))}
    best, best_a = np.inf, alphas[0]
    for a in alphas:
        err = 0.0
        for f in range(5):
            tr = [i for i in items if folds[groups[i]] != f]
            te = [i for i in items if folds[groups[i]] == f]
            m = Ridge(alpha=a).fit(np.stack([emb[i] for i in tr]), [target[i] for i in tr])
            err += ((m.predict(np.stack([emb[i] for i in te])) - np.array([target[i] for i in te])) ** 2).sum()
        if err < best:
            best, best_a = err, a
    m = Ridge(alpha=best_a).fit(np.stack([emb[i] for i in items]), [target[i] for i in items])
    return (m.coef_, float(m.intercept_)), {"alpha": best_a, "cv_sse": best}
