import base64
import io
import json
import os

import numpy as np
import torch
import torch.nn as nn

_DIR = os.path.dirname(__file__)
_META = json.load(open(os.path.join(_DIR, "form_model_meta.json")))
FIXED_LEN = _META["fixed_len"]
IN_FEATS = _META["in_feats"]
LI = _META["landmark_indices"]


class FormNet(nn.Module):
    def __init__(self, in_feats=IN_FEATS):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_feats, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.4), nn.Linear(64, 1))

    def forward(self, x):
        return self.head(self.conv(x.transpose(1, 2))).squeeze(1)


_model = None


def _load():
    global _model
    if _model is None:
        with open(os.path.join(_DIR, "form_model_b64.txt")) as f:
            raw = base64.b64decode(f.read())
        m = FormNet()
        m.load_state_dict(torch.load(io.BytesIO(raw), map_location="cpu"))
        m.eval()
        _model = m
    return _model


def _normalize(frame):
    xyz = np.asarray(frame, dtype=np.float32)[:, :3]
    hip = (xyz[LI["l_hip"]] + xyz[LI["r_hip"]]) / 2
    sho = (xyz[LI["l_sho"]] + xyz[LI["r_sho"]]) / 2
    torso = np.linalg.norm(sho - hip) + 1e-6
    return ((xyz - hip) / torso).reshape(-1)


def _resample(seq, n):
    t = seq.shape[0]
    if t == n:
        return seq
    idx = np.linspace(0, t - 1, n)
    lo = np.floor(idx).astype(int)
    hi = np.ceil(idx).astype(int)
    w = (idx - lo)[:, None].astype(np.float32)
    return seq[lo] * (1 - w) + seq[hi] * w


def predict(frames):
    feats = _resample(np.stack([_normalize(f) for f in frames]), FIXED_LEN)
    x = torch.tensor(feats[None], dtype=torch.float32)
    with torch.no_grad():
        logit = _load()(x).item()
    prob_good = float(1 / (1 + np.exp(-logit)))
    return ("good" if logit > 0 else "bad"), prob_good