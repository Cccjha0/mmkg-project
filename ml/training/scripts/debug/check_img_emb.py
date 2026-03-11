from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]

has_img = torch.load(PROJECT_ROOT / "data/cache/openbg_img/has_img.pt")
img = torch.load(PROJECT_ROOT / "data/cache/openbg_img/img_emb.pt")
raw = torch.load(PROJECT_ROOT / "data/cache/openbg_img/img_emb_raw.pt")

print("has_img.sum =", int(has_img.sum()))
print("img_emb shape =", tuple(img.shape))
print("raw shape =", tuple(raw.shape))

idx_no = int((~has_img).nonzero()[0])
idx_yes = int((has_img).nonzero()[0])

print("no-img raw norm =", float(raw[idx_no].norm()))
print("yes-img raw norm =", float(raw[idx_yes].norm()))
