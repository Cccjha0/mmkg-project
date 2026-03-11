from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
text = torch.load(PROJECT_ROOT / "data/cache/openbg_img/text_emb.pt")
has_text = torch.load(PROJECT_ROOT / "data/cache/openbg_img/has_text.pt")
print(text.shape, int(has_text.sum()), has_text.shape)
print("text norm sample:", float(text[0].norm()))
