import torch

has_img = torch.load("cache/openbg_img/has_img.pt")
img = torch.load("cache/openbg_img/img_emb.pt")
raw = torch.load("cache/openbg_img/img_emb_raw.pt")

print("has_img.sum =", int(has_img.sum()))
print("img_emb shape =", tuple(img.shape))
print("raw shape =", tuple(raw.shape))

idx_no = int((~has_img).nonzero()[0])
idx_yes = int((has_img).nonzero()[0])

print("no-img raw norm =", float(raw[idx_no].norm()))
print("yes-img raw norm =", float(raw[idx_yes].norm()))