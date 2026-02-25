import torch
text = torch.load("cache/openbg_img/text_emb.pt")
has_text = torch.load("cache/openbg_img/has_text.pt")
print(text.shape, int(has_text.sum()), has_text.shape)
print("text norm sample:", float(text[0].norm()))