import torch

from src.models.openbg_img_gated_lp import OpenBGImgGatedLP


def build_model(cfg: dict):
    model_name = cfg["model"]["name"]

    if model_name == "openbg_img_gated":
        cache_dir = cfg["dataset"]["cache_dir"]
        d = cfg["embedding"]["d"]
        tr = cfg["training"]
        num_relations = cfg["model"]["num_relations"]
        use_layernorm = cfg["model"].get("use_layernorm", True)
        use_fusion = cfg["model"].get("use_fusion", True)
        use_residual = cfg["model"].get("use_residual", True)
        neg_ratio = tr.get("neg_ratio", 10)
        adv_temperature = tr.get("adv_temperature", 1.0)
        img_dropout = tr.get("img_dropout", 0.0)

        text_emb = torch.load(f"{cache_dir}/text_emb.pt")
        img_emb = torch.load(f"{cache_dir}/img_emb.pt")
        has_img = torch.load(f"{cache_dir}/has_img.pt")

        model = OpenBGImgGatedLP(
            text_emb=text_emb,
            img_emb=img_emb,
            has_img=has_img,
            num_relations=num_relations,
            d=d,
            use_layernorm=use_layernorm,
            neg_ratio=neg_ratio,
            adv_temperature=adv_temperature,
            img_dropout=img_dropout,
            use_fusion=use_fusion,
            use_residual=use_residual,
        )
        num_entities = text_emb.shape[0]
        return model, num_entities

    raise ValueError(f"Unknown model.name: {model_name}")
