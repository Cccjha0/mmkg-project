# OpenBG-Link Prediction

A systematic study on link prediction in commodity knowledge graphs,
focusing on textual and multimodal (text + image) fusion strategies.

This project is developed for undergraduate thesis research.

---

## 📌 Project Overview

We conduct a systematic comparison of different modeling strategies for link prediction on:

- **OpenBG500L** (Text-based Knowledge Graph)
- **OpenBG-IMG** (Multimodal Knowledge Graph with text and images)

Our goal is to investigate:

- Whether textual information improves KGE
- Whether multimodal fusion improves performance
- Whether dynamic fusion (Gated) outperforms static fusion (Early)

---

## 📂 Repository Structure

```
openbg-lp/
 │
 ├── datasets/
 │   ├── openbg500/
 │   │   └── raw/
 │   └── openbg_img/
 │       └── raw/
 │
 ├── cache/               # Precomputed embeddings (NOT tracked by git)
 ├── configs/             # Experiment configs
 ├── scripts/             # Entry scripts
 ├── src/
 │   ├── data/
 │   ├── eval/
 │   ├── models/
 │   ├── train/
 │   └── utils/
 │
 ├── outputs/             # Experiment outputs (NOT tracked by git)
 ├── requirements.txt
 └── README.md
```

---

## 📊 Datasets

⚠️ Datasets are **NOT included** in this repository due to size limitations.

Please download the datasets separately and place them in the following directories:

```
datasets/openbg500/raw/
datasets/openbg_img/raw/
```

### Required Files

#### OpenBG500
- OpenBG500-L_train.tsv
- OpenBG500-L_dev.tsv
- OpenBG500-L_test.tsv
- OpenBG500-L_entity2text.tsv
- OpenBG500-L_relation2text.tsv

#### OpenBG-IMG
- OpenBG-IMG_train.tsv
- OpenBG-IMG_dev.tsv
- OpenBG-IMG_test.tsv
- OpenBG-IMG_entity2text.tsv
- OpenBG-IMG_relation2text.tsv
- OpenBG-IMG_images/ (ent_xxxxxx/image_0.jpg)

---

## 🛠 Environment Setup

### 1️⃣ Create virtual environment (recommended)

```bash
python -m venv venv
venv\Scripts\activate
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## ⚙️ Step 1: Build Feature Cache (OpenBG-IMG)

Before training multimodal models, precompute embeddings:

```bash
python scripts/build_cache_openbg_img.py --config configs/openbg_img_gated.yaml
```

This will generate:

```
cache/openbg_img/
text_emb.pt
img_emb.pt
has_text.pt
has_img.pt
```

---

## 🚀 Step 2: Train Models

### 🔹 OpenBG500

Text-KGE:

```bash
python scripts/run_openbg500_textkge.py --config configs/openbg500_textkge.yaml
```

Text-RGCN:

```bash
python scripts/run_openbg500_textrgcn.py --config configs/openbg500_textrgcn.yaml
```

---

### 🔹 OpenBG-IMG

Early Fusion:

```bash
python scripts/run_openbg_img_early.py --config configs/openbg_img_early.yaml
```

Gated Fusion:

```bash
python scripts/run_openbg_img_gated.py --config configs/openbg_img_gated.yaml
```

---

## 📈 Evaluation Metrics

All models are evaluated using:

- Filtered MRR
- Hits@1
- Hits@3
- Hits@10

Evaluation follows the standard filtered ranking protocol.

---

## 📊 Outputs

Experiment outputs are stored under:

```
outputs/
openbg_img/
openbg500/
```

Each run contains:

- metrics.csv
- best.ckpt
- predictions_topk.jsonl
- gate_stats.csv (for Gated model)

---

## 🔬 Research Contributions

This project compares:

- Structure-only KGE
- Text-enhanced KGE
- Early Fusion (static multimodal fusion)
- Gated Fusion (dynamic multimodal fusion)

We analyze both quantitative performance and fusion interpretability.

---

## 👥 Team

- Member A – Text-KGE
- Member B – Text-RGCN
- Member C – Early Fusion
- Member D – Gated Fusion

---

## 📌 Reproducibility

All experiments are controlled by config files under `configs/`.

To reproduce results:

1. Place datasets correctly.
2. Build cache.
3. Run training scripts.
4. Check outputs under `outputs/`.

---

## 📄 License

For academic research use only.