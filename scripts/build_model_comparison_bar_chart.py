import matplotlib.pyplot as plt
import numpy as np

models = [
    "Text-ComplEx",
    "Text-RGCN",
    "Early Fusion",
    "Gate Fusion"
]

mrr_mean = [0.34, 0.37, 0.41, 0.465]
mrr_std = [0.006, 0.009, 0.012, 0.028]

x = np.arange(len(models))

plt.figure(figsize=(7,5))

plt.bar(
    x,
    mrr_mean,
    yerr=mrr_std,
    capsize=5
)

plt.xticks(x, models)
plt.ylabel("MRR")
plt.title("Model Comparison on OpenBG-IMG")

plt.tight_layout()
plt.savefig("model_comparison_mrr.png", dpi=300)
plt.show()