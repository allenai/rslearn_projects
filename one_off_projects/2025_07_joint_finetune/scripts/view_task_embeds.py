"""Visualize task embeddings from a checkpoint."""

import argparse
import yaml
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--project", default="helios_finetune_cosine_lr")
parser.add_argument("--ckpt", default="last.ckpt")
parser.add_argument("--out_dir", default="../data")
args = parser.parse_args()

# load data
path = f"/weka/dfive-default/rslearn-eai/projects/{args.project}/{args.model}/checkpoints/{args.ckpt}"
cfg_path = f"/weka/dfive-default/rslearn-eai/projects/{args.project}/{args.model}/checkpoints/config.yaml"
with open(cfg_path, "r") as f:
    cfg = yaml.safe_load(f)
    d2t = cfg['model']['init_args']['model']['init_args']['decoder_to_target']
    idx_to_target = []
    for decoder, targets in d2t.items():
        for target in targets:
            idx_to_target.append(target)
print(f"Order: {idx_to_target}")

checkpoint = torch.load(path)
task_embeddings = checkpoint['state_dict']['model.trunk.task_embedding.embed.weight']
embeds = task_embeddings.detach().cpu().numpy()
print(f"Task embeddings shape: {embeds.shape}")

# calculate cosine similarities
cos_sim_matrix = cosine_similarity(embeds)
print("Cosine similarity matrix:")
print(cos_sim_matrix)

# plot pca
fig, axes = plt.subplots(1, 1, figsize=(10, 10))

pca = PCA(n_components=2)
pca_embeddings = pca.fit_transform(embeds)
axes.scatter(pca_embeddings[:, 0], pca_embeddings[:, 1], s=100, alpha=0.7)
for i, (x, y) in enumerate(pca_embeddings):
    axes.annotate(f'{idx_to_target[i]}', (x, y), xytext=(5, 5), textcoords='offset points')
axes.set_title(f'PCA Visualization (explained variance: {pca.explained_variance_ratio_.sum():.3f})')
axes.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.3f})')
axes.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.3f})')

plt.tight_layout()
plt.savefig(f"{args.out_dir}/task_embeds__{args.model}.png")
