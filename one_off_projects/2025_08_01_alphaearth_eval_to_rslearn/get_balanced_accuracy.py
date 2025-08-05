import argparse
import json
import random

import numpy as np
import torch
from rslearn.dataset import Dataset
from sklearn.metrics import balanced_accuracy_score
from torch import nn
from upath import UPath


def run_knn_for_k(
    train_embeddings: torch.Tensor,
    train_labels: torch.Tensor,
    test_embeddings: torch.Tensor,
    num_classes: int,
    k: int,
) -> torch.Tensor:
    device = torch.device("cuda")
    train_embeddings = train_embeddings.to(device=device)
    train_labels = train_labels.to(device=device)
    test_embeddings = test_embeddings.to(device=device)

    cos = nn.CosineSimilarity(dim=-1)
    all_preds = []
    for idx in range(test_embeddings.shape[0]):
        test_embedding = test_embeddings[idx].unsqueeze(dim=0)
        test_embedding = (
            test_embeddings[idx].unsqueeze(dim=0).repeat(train_embeddings.shape[0], 1)
        )
        sims = cos(test_embedding, train_embeddings)
        top_k = torch.topk(sims, k=k)
        top_k_values = top_k.values
        top_k_indices = top_k.indices

        fetched_labels = train_labels[top_k_indices]
        fetched_onehots = nn.functional.one_hot(fetched_labels, num_classes=num_classes)
        distances = top_k_values.clone().div_(0.07).exp_()
        weighted_sum_onehots = (distances.unsqueeze(dim=1) * fetched_onehots).sum(dim=0)
        prediction = torch.argmax(weighted_sum_onehots)
        all_preds.append(prediction)

    return torch.LongTensor(all_preds).cpu()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ds_path",
        type=str,
        help="The path to the rslearn dataset",
        required=True,
    )
    parser.add_argument(
        "--repeats",
        type=int,
        help="Number of repeats (folds)",
        required=True,
    )
    parser.add_argument(
        "--samples",
        type=int,
        help="Number of samples",
        required=True,
    )
    parser.add_argument(
        "--k",
        type=int,
        help="k for k nearest neighbor",
        required=True,
    )
    parser.add_argument(
        "--classes",
        type=str,
        help="Comma-separated list of classes",
        required=True,
    )
    parser.add_argument(
        "--embed_fname",
        type=str,
        help="Filename to use to save the embeddings",
        required=True,
    )
    args = parser.parse_args()
    class_names = args.classes.split(",")

    # Get all train and test embeddings.
    dataset = Dataset(UPath(args.ds_path))
    train_embedding_list = []
    train_labels = []
    test_embedding_list = []
    test_labels = []
    for window in dataset.load_windows(workers=32):
        embedding_fname = window.path / args.embed_fname
        if not embedding_fname.exists():
            print(f"skipping embeddings for window {window.group}/{window.name} since {embedding_fname} does not exist")
            continue
        with embedding_fname.open("rb") as f:
            embedding = torch.as_tensor(np.load(f))
        label_fname = window.path / "layers" / "label" / "data.geojson"
        with label_fname.open("r") as f:
            fc = json.load(f)
        class_name = fc["features"][0]["properties"]["label"]
        label = class_names.index(class_name)

        if window.group == "train":
            train_embedding_list.append(embedding)
            train_labels.append(label)
        elif window.group == "test":
            test_embedding_list.append(embedding)
            test_labels.append(label)
        else:
            raise ValueError(f"expected all windows to be in train or test but got group {window.group}")
    test_embeddings = torch.stack(test_embedding_list, dim=0)
    test_labels = torch.tensor(test_labels, dtype=torch.int64)

    accuracy_scores = []
    for fold_idx in range(args.repeats):
        print(f"evaluating for fold {fold_idx}")
        # Take subset of train to use.
        cur_train_indices = random.sample(list(range(len(train_embedding_list))), args.samples)
        cur_train_embeddings = torch.stack([train_embedding_list[idx] for idx in cur_train_indices], dim=0)
        cur_train_labels = torch.tensor([train_labels[idx] for idx in cur_train_indices], dtype=torch.int64)

        preds = run_knn_for_k(cur_train_embeddings, cur_train_labels, test_embeddings, len(class_names), args.k)
        accuracy_score = balanced_accuracy_score(test_labels, preds)
        accuracy_scores.append(accuracy_score)

        print(f"fold {fold_idx}: got accuracy {accuracy_score}")

    print(np.mean(accuracy_scores))
