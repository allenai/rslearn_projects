import argparse
import json
import random
import multiprocessing

import numpy as np
import rasterio
import tqdm
import torch
from rslearn.const import WGS84_PROJECTION
from rslearn.dataset import Dataset, Window
from rslearn.utils.raster_format import GeotiffRasterFormat
from sklearn.metrics import balanced_accuracy_score
from torch import nn
from upath import UPath


def load_embedding(embedding_fname: UPath) -> torch.Tensor | None:
    if not embedding_fname.exists():
        print(f"warning: no embedding at {embedding_fname}")
        return None
    with embedding_fname.open("rb") as f:
        return torch.as_tensor(np.load(f))


def load_gse_embedding(window: Window) -> torch.Tensor | None:
    # Search through the layer directories for a GeoTIFF that is not 0s.
    for fname in window.path.glob("layers/gse*/*/geotiff.tif"):
        with rasterio.open(fname.path) as raster:
            array = raster.read()
        if array[:, array.shape[1]//2, array.shape[2]//2].max() == 0:
            continue
        return torch.tensor(array[:, array.shape[1]//2, array.shape[2]//2]) / 8192 - 1
    return None


def load_location_embedding(window: Window) -> torch.Tensor:
    wgs84_geom = window.get_geometry().to_projection(WGS84_PROJECTION)
    lon = wgs84_geom.shp.centroid.x
    lat = wgs84_geom.shp.centroid.y
    return torch.tensor([lon, lat])


def load_pixel_embedding(window: Window) -> torch.Tensor:
    """Get Sentinel-2 pixel values embedding.

    We load center 4x4, averaging on all bands but concatenating over time.
    """
    for group_idx in range(12):
        if not window.is_layer_completed("sentinel2", group_idx):
            return None
    values = []
    for group_idx in range(12):
        cur_values = []
        for bands in [["B01", "B09", "B10"], ["B02", "B03", "B04", "B08"], ["B05", "B06", "B07", "B8A", "B11", "B12"]]:
            raster_dir = window.get_raster_dir("sentinel2", bands, group_idx)
            array = GeotiffRasterFormat().decode_raster(raster_dir, window.projection, window.bounds)
            center_col = array.shape[2] // 2
            center_row = array.shape[1] // 2
            center_crop = array[:, center_row-2:center_row+2, center_col-2:center_col+2]
            mean_values = center_crop.mean(axis=(1, 2))
            cur_values.extend(mean_values.tolist())
        values.append(torch.tensor(cur_values))
    return torch.cat(values, dim=0)


def run_knn_for_k(
    train_embeddings: torch.Tensor,
    train_labels: torch.Tensor,
    test_embeddings: torch.Tensor,
    num_classes: int,
    k: int,
    sim_mode: str,
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

        if sim_mode == "cos":
            sims = cos(test_embedding, train_embeddings)
        elif sim_mode == "l2":
            sims = -torch.square(test_embedding - train_embeddings).sum(dim=-1)

        top_k = torch.topk(sims, k=k)
        top_k_indices = top_k.indices

        fetched_labels = train_labels[top_k_indices]
        fetched_onehots = nn.functional.one_hot(fetched_labels, num_classes=num_classes)
        sum_onehots = fetched_onehots.sum(dim=0)
        prediction = torch.argmax(sum_onehots)
        all_preds.append(prediction)

    return torch.LongTensor(all_preds).cpu()


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver")

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
        help="Number of samples per class or partition",
        required=True,
    )
    parser.add_argument(
        "--k",
        type=int,
        help="k for k nearest neighbor",
        required=True,
    )
    parser.add_argument(
        "--embed_fname",
        type=str,
        help="Filename to use to save the embeddings, or 'gse' to load google satellite embedding from layer dir, or 'loc' to use location",
        required=True,
    )
    parser.add_argument(
        "--workers",
        type=int,
        help="Number of worker processes",
        default=32,
    )
    parser.add_argument(
        "--sim_mode",
        type=str,
        help="Similarity mode, cos or l2",
        default="l2",
    )
    args = parser.parse_args()

    # Load windows and all embeddings.
    dataset = Dataset(UPath(args.ds_path))
    windows = dataset.load_windows(workers=args.workers)
    p = multiprocessing.Pool(args.workers)
    if args.embed_fname == "gse":
        embeddings = list(tqdm.tqdm(p.imap(load_gse_embedding, windows), desc="Loading embeddings", total=len(windows)))
    elif args.embed_fname == "loc":
        embeddings = list(tqdm.tqdm(p.imap(load_location_embedding, windows), desc="Loading embeddings", total=len(windows)))
    elif args.embed_fname == "pixel":
        embeddings = list(tqdm.tqdm(p.imap(load_pixel_embedding, windows), desc="Loading embeddings", total=len(windows)))
    else:
        embeddings = list(tqdm.tqdm(p.imap(load_embedding, [window.path / args.embed_fname for window in windows], chunksize=1), desc="Loading embeddings", total=len(windows)))
    p.close()

    # Get train and test embeddings, along with label / partition.
    class_names = []
    train_embedding_list = []
    train_labels = []
    train_partitions = []
    test_embedding_list = []
    test_labels = []
    for window, embedding in zip(windows, embeddings):
        if embedding is None:
            print("Missing embedding - skipping")
            continue

        class_name = window.options["label"]
        if class_name not in class_names:
            class_names.append(class_name)
        label = class_names.index(class_name)

        if window.group == "train":
            train_embedding_list.append(embedding)
            train_labels.append(label)

            # Get partition, or for classification tasks we use the label instead.
            if window.options["partition"] is not None:
                train_partitions.append(window.options["partition"])
            else:
                train_partitions.append(label)

        elif window.group == "test":
            test_embedding_list.append(embedding)
            test_labels.append(label)

        else:
            raise ValueError(f"expected all windows to be in train or test but got group {window.group}")

    test_embeddings = torch.stack(test_embedding_list, dim=0)
    test_labels = torch.tensor(test_labels, dtype=torch.int64)

    print(f"got n_train={len(train_embedding_list)}, n_test={len(test_embedding_list)}, classes={class_names}, embed_size={len(test_embeddings[0])}")

    train_indexes_by_partition = {}
    for train_idx, partition in enumerate(train_partitions):
        if partition not in train_indexes_by_partition:
            train_indexes_by_partition[partition] = []
        train_indexes_by_partition[partition].append(train_idx)

    print({partition: len(partition_indexes) for partition, partition_indexes in train_indexes_by_partition.items()})

    accuracy_scores = []
    for fold_idx in range(args.repeats):
        print(f"evaluating for fold {fold_idx}")
        # Take a balanced subset of the training data.
        cur_train_indices = []
        for partition_indexes in train_indexes_by_partition.values():
            if args.samples == 0:
                cur_train_indices.extend(partition_indexes)
            else:
                cur_train_indices.extend(random.sample(partition_indexes, args.samples))

        cur_train_embeddings = torch.stack([train_embedding_list[idx] for idx in cur_train_indices], dim=0)
        cur_train_labels = torch.tensor([train_labels[idx] for idx in cur_train_indices], dtype=torch.int64)

        preds = run_knn_for_k(cur_train_embeddings, cur_train_labels, test_embeddings, len(class_names), args.k, args.sim_mode)
        accuracy_score = balanced_accuracy_score(test_labels, preds)
        accuracy_scores.append(accuracy_score)

        print(f"fold {fold_idx}: got accuracy {accuracy_score}")

    print(np.mean(accuracy_scores))
