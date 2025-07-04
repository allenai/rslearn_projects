"""
Multi-dataset data module for training on multiple datasets with different modalities.
Basic data flow:
1. Build individual RslearnDataModule instances from dataset configs
2. Build a MultiWrapperDataset from the DataLoaders of these RslearnDataModule instances
3. Wrap the MultiWrapperDataset in a DataLoader shell
4. Pass the DataLoader to the LightningDataModule and use as normal
"""

import random
from typing import Dict, Optional, List
import lightning as L
import torch
from torch.utils.data import DataLoader, IterableDataset
from rslearn.train.data_module import RslearnDataModule
from rslearn.train.tasks.multi_task import MultiTask


class MultiWrapperDataset(IterableDataset):
    """
    Wrap dataloaders produced by RslearnDataModule instances, this is meant
    to be then wrapped in a DataLoader which feeds batches without modification.
    """

    def __init__(self, dataloaders: List[DataLoader], tasks: List[str], strategy: str = "random"):
        """
        dataloaders: list of DataLoader objects
        tasks: list of task names, one for each dataloader
        strategy: "random" or "round_robin"
        """
        assert len(dataloaders) == len(tasks), "number of dataloaders and tasks must match"
        self.dataloaders = dataloaders
        self.strategy = strategy
        self.tasks = tasks.copy()

    def __iter__(self):
        self.iterators = [iter(dl) for dl in self.dataloaders]
        while True:
            if self.strategy == "random":
                idx = random.randint(0, len(self.iterators) - 1)
            elif self.strategy == "round_robin":
                idx = getattr(self, 'last_idx', -1) + 1
                idx %= len(self.iterators)
                self.last_idx = idx
            else:
                raise ValueError("Unknown strategy")

            try:
                batch = next(self.iterators[idx])
                for instance in batch[2]:
                    instance["dataset_source"] = self.tasks[idx]
                yield batch
            except StopIteration:
                self.iterators.pop(idx)
                self.tasks.pop(idx)
                if len(self.iterators) == 0:
                    break
                if self.strategy == "round_robin":
                    self.last_idx -= 1


class MultiDatasetDataModule(L.LightningDataModule):
    """Data module that manages multiple RslearnDataModule instances.
    
    This module creates and manages multiple RslearnDataModule instances, each handling
    a different dataset. It provides a unified interface for training on multiple datasets
    with different modalities and labels.
    
    Each dataset can have different:
    - Input modalities (e.g., Sentinel-2 vs Landsat)
    - Label schemas (e.g., different classification classes)
    - Task types (e.g., classification vs detection)
    - Transforms and preprocessing
    """

    def __init__(
        self,
        dataset_configs: Dict[str, RslearnDataModule],
        task: MultiTask,
        batch_size: int = 16,
        num_workers: int = 32,
        **kwargs
    ):
        super().__init__()
        
        self.tasks = list(dataset_configs.keys())
        self.data_modules = dataset_configs
        self.global_batch_size = batch_size
        self.global_num_workers = num_workers

    def setup(self, stage: Optional[str] = None):
        """Set up the datasets for the given stage.
        
        Args:
            stage: The stage to set up ('fit', 'validate', 'test', 'predict')
        """
        for data_module in self.data_modules.values():
            data_module.setup(stage)

    def _get_dataloader(self, split: str) -> DataLoader[dict[str, torch.Tensor]]:
        dataloaders = []
        for data_module in self.data_modules.values():
            dataloaders.append(data_module._get_dataloader(split))
        return DataLoader(
            MultiWrapperDataset(dataloaders, self.tasks),
            batch_size=None,
            num_workers=0,    # handle splitting and multiprocessing in the 
            shuffle=False,    # individual dataloaders spawned by rslearn
            pin_memory=True,  # unclear if we need this, haven't checked properly
        )

    def train_dataloader(self) -> DataLoader:
        return self._get_dataloader("train")

    def val_dataloader(self) -> DataLoader:
        return self._get_dataloader("val")

    def test_dataloader(self) -> DataLoader:
        return self._get_dataloader("test")

    def predict_dataloader(self) -> DataLoader:
        return self._get_dataloader("predict")
