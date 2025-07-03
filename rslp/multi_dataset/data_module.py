"""Multi-dataset data module for training on multiple datasets with different modalities."""

from typing import Any, Dict, List, Optional

import lightning as L
import torch
from torch.utils.data import ConcatDataset, DataLoader
from rslearn.train.data_module import RslearnDataModule
from rslearn.train.tasks.multi_task import MultiTask
from rslearn.train.dataset import SplitConfig


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
        dataset_configs: List[Dict[str, Any]],
        task: MultiTask | None = None,
        batch_size: int = 16,
        num_workers: int = 32,
        **kwargs
    ):
        super().__init__()
        
        self.dataset_configs = dataset_configs
        self.global_batch_size = batch_size
        self.global_num_workers = num_workers
        self.multi_task = task

        # Store individual data modules
        self.data_modules = []
        
        # Combined datasets for each stage
        self.train_datasets = []
        self.val_datasets = []
        self.test_datasets = []
        self.predict_datasets = []

    def setup(self, stage: Optional[str] = None):
        """Set up the datasets for the given stage.
        
        Args:
            stage: The stage to set up ('fit', 'validate', 'test', 'predict')
        """
        # Clear previous datasets
        self.train_datasets = []
        self.val_datasets = []
        self.test_datasets = []
        self.predict_datasets = []
        
        # Create RslearnDataModule instances for each config
        for config in self.dataset_configs:
            # Extract dataset-specific batch_size and num_workers, fall back to global
            dataset_batch_size = config.get('batch_size', self.global_batch_size)
            dataset_num_workers = config.get('num_workers', self.global_num_workers)
            
            # Create the data module
            data_module = RslearnDataModule(
                inputs=config["inputs"],
                task=config["task"],
                path=config["path"],
                path_options=config.get("path_options", {}),
                batch_size=dataset_batch_size,
                num_workers=dataset_num_workers,
                default_config=SplitConfig(**config.get("default_config", {})),
                train_config=SplitConfig(**config.get("train_config", {})),
                val_config=SplitConfig(**config.get("val_config", {})),
                test_config=SplitConfig(**config.get("test_config", {})),
                predict_config=SplitConfig(**config.get("predict_config", {})),
            )
            
            # Set up the data module for the current stage
            data_module.setup(stage)
            self.data_modules.append(data_module)
            
            # Collect datasets based on stage
            if stage == "fit" or stage is None:
                if "train" in data_module.datasets:
                    self.train_datasets.append(data_module.datasets["train"])
                if "val" in data_module.datasets:
                    self.val_datasets.append(data_module.datasets["val"])
            elif stage == "validate":
                if "val" in data_module.datasets:
                    self.val_datasets.append(data_module.datasets["val"])
            elif stage == "test":
                if "test" in data_module.datasets:
                    self.test_datasets.append(data_module.datasets["test"])
            elif stage == "predict":
                if "predict" in data_module.datasets:
                    self.predict_datasets.append(data_module.datasets["predict"])
    
    def train_dataloader(self) -> DataLoader:
        """Create training data loader.
        
        Returns:
            Training data loader
        """
        if not self.train_datasets:
            raise ValueError("No training datasets available. Make sure to call setup('fit') first.")
        
        # Concatenate all training datasets
        combined_dataset = ConcatDataset(self.train_datasets)
        
        return DataLoader(
            combined_dataset,
            batch_size=self.global_batch_size,
            shuffle=True,
            num_workers=self.global_num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn
        )
    
    def val_dataloader(self) -> DataLoader:
        """Create validation data loader.
        
        Returns:
            Validation data loader
        """
        if not self.val_datasets:
            raise ValueError("No validation datasets available. Make sure to call setup('fit') or setup('validate') first.")
        
        # Concatenate all validation datasets
        combined_dataset = ConcatDataset(self.val_datasets)
        
        return DataLoader(
            combined_dataset,
            batch_size=self.global_batch_size,
            shuffle=False,
            num_workers=self.global_num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn
        )
    
    def test_dataloader(self) -> DataLoader:
        """Create test data loader.
        
        Returns:
            Test data loader
        """
        if not self.test_datasets:
            raise ValueError("No test datasets available. Make sure to call setup('test') first.")
        
        # Concatenate all test datasets
        combined_dataset = ConcatDataset(self.test_datasets)
        
        return DataLoader(
            combined_dataset,
            batch_size=self.global_batch_size,
            shuffle=False,
            num_workers=self.global_num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn
        )
    
    def predict_dataloader(self) -> DataLoader:
        """Create predict data loader.
        
        Returns:
            Predict data loader
        """
        if not self.predict_datasets:
            raise ValueError("No predict datasets available. Make sure to call setup('predict') first.")
        
        # Concatenate all predict datasets
        combined_dataset = ConcatDataset(self.predict_datasets)
        
        return DataLoader(
            combined_dataset,
            batch_size=self.global_batch_size,
            shuffle=False,
            num_workers=self.global_num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn
        )
    
    def _collate_fn(self, batch):
        """Custom collate function to handle different dataset outputs.
        
        Args:
            batch: Batch of samples from different datasets
            
        Returns:
            Collated batch
        """
        # For now, use default collate
        # In the future, this could handle dataset-specific collation
        return torch.utils.data.dataloader.default_collate(batch)

    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about the loaded datasets.
        
        Returns:
            Dictionary with dataset information
        """
        info = {
            "num_datasets": len(self.dataset_configs),
            "dataset_paths": [config["path"] for config in self.dataset_configs],
            "train_samples": sum(len(ds) for ds in self.train_datasets) if self.train_datasets else 0,
            "val_samples": sum(len(ds) for ds in self.val_datasets) if self.val_datasets else 0,
            "test_samples": sum(len(ds) for ds in self.test_datasets) if self.test_datasets else 0,
            "predict_samples": sum(len(ds) for ds in self.predict_datasets) if self.predict_datasets else 0,
        }
        return info
    
    def get_data_modules(self) -> List[RslearnDataModule]:
        """Get the individual data modules.
        
        Returns:
            List of RslearnDataModule instances
        """
        return self.data_modules 