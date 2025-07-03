"""Custom multi-task model for handling multiple datasets with different modalities."""

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from rslearn.models.multitask import MultiTaskModel


class MultiDatasetModel(MultiTaskModel):
    """Multi-task model that can handle different task types and class counts.
    
    This model extends the standard MultiTaskModel to handle:
    - Different task types (segmentation, classification, regression)
    - Different class counts for the same task type
    - Dataset-specific task routing
    """

    def __init__(
        self,
        encoder: List[Dict[str, Any]],
        decoders: Dict[str, List[Dict[str, Any]]],
        dataset_task_mapping: Optional[Dict[str, str]] = None,
        **kwargs
    ):
        """Initialize the multi-dataset model.
        
        Args:
            encoder: List of encoder configurations
            decoders: Dictionary mapping task names to decoder configurations
            dataset_task_mapping: Optional mapping from dataset names to task names
            **kwargs: Additional arguments passed to parent
        """
        super().__init__(encoder=encoder, decoders=decoders, **kwargs)
        
        self.dataset_task_mapping = dataset_task_mapping or {}
        
        # Store task types for each decoder
        self.task_types = {}
        for task_name, decoder_configs in decoders.items():
            # Determine task type from the last decoder (head)
            last_decoder = decoder_configs[-1]
            if "SegmentationHead" in last_decoder["class_path"]:
                self.task_types[task_name] = "segmentation"
            elif "ClassificationHead" in last_decoder["class_path"]:
                self.task_types[task_name] = "classification"
            elif "RegressionHead" in last_decoder["class_path"]:
                self.task_types[task_name] = "regression"
            else:
                self.task_types[task_name] = "unknown"

    def forward(
        self, 
        inputs: List[Dict[str, Any]], 
        targets: Optional[List[Dict[str, Any]]] = None,
        dataset_info: Optional[List[str]] = None
    ) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Forward pass with dataset-aware task routing.
        
        Args:
            inputs: List of input dictionaries
            targets: List of target dictionaries
            dataset_info: Optional list of dataset identifiers for each sample
            
        Returns:
            Tuple of (outputs, losses)
        """
        # Get encoder features
        features = self.encoder(inputs)
        
        outputs = {}
        losses = {}
        
        # Process each sample based on its dataset
        for i, (input_dict, target_dict) in enumerate(zip(inputs, targets or [])):
            # Determine which task to use for this sample
            task_name = self._get_task_for_sample(i, dataset_info, input_dict, target_dict)
            
            if task_name not in self.decoders:
                continue
                
            # Get decoder for this task
            decoder = self.decoders[task_name]
            
            # Forward through decoder
            decoder_output = decoder(features[i:i+1])
            
            # Store output
            if task_name not in outputs:
                outputs[task_name] = []
            outputs[task_name].append(decoder_output)
            
            # Compute loss if targets are provided
            if targets and target_dict and task_name in target_dict:
                task_target = target_dict[task_name]
                loss = self._compute_loss(decoder_output, task_target, task_name)
                if task_name not in losses:
                    losses[task_name] = []
                losses[task_name].append(loss)
        
        # Stack outputs and losses
        for task_name in outputs:
            if outputs[task_name]:
                outputs[task_name] = torch.cat(outputs[task_name], dim=0)
            if task_name in losses and losses[task_name]:
                losses[task_name] = torch.stack(losses[task_name]).mean()
        
        return outputs, losses

    def _get_task_for_sample(
        self, 
        sample_idx: int, 
        dataset_info: Optional[List[str]], 
        input_dict: Dict[str, Any], 
        target_dict: Dict[str, Any]
    ) -> str:
        """Determine which task to use for a given sample.
        
        Args:
            sample_idx: Index of the sample
            dataset_info: List of dataset identifiers
            input_dict: Input dictionary for the sample
            target_dict: Target dictionary for the sample
            
        Returns:
            Task name to use for this sample
        """
        # If we have explicit dataset info, use the mapping
        if dataset_info and sample_idx < len(dataset_info):
            dataset_name = dataset_info[sample_idx]
            if dataset_name in self.dataset_task_mapping:
                return self.dataset_task_mapping[dataset_name]
        
        # Otherwise, infer from available targets
        if target_dict:
            # Find the first available task
            for task_name in self.decoders.keys():
                if task_name in target_dict:
                    return task_name
        
        # Fallback to first available task
        return list(self.decoders.keys())[0]

    def _compute_loss(
        self, 
        output: torch.Tensor, 
        target: torch.Tensor, 
        task_name: str
    ) -> torch.Tensor:
        """Compute loss for a specific task.
        
        Args:
            output: Model output
            target: Target values
            task_name: Name of the task
            
        Returns:
            Loss value
        """
        task_type = self.task_types.get(task_name, "unknown")
        
        if task_type == "segmentation":
            # Use cross-entropy for segmentation
            return nn.functional.cross_entropy(output, target)
        elif task_type == "classification":
            # Use cross-entropy for classification
            return nn.functional.cross_entropy(output, target)
        elif task_type == "regression":
            # Use MSE for regression
            return nn.functional.mse_loss(output.squeeze(), target)
        else:
            # Default to MSE
            return nn.functional.mse_loss(output, target)

    def get_task_info(self) -> Dict[str, Any]:
        """Get information about the tasks in this model.
        
        Returns:
            Dictionary with task information
        """
        info = {
            "num_tasks": len(self.decoders),
            "task_names": list(self.decoders.keys()),
            "task_types": self.task_types,
            "dataset_mapping": self.dataset_task_mapping,
        }
        return info 