"""Measure model throughput for rslearn-trained Helios checkpoints.

This script loads a Helios model from a run directory and measures its inference throughput
with configurable batch sizes, image sizes, and modality configurations.
"""

import os
import argparse
import json
import yaml
import importlib
import time
import torch
import numpy as np
from typing import Dict, List, Tuple, Any
from contextlib import contextmanager

from rslearn.train.lightning_module import RestoreConfig


class ThroughputMeasurer:
    """Configurable throughput measurement for Helios models."""
    
    def __init__(
        self,
        run_dir: str,
        project: str = "2025_08_29_finetune_benchmarks",
        base_dir: str = "/weka/dfive-default/rslearn-eai/projects",
        device: str = "cuda",
        warmup_runs: int = 5,
        measurement_runs: int = 20,
        batch_sizes: List[int] = [1, 4, 8, 16],
        image_sizes: List[int] = [224, 448, 672],
        modalities: List[str] = ["sentinel2_l2a"],
        dtype: torch.dtype = torch.float16,
        compile_model: bool = False,
        dataset_source: str = "sentinel2_l2a",
    ):
        self.run_dir = run_dir
        self.project = project
        self.base_dir = base_dir
        self.device = device
        self.warmup_runs = warmup_runs
        self.measurement_runs = measurement_runs
        self.batch_sizes = batch_sizes
        self.image_sizes = image_sizes
        self.modalities = modalities
        self.dtype = dtype
        self.compile_model = compile_model
        self.dataset_source = dataset_source
        
        self.model = None
        self.model_config = None
        self.results = {}
        
        # Modality-specific configurations based on Helios constants
        self.modality_configs = {
            "sentinel2_l2a": {"num_bands": 12, "band_sets": 3, "is_multitemporal": True},
            "sentinel1": {"num_bands": 2, "band_sets": 1, "is_multitemporal": True},
            "worldcover": {"num_bands": 1, "band_sets": 1, "is_multitemporal": False},
            "openstreetmap_raster": {"num_bands": 30, "band_sets": 1, "is_multitemporal": False},
            "landsat": {"num_bands": 11, "band_sets": 2, "is_multitemporal": True},
        }
        
        # Default sequence length for multitemporal data
        self.max_sequence_length = 12
        
    def load_model(self) -> None:
        """Load the model from the run directory."""
        print("Loading model...")
        
        # Load the finetune config
        finetune_config_path = os.path.join(self.base_dir, self.project, self.run_dir, "checkpoints", "config.yaml")
        if not os.path.exists(finetune_config_path):
            raise FileNotFoundError(f"Config file not found: {finetune_config_path}")
            
        with open(finetune_config_path, "r") as f:
            finetune_config = yaml.safe_load(f)
            
        # Extract model configuration
        enc = finetune_config["model"]["init_args"]["model"]["init_args"]["encoder"][0]
        self.model_config = enc["init_args"]
        base_class = self._resolve_class_path(enc["class_path"])
        
        print(f"Model class: {base_class}")
        print(f"Model config: {json.dumps(self.model_config, indent=2)}")
        
        # Load checkpoint
        ckpt_path = os.path.join(self.base_dir, self.project, self.run_dir, "checkpoints", "last.ckpt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
            
        restore_config = RestoreConfig(
            ckpt_path,
            selector=["state_dict"],
            ignore_prefixes=["model.decoders"],
            remap_prefixes=[("model.encoder.0.model.", "model.encoder.")]
        )
        state_dict = restore_config.get_state_dict()
        
        # Initialize model
        self.model = base_class(**self.model_config)
        self.model.load_state_dict(state_dict, strict=False)
        self.model = self.model.to(self.device).to(self.dtype)
        self.model.eval()
        
        # Check if this is a TaskConditionedHelios model and get available tasks
        if hasattr(self.model, 'tasks'):
            available_tasks = self.model.tasks
            print(f"Available tasks: {available_tasks}")
            
            # Update dataset_source to use the first available task if current one is not available
            if self.dataset_source not in available_tasks:
                self.dataset_source = available_tasks[0]
                print(f"Updated dataset_source to: {self.dataset_source}")
        else:
            print("Model is not task-conditioned, using provided dataset_source")
        
        # Try to determine what modalities the model was trained with
        # by checking the model config or trying a simple forward pass
        print(f"Using modalities: {self.modalities}")
        print(f"Dataset source: {self.dataset_source}")
        
        # Compile model if requested
        if self.compile_model and hasattr(torch, 'compile'):
            print("Compiling model...")
            self.model = torch.compile(self.model)
            
        print("Model loaded successfully!")
        
        # Count model parameters
        self._count_parameters()
        
    def _count_parameters(self) -> None:
        """Count and display model parameters."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"Model Parameters:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Non-trainable parameters: {total_params - trainable_params:,}")
        
        # Store parameter counts in results
        self.total_params = total_params
        self.trainable_params = trainable_params
        
    def _resolve_class_path(self, class_path: str):
        """Resolve class from string path."""
        module_path, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    
    @contextmanager
    def _timer(self):
        """Context manager for timing operations."""
        if self.device == "cuda":
            torch.cuda.synchronize()
        start = time.time()
        try:
            yield
        finally:
            if self.device == "cuda":
                torch.cuda.synchronize()
            end = time.time()
        return end - start
    
    def _generate_inputs(self, batch_size: int, image_size: int) -> List[Dict[str, Any]]:
        """Generate random Helios-compatible input data for testing."""
        inputs = []
        
        for _ in range(batch_size):
            sample = {"dataset_source": self.dataset_source}
            
            # Always provide all modalities that the model might expect
            # This ensures the model doesn't fail when looking for specific modalities
            all_modalities = ["sentinel2_l2a", "sentinel1", "worldcover", "openstreetmap_raster", "landsat"]
            
            for modality in all_modalities:
                if modality in self.modality_configs:
                    config = self.modality_configs[modality]
                    num_bands = config["num_bands"]
                    is_multitemporal = config["is_multitemporal"]
                    
                    if is_multitemporal:
                        # For multitemporal data: (timesteps * num_bands, height, width)
                        # Use max_sequence_length timesteps
                        modality_data = torch.randn(
                            self.max_sequence_length * num_bands, image_size, image_size,
                            device=self.device,
                            dtype=self.dtype
                        )
                    else:
                        # For static data: (num_bands, height, width)
                        modality_data = torch.randn(
                            num_bands, image_size, image_size,
                            device=self.device,
                            dtype=self.dtype
                        )
                    
                    sample[modality] = modality_data
            
            inputs.append(sample)
        
        return inputs
    
    def _measure_forward_pass(self, batch_size: int, image_size: int) -> Tuple[float, float]:
        """Measure a single forward pass."""
        inputs = self._generate_inputs(batch_size, image_size)
        
        # Warmup
        with torch.no_grad():
            for i in range(self.warmup_runs):
                try:
                    output = self.model(inputs)
                    if output is None:
                        raise ValueError("Model returned None during warmup")
                    if i == 0:  # Print debug info only for first warmup
                        print(f"  Model output type: {type(output)}")
                        if isinstance(output, list):
                            print(f"  Output list length: {len(output)}")
                            for j, item in enumerate(output):
                                print(f"  Output[{j}] type: {type(item)}, shape: {item.shape if hasattr(item, 'shape') else 'N/A'}")
                except Exception as e:
                    raise ValueError(f"Error during warmup: {e}")
        
        # Actual measurement
        times = []
        with torch.no_grad():
            for i in range(self.measurement_runs):
                try:
                    # Simple timing without context manager
                    if self.device == "cuda":
                        torch.cuda.synchronize()
                    start = time.time()
                    
                    output = self.model(inputs)
                    
                    if self.device == "cuda":
                        torch.cuda.synchronize()
                    end = time.time()
                    
                    elapsed = end - start
                    
                    if output is None:
                        raise ValueError("Model returned None during measurement")
                    # Additional validation
                    if isinstance(output, list) and len(output) > 0:
                        if output[0] is None:
                            raise ValueError("Model output[0] is None")
                    elif output is None:
                        raise ValueError("Model output is None")
                    
                    times.append(elapsed)
                except Exception as e:
                    print(f"  Error in measurement run {i+1}: {e}")
                    print(f"  Error type: {type(e)}")
                    import traceback
                    traceback.print_exc()
                    raise ValueError(f"Error during measurement: {e}")
        
        mean_time = np.mean(times)
        std_time = np.std(times)
        
        return mean_time, std_time
    
    def measure_throughput(self) -> Dict:
        """Measure throughput across different configurations."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
            
        print("Starting throughput measurements...")
        self.results = {
            "run_dir": self.run_dir,
            "project": self.project,
            "device": self.device,
            "dtype": str(self.dtype),
            "compile_model": self.compile_model,
            "modalities": self.modalities,
            "dataset_source": self.dataset_source,
            "total_parameters": getattr(self, 'total_params', 0),
            "trainable_parameters": getattr(self, 'trainable_params', 0),
            "measurements": {}
        }
        
        total_configs = len(self.batch_sizes) * len(self.image_sizes)
        current_config = 0
        
        for batch_size in self.batch_sizes:
            for image_size in self.image_sizes:
                current_config += 1
                print(f"Measuring [{current_config}/{total_configs}] - Batch: {batch_size}, ImageSize: {image_size}x{image_size}")
                
                try:
                    mean_time, std_time = self._measure_forward_pass(batch_size, image_size)
                    
                    # Calculate throughput metrics
                    pixels_per_second = (batch_size * image_size * image_size) / mean_time
                    samples_per_second = batch_size / mean_time
                    
                    # Calculate efficiency metrics
                    total_params = self.results.get('total_parameters', 0)
                    params_per_second = total_params / mean_time if total_params > 0 else 0
                    flops_per_param = (batch_size * image_size * image_size) / total_params if total_params > 0 else 0
                    
                    config_key = f"batch_{batch_size}_img_{image_size}"
                    self.results["measurements"][config_key] = {
                        "batch_size": batch_size,
                        "image_size": image_size,
                        "mean_time_ms": mean_time * 1000,
                        "std_time_ms": std_time * 1000,
                        "pixels_per_second": pixels_per_second,
                        "samples_per_second": samples_per_second,
                        "params_per_second": params_per_second,
                        "flops_per_param": flops_per_param,
                        "memory_allocated_gb": torch.cuda.memory_allocated() / 1e9 if self.device == "cuda" else 0,
                        "memory_reserved_gb": torch.cuda.memory_reserved() / 1e9 if self.device == "cuda" else 0,
                    }
                    
                    print(f"  Time: {mean_time*1000:.2f}±{std_time*1000:.2f} ms")
                    print(f"  Throughput: {pixels_per_second:.0f} pixels/s, {samples_per_second:.2f} samples/s")
                    
                except Exception as e:
                    print(f"  Error: {e}")
                    config_key = f"batch_{batch_size}_img_{image_size}"
                    self.results["measurements"][config_key] = {
                        "batch_size": batch_size,
                        "image_size": image_size,
                        "error": str(e)
                    }
        
        return self.results
    
    def save_results(self, output_path: str) -> None:
        """Save results to JSON file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to: {output_path}")
    
    def print_summary(self) -> None:
        """Print a summary of the results."""
        print("\n" + "="*80)
        print("THROUGHPUT MEASUREMENT SUMMARY")
        print("="*80)
        print(f"Run Directory: {self.run_dir}")
        print(f"Project: {self.project}")
        print(f"Device: {self.device}")
        print(f"Data Type: {self.dtype}")
        print(f"Compiled: {self.compile_model}")
        print(f"Total Parameters: {self.results.get('total_parameters', 0):,}")
        print(f"Trainable Parameters: {self.results.get('trainable_parameters', 0):,}")
        print("-"*80)
        
        if "measurements" in self.results:
            print(f"{'Configuration':20} | {'Pixels/s':>10} | {'Samples/s':>8} | {'Time(ms)':>8} | {'Params/s':>12}")
            print("-" * 80)
            for config_key, data in self.results["measurements"].items():
                if "error" not in data:
                    print(f"{config_key:20} | {data['pixels_per_second']:10.0f} | {data['samples_per_second']:8.2f} | {data['mean_time_ms']:8.1f} | {data['params_per_second']:12.0f}")
                else:
                    print(f"{config_key:20} | ERROR: {data['error']}")
        
        print("="*80)


def main():
    parser = argparse.ArgumentParser(description="Measure model throughput")
    parser.add_argument("--run", type=str, required=True, help="Run directory name")
    parser.add_argument("--project", type=str, default="2025_08_29_finetune_benchmarks", help="Project name")
    parser.add_argument("--base_dir", type=str, default="/weka/dfive-default/rslearn-eai/projects", help="Base directory for projects")
    parser.add_argument("--output", type=str, help="Output file for results")
    
    # Measurement parameters
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to run on")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32", "bfloat16"], help="Data type")
    parser.add_argument("--warmup_runs", type=int, default=5, help="Number of warmup runs")
    parser.add_argument("--measurement_runs", type=int, default=20, help="Number of measurement runs")
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[1, 8, 16], help="Batch sizes to test")
    parser.add_argument("--image_sizes", type=int, nargs="+", default=[128, 224], help="Image sizes to test")
    parser.add_argument("--modalities", type=str, nargs="+", default=["sentinel2_l2a", "sentinel1", "worldcover"], help="Modalities to include")
    parser.add_argument("--dataset_source", type=str, default="sentinel2_l2a", help="Dataset source for task conditioning")
    parser.add_argument("--compile", action="store_true", help="Compile model with torch.compile")
    
    args = parser.parse_args()
    
    # Convert dtype string to torch dtype
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]
    
    # Initialize measurer
    measurer = ThroughputMeasurer(
        run_dir=args.run,
        project=args.project,
        base_dir=args.base_dir,
        device=args.device,
        warmup_runs=args.warmup_runs,
        measurement_runs=args.measurement_runs,
        batch_sizes=args.batch_sizes,
        image_sizes=args.image_sizes,
        modalities=args.modalities,
        dataset_source=args.dataset_source,
        dtype=dtype,
        compile_model=args.compile,
    )
    
    try:
        # Load model and measure throughput
        measurer.load_model()
        results = measurer.measure_throughput()

        # Save and display results
        if args.output is not None:
            measurer.save_results(args.output)
        measurer.print_summary()
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
