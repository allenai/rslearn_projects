"""Build PCA artifacts in olmoearth_run's pickle format, without olmoearth_run or sklearn.

The pickle names olmoearth_run's PcaArtifact and scikit-learn's IncrementalPCA by module
path, so stand-in classes are registered under those paths just long enough to dump.
"""

import pickle
import sys
import types
from pathlib import Path

import numpy as np

from rslp.large_scale_embeddings import pca

_ARTIFACT_MODULE = "olmoearth_run.shared.tools.pca_artifact"
_PCA_MODULE = "sklearn.decomposition._incremental_pca"


def fit_int8_basis(samples: np.ndarray) -> pca.PcaArtifact:
    """Fit a basis on int8 embedding vectors the way olmoearth_run does.

    Args:
        samples: int8 array of shape (pixels, dimensions).

    Returns:
        the in-memory artifact.
    """
    values = samples.astype(np.float32)
    mean = values.mean(axis=0)
    _, singular, right = np.linalg.svd(values - mean, full_matrices=False)
    components = right[: pca.PCA_N_COMPONENTS]
    variance = singular**2
    transformed = (values - mean) @ components.T
    bounds = np.percentile(
        transformed, [pca.NORM_PERCENTILE_LOW, pca.NORM_PERCENTILE_HIGH], axis=0
    )
    return pca.PcaArtifact(
        mean=mean.astype(np.float32),
        components=components.astype(np.float32),
        norm_bounds=bounds.astype(np.float32),
        explained_variance_ratio=(variance / variance.sum())[: pca.PCA_N_COMPONENTS],
        metadata={"n_samples_seen": len(samples)},
    )


def write_olmoearth_run_artifact(path: Path, artifact: pca.PcaArtifact) -> str:
    """Pickle an artifact exactly as olmoearth_run's fit-embedding-pca lays it out.

    Args:
        path: the .pkl file to write.
        artifact: the arrays to store.

    Returns:
        the path as a string, ready to pass as an artifact_path.
    """
    wrapper_cls = type("PcaArtifact", (), {"__module__": _ARTIFACT_MODULE})
    pca_cls = type("IncrementalPCA", (), {"__module__": _PCA_MODULE})
    estimator = pca_cls()
    estimator.__dict__.update(
        {
            "n_components": pca.PCA_N_COMPONENTS,
            "whiten": False,
            "mean_": artifact.mean.astype(np.float64),
            "components_": artifact.components.astype(np.float64),
            "explained_variance_ratio_": artifact.explained_variance_ratio,
            "n_samples_seen_": np.int64(artifact.metadata.get("n_samples_seen", 1000)),
        }
    )
    wrapper = wrapper_cls()
    wrapper.__dict__.update({"pca": estimator, "norm_bounds": artifact.norm_bounds})

    added: list[str] = []
    replaced: list[tuple[types.ModuleType, str, object]] = []
    for module_name, cls in ((_ARTIFACT_MODULE, wrapper_cls), (_PCA_MODULE, pca_cls)):
        parts = module_name.split(".")
        for i in range(1, len(parts) + 1):
            name = ".".join(parts[:i])
            if name not in sys.modules:
                sys.modules[name] = types.ModuleType(name)
                added.append(name)
        module = sys.modules[module_name]
        if hasattr(module, cls.__name__):
            replaced.append((module, cls.__name__, getattr(module, cls.__name__)))
        setattr(module, cls.__name__, cls)
    try:
        path.write_bytes(pickle.dumps(wrapper))
    finally:
        # Put back anything real, e.g. scikit-learn when it happens to be installed.
        for module, name, original in replaced:
            setattr(module, name, original)
        for name in added:
            del sys.modules[name]
    return str(path)
