import torch


def _unwrap(obj):
    if isinstance(obj, dict):
        if "embeddings" in obj:
            return obj["embeddings"]
        return next(v for v in obj.values() if isinstance(v, torch.Tensor))
    return obj


def load_feature_matrix(path: str, mmap: bool = True) -> torch.Tensor:
    try:
        obj = torch.load(path, weights_only=False, mmap=mmap)
    except (RuntimeError, ValueError):
        obj = torch.load(path, weights_only=False, mmap=False)
    tensor = _unwrap(obj)
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"expected a tensor in {path}, got {type(tensor)}")
    return tensor
