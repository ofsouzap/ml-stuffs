from typing import Dict, Optional, Any
from pathlib import Path
import numpy as np
import numpy.typing as npt


_cache_dirpath: Path = Path(__file__).parent / ".cached_data"


def save_numpy_array(cache_name: str, xs: npt.NDArray) -> None:
    path = _cache_dirpath/(cache_name+".npy")
    np.save(path, xs, allow_pickle=False)


def save_numpy_arrays(cache_name: str, named_arrays: Dict[str, npt.NDArray]) -> None:
    path = _cache_dirpath/(cache_name+".npz")
    np.savez(path, **named_arrays)


def try_load_numpy_array(cache_name: str) -> Optional[npt.NDArray]:
    path = _cache_dirpath/(cache_name+".npy")
    if path.is_file():
        return np.load(path, allow_pickle=False)
    else:
        return None


def try_load_numpy_arrays(cache_name: str) -> Optional[Dict[str, npt.NDArray]]:
    path = _cache_dirpath/(cache_name+".npz")
    if path.is_file():
        return np.load(path, allow_pickle=False)
    else:
        return None
