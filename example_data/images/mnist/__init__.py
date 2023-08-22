from typing import Tuple
import numpy as np
import numpy.typing as npt
from sklearn.datasets import fetch_openml
from _cache import save_numpy_arrays, try_load_numpy_arrays



IMAGE_DIMENSIONS: Tuple[int, int] = (28,28)
__CACHE_NAME: str = "mnist_dataset"


def __fetch_mnist_dataset() -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.int16]]:

    dataset = fetch_openml("mnist_784", parser="auto")

    imgs_flattened: npt.NDArray[np.float64] = dataset.data.to_numpy()
    labels: npt.NDArray[np.int16] = dataset.target.to_numpy().astype(np.int16)

    assert imgs_flattened.ndim == 2
    assert imgs_flattened.shape[0] == labels.shape[0]

    imgs = imgs_flattened.reshape(
        (imgs_flattened.shape[0], IMAGE_DIMENSIONS[0], IMAGE_DIMENSIONS[1])
    )

    return imgs, labels


def load_dataset() -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.int16]]:

    cache = try_load_numpy_arrays(__CACHE_NAME)

    if cache:
        return cache["imgs"], cache["labels"]
    else:
        imgs, labels = __fetch_mnist_dataset()
        save_numpy_arrays(__CACHE_NAME, {"imgs": imgs, "labels": labels})
        return imgs, labels
