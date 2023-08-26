from typing import Iterable, Tuple
import pytest
import numpy as np
import numpy.typing as npt
from numpy.testing import assert_allclose
from images.feature_extraction import hog


_CASES: Iterable[Tuple[npt.NDArray[np.float64],Tuple[int,int],int,bool,npt.NDArray[np.int32]]] = [
    (
        np.array([
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
        ], dtype=np.float64),
        (2,2),
        2,
        False,
        np.array([
            [ [4,0], [0,4] ],
            [ [0,4], [4,0] ],
        ], dtype=np.int32)
    )
]


@pytest.mark.parametrize(("img","cell_size","nbins","signed","exp_hists"), _CASES)
def test_dimensions(img: npt.NDArray[np.float64], cell_size: Tuple[int,int], nbins: int, signed: bool, exp_hists: npt.NDArray[np.int32]):
    out = hog(img, cell_size, nbins, signed)
    assert out.shape[0] == img.shape[0]/cell_size[0]
    assert out.shape[1] == img.shape[1]/cell_size[1]
    assert out.shape[2] == nbins


@pytest.mark.parametrize(("img","cell_size","nbins","signed","exp_hists"), _CASES)
def test_values(img: npt.NDArray[np.float64], cell_size: Tuple[int,int], nbins: int, signed: bool, exp_hists: npt.NDArray[np.int32]):
    out = hog(img, cell_size, nbins, signed)
    assert_allclose(out, exp_hists)
