from typing import Tuple, List
import numpy as np
import numpy.typing as npt
from images.convolution import convolve2d, SOBEL_COMPLEX


def histogram_of_oriented_gradients(
    img: npt.NDArray[np.float64],
    cell_size: Tuple[int, int],
    nbins: int,
    signed: bool = False) -> npt.NDArray[np.int32]:
    """Produces a histogram of oriented gradients for each cell of an image

Parameters:

    img - a NxM array of pixel values for the image to use

    cell_size - the x and y (respectively) dimensions of cell to use. The cells must fit exactly into the image.

    nbins - the number of bins to use in the histogram for each cell

    signed - whether to consider the gradients in the image as signed or unsigned

Returns:

    histograms - a (N/cell_size[0])x(M/cell_size[1])x(nbins) array of histograms for each cell. \
The value `histograms[i,j,k]` will give you the number of pixels in cell `(i,j)` are in the bin `(k*max/nbins) <= gradient_angle < ((k+1)*max/nbins)` \
where `max` is 180 degrees when `signed==False` or 360 degrees when `signed==True`
"""

    # Input checks

    if img.ndim != 2:
        raise ValueError("Image isn't a 2D array")

    if img.shape[0] % cell_size[0] != 0:
        raise ValueError("Cell size x-dimension isn't a divisor of image x-dimension")

    if img.shape[1] % cell_size[1] != 0:
        raise ValueError("Cell size y-dimension isn't a divisor of image y-dimension")

    # Preparation

    bin_min = 0
    bin_max = np.pi if not signed else 2*np.pi

    # Calculate gradients of pixels

    pixel_convolutions = convolve2d(img, SOBEL_COMPLEX, mode="same")
    pixel_gradients: npt.NDArray[np.float64] = np.angle(pixel_convolutions, deg=False)

    if not signed:
        pixel_gradients[pixel_gradients<0] += np.pi

    # Produce histograms

    cell_count: Tuple[int,int] = (
        int(img.shape[0]/cell_size[0]),
        int(img.shape[1]/cell_size[1]),
    )

    histograms = np.zeros(shape=(cell_count[0],cell_count[1],nbins), dtype=np.int32)

    for cell_x_idx in range(cell_count[0]):
        for cell_y_idx in range(cell_count[1]):

            cell = pixel_gradients[
                cell_x_idx*cell_size[0]:(cell_x_idx+1)*cell_size[0],
                cell_y_idx*cell_size[1]:(cell_y_idx+1)*cell_size[1],
            ]

            cell_histogram,_ = np.histogram(
                cell.flatten(),
                bins=nbins,
                range=(bin_min,bin_max),
                density=False
            )

            histograms[cell_x_idx,cell_y_idx] = cell_histogram

    return histograms


hog = histogram_of_oriented_gradients
"""Alias for `histogram_of_oriented_gradients`"""
