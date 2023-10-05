import numpy as np
import numpy.typing as npt


class Kernel:
    """A class for an immutable kernel to be used for an image convolution"""

    def __init__(self,
                 mat: npt.NDArray,
                 normalise: bool = True):

        assert mat.ndim == 2, "Kernel matrix must be a two-dimensional"
        assert mat.shape[0] == mat.shape[1], "Kernel matrix must be sqaure"
        assert mat.shape[0] % 2 == 1, "Kernel matrix must have odd number of rows and columns"

        if normalise:
            self.__mat = mat / np.sum(mat, axis=None)
        else:
            self.__mat = mat

    def __str__(self) -> str:
        return "[ " + " ".join(
            ["[" + " ".join(map(
                str,
                self.__mat[j,:].tolist()
            )) + "]" for j in range(self.mat_width)]
        ) + " ]"

    def __repr__(self) -> str:
        return self.__str__()

    def get_matrix(self) -> npt.NDArray:
        """Gets a copy of the kernel's matrix"""
        return self.__mat.copy()

    @property
    def range(self) -> int:
        """The "range" of the kernel. This is the number of cells from (but excluding) the center to any edge.

For example, for a kernel with a 3x3 matrix, the "range" returned will be 1
"""
        return self.__mat.shape[0] // 2

    @property
    def mat_width(self) -> int:
        """The number of rows and columns of the kernel's matrix"""
        return self.__mat.shape[0]

    def get_val(self, dx: int, dy: int):
        return self.__mat[self.range+dx, self.range+dy]

    def convolve_single(self, img: npt.NDArray) -> npt.NDArray:
        """Convolves a single image with this kernel and returns the output.

Parameters:

    img - the image represented as a matrix

Returns:

    out - the output image, having been convolved with this kernel
"""

        assert img.ndim == 2, "Image must be a two-dimensional array"
        assert img.shape[0] >= self.mat_width, "Input image width must be greater than the kernel's matrix width"
        assert img.shape[1] >= self.mat_width, "Input image height must be greater than the kernel's matrix height"

        out_width = img.shape[0] - (2 * self.range)
        out_height = img.shape[1] - (2 * self.range)

        out = np.zeros(shape=(out_width,out_height), dtype=img.dtype)

        for dy in range(-self.range, self.range+1):
            for dx in range(-self.range, self.range+1):

                shifted = np.roll(
                    np.roll(
                        img,
                        dy,
                        axis=0
                    ),
                    dx,
                    axis=1
                )[:-dy,:-dx]

                assert shifted.shape == out.shape

                out[:,:] += shifted * self.get_val(dx, dy)

        return out

        # TODO - make more efficient

    def convolve_multi(self, imgs: npt.NDArray) -> npt.NDArray:
        """Convolves multiple images with this kernel and returns the output.

Parameters:

    imgs - the images represented as matrices

Returns:

    outs - the output images, having been convolved with this kernel
"""

        assert imgs.ndim == 3, "Image must be a three-dimensional array"
        assert imgs.shape[1] >= self.mat_width, "Input image width must be greater than the kernel's matrix width"
        assert imgs.shape[2] >= self.mat_width, "Input image height must be greater than the kernel's matrix height"

        raise NotImplementedError()  # TODO
