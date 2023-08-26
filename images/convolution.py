import numpy as np
from scipy.signal import convolve2d


SOBEL_X = np.array([
    [ -1,  0,  1 ],
    [ -2,  0,  2 ],
    [ -1,  0,  1 ],
])

SOBEL_Y = np.array([
    [ -1, -2, -1 ],
    [  0,  0,  0 ],
    [  1,  2,  1 ],
])

SOBEL_COMPLEX = np.array([
    [ -1-1j,  0-1j, +1-1j ],
    [ -1+0j,  0+0j, +1+0j ],
    [ -1+1j,  0+1j, +1+1j ],
], dtype=np.complex128)

DERIVATIVE_X = np.array([
    [ -1, 0, 1 ]
])

DERIVATIVE_Y = np.array([
    [ -1 ],
    [  0 ],
    [  1 ]
])
