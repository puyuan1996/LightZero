#  cython --cplus -a -I /Users/puyuan/opt/anaconda3/lib/python3.8/site-packages/numpy/core/include -o legal_actions.c legal_actions.pyx

import numpy as np
cimport numpy as np
np.import_array()

cpdef list legal_actions(np.ndarray[np.int32_t, ndim=2] board, int board_size):
    cdef np.ndarray[np.int32_t, ndim=1] zero_positions = np.argwhere(board == 0)
    cdef list legal_actions = [i * board_size + j for i, j in zero_positions]
    return legal_actions

