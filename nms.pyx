# nms.pyx
import numpy as np
cimport numpy as cnp
from libc.stdlib cimport malloc, free
import cython
from cython.parallel import prange

# Define the data types for numpy arrays
DTYPE = np.int32
ctypedef cnp.int32_t DTYPE_t
ctypedef cnp.float32_t DTYPE_t_f
@cython.boundscheck(False)
@cython.wraparound(False)
def nms(cnp.ndarray[DTYPE_t, ndim=2] px_id_pos,
        cnp.ndarray[DTYPE_t, ndim=1] px_id,
        cnp.ndarray[DTYPE_t_f, ndim=2] pred_pos,
        cnp.ndarray[DTYPE_t_f, ndim=1] score,
        cnp.ndarray[DTYPE_t, ndim=1] stop_id,
        int H, int W, int est_cnt, int px_id_len, float iou_threshold):

    cdef int n = 0
    cdef list pt_list = []
    cdef int i, t
    cdef dict pt_dict
    cdef bint T
    cdef list pred_points = []


    for i in range(px_id_len):
        pt_dict = {}
        t = stop_id[i]
        idpx = px_id_pos[i, :t+1]
        pred = pred_pos[i, :t+1]
        # pt_dict["pos_id"] = idpx
        # pt_dict["pos_cnt"] = pred
        pt_dict["id2cnt"]={k:v for k,v in zip(idpx,pred)}
        pt_dict["cnt"] = (np.sum(pred))
        pt_dict["loc"] = px_id[i]
        pt_dict["i"] = px_id[i] // W
        pt_dict["j"] = px_id[i] % W
        pt_dict["score"] = score[i]

        T = True
        for previous_pt in pt_list:
            if iou(previous_pt, pt_dict) > iou_threshold:
                T = False
                break
        if T:
            if n >= est_cnt:
                break
            else:
                pt_list.append(pt_dict)
                n += 1
    pred_points = []
    for pt in pt_list:
        i, j = pt["i"], pt["j"]
        pred_points.append([2 * j + 1.5, 2 * i + 1.5])

    return pred_points

@cython.boundscheck(False)
@cython.wraparound(False)
cdef float iou(dict pt_dict1, dict pt_dict2):
    cdef int i1, j1, i2, j2
    cdef float c1, c2, I

    i1, j1 = pt_dict1["i"], pt_dict1["j"]
    i2, j2 = pt_dict2["i"], pt_dict2["j"]
    if abs(i1 - i2) >= 17 or abs(j1 - j2) >= 17:
        return 0.0

    c1 = pt_dict1["cnt"]
    c2 = pt_dict2["cnt"]

    I = sum([pt_dict2["id2cnt"][k] for k in pt_dict1["id2cnt"].keys()&pt_dict2["id2cnt"].keys()])
    iou = I / (c1 + c2 - I + 1e-6)

    return iou
