import sys
import numpy as np
from numba import prange, jit


@jit(nopython=True, parallel=True, fastmath=True)
def naive_stereo_estimation(window_size, height, width, left_image,
                            right_image, scale, cost_type):
    naive_disparities = np.zeros((height, width), dtype=np.uint8)
    half_window_size = window_size // 2

    for i in prange(half_window_size, height - half_window_size):
        for j in range(half_window_size, width - half_window_size):
            cmin = sys.maxsize
            disparity = 0

            for d in range(-j + half_window_size,
                           width - j - half_window_size):
                left_sub = left_image[
                           i - half_window_size: i + half_window_size + 1,
                           j - half_window_size: j + half_window_size + 1,
                           ]
                right_sub = right_image[
                            i - half_window_size: i + half_window_size + 1,
                            j - half_window_size + d: j + half_window_size + 1 + d,
                            ]

                if cost_type == "ssd":
                    c = np.sum((left_sub - right_sub) ** 2)

                if cost_type == "sad":
                    c = np.sum(np.abs(left_sub - right_sub))

                if cost_type == "mse":
                    c = np.mean((left_sub - right_sub) ** 2)

                if cost_type == "rmse":
                    c = np.sqrt(np.mean((left_sub - right_sub) ** 2))

                if c < cmin:
                    cmin = c
                    disparity = d

            naive_disparities[i - half_window_size, j - half_window_size] = (
                    abs(disparity) * scale
            )
    return naive_disparities


@jit(nopython=True, parallel=True, fastmath=True)
def dynamic_programming_stereo_estimation(
        window_size, height, width, left_image, right_image, scale, weight,
        cost_type
):
    dp_disparities = np.zeros((height, width), dtype=np.uint8)
    half_window_size = window_size // 2

    for row in prange(half_window_size, height - half_window_size):
        C = np.zeros((width, width), dtype=np.float64)
        M = np.ones((width, width), dtype=np.uint8)

        for i in prange(width):
            C[i, 0] = i * weight
            C[0, i] = i * weight

        for i in prange(half_window_size, width - half_window_size):
            for j in prange(half_window_size, width - half_window_size):
                left_sub = left_image[
                           row - half_window_size: row + half_window_size + 1,
                           i - half_window_size: i + half_window_size + 1,
                           ]
                right_sub = right_image[
                            row - half_window_size: row + half_window_size + 1,
                            j - half_window_size: j + half_window_size + 1,
                            ]

                if cost_type == "ssd":
                    c = np.sum((left_sub - right_sub) ** 2)

                if cost_type == "sad":
                    c = np.sum(np.abs(left_sub - right_sub))

                if cost_type == "mse":
                    c = np.mean((left_sub - right_sub) ** 2)

                if cost_type == "rmse":
                    c = np.sqrt(np.mean((left_sub - right_sub) ** 2))


                min1 = C[i - 1, j - 1] + c
                min2 = C[i - 1, j] + weight
                min3 = C[i, j - 1] + weight

                cmin = min(min1, min2, min3)
                C[i, j] = cmin
                if cmin == min1:
                    M[i, j] = 1
                if cmin == min2:
                    M[i, j] = 2
                if cmin == min3:
                    M[i, j] = 3

        p = width - 1
        q = width - 1
        while p != 0 and q != 0:
            if M[p, q] == 1:
                # p matches q
                dp_disparities[row, p] = abs(p - q) * scale
                p -= 1
                q -= 1
            if M[p, q] == 2:
                # p is unmatched
                p -= 1
            if M[p, q] == 3:
                # q is unmatched
                q -= 1
    return dp_disparities
