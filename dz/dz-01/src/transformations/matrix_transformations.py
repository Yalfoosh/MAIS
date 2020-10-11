# Copyright 2020 Yalfoosh
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import numpy as np


def array_2d_to_zigzag(array_2d: np.ndarray):
    """
    Converts a 2D array into a 1D array by zigzag scanning.

    Example:
        [[1, 2, 3],
         [4, 5, 6],
         [7, 8, 9]]

         [[1, 2, 4],
          [7, 5, 3],
          [6, 8, 9]]
    :param array_2d:
        A np.ndarray of shape AxB.

    :return:
        A np.ndarray of shape 1xAB.
    """
    return np.concatenate(
        [
            np.diagonal(array_2d[::-1, :], k)[:: (2 * (k % 2) - 1)]
            for k in range(1 - array_2d.shape[0], array_2d.shape[0])
        ]
    )


def zigzag_pixel_blocks(pixel_blocks: np.ndarray):
    """
    Converts the 2D arrays in pixel blocks into 1D arrays by zigzag scanning.

    :param pixel_blocks:
        A np.ndarray of shape AxBx8x8x3, where A = H/8, B = W/8.

    :return:
        A np.ndarray of shape AxBx3x64, where A = H/8, B = W/8.
    """
    to_return = list()

    for pixel_block_row in pixel_blocks:
        current_row = list()

        for pixel_block in pixel_block_row:
            current_block = list()

            for pixel_sector in pixel_block.transpose(2, 0, 1):
                current_block.append(array_2d_to_zigzag(pixel_sector))

            current_row.append(current_block)

        to_return.append(current_row)

    return np.array(to_return)
