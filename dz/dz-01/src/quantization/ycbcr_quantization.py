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

from . import constants


def get_quantization_tensor(
    y_table=constants.K1_TABLE, cb_table=constants.K2_TABLE, cr_table=constants.K2_TABLE
) -> np.ndarray:
    """
    Gets a tensor used to quantize DCT blocks.

    :param y_table:
        A table representing the quantization table for the Y component of an image.
    :param cb_table:
        A table representing the quantization table for the Cb component of an image.
    :param cr_table:
        A table representing the quantization table for the Cr component of an image.

    :return:
        A np.ndarray of shape 8x8x3 used to quantize DCT blocks.
    """
    return np.stack([y_table, cb_table, cr_table]).transpose((1, 2, 0))


def quantize_pixel_block(
    pixel_block: np.ndarray, quantization_tensor: np.ndarray
) -> np.ndarray:
    """
    Quantizes a single pixel block.

    :param pixel_block:
        A np.ndarray of shape 8x8x3 you wish to quantize.
    :param quantization_tensor:
        A np.ndarray of shape 8x8x3 you wish to quantize with.

    :return:
        A np.ndarray of shape 8x8x3: the quantization result.
    """
    return np.rint(pixel_block / quantization_tensor).astype(int)


def quantize(pixel_blocks: np.ndarray, quantization_tensor: np.ndarray) -> np.ndarray:
    """
    Quantizes an image comprised of pixel blocks.

    :param pixel_blocks:
        A np.ndarray of shape AxBx8x8x3, where 8A and 8B are the height and width of
        the original image, respectively.
    :param quantization_tensor:
        A np.ndarray of shape 8x8x3 you with to quantize the pixel blocks with.

    :return:
        A np.ndarray of shape AxBx8x8x3: the quantization result.
    """
    to_return = list()

    for pixel_block_row in pixel_blocks:
        current_row = list()

        for pixel_block in pixel_block_row:
            current_row.append(quantize_pixel_block(pixel_block, quantization_tensor))

        to_return.append(current_row)

    return np.array(to_return)
