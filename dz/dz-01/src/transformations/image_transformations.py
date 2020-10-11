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

from sys import stdout
from typing import Union

import numpy as np
from tqdm import tqdm

from . import constants


def rgb_to_ycbcr(
    pixel_data: np.ndarray,
    y_coefficients=constants.DEFAULT_Y_COEFFICIENTS,
    cb_coefficients=constants.DEFAULT_CB_COEFFICIENTS,
    cr_coefficients=constants.DEFAULT_CR_COEFFICIENTS,
    y_addition: Union[float, int] = constants.DEFAULT_Y_ADDITION,
    cb_addition: Union[float, int] = constants.DEFAULT_CB_ADDITION,
    cr_addition: Union[float, int] = constants.DEFAULT_CR_ADDITION,
) -> np.ndarray:
    """
    Converts a RGB matrix into a YCbCr matrix.

    :param pixel_data:
        A np.ndarray of shape HxWx3 you wish to convert to YCbCr.
    :param y_coefficients:
        A list of 3 RGB weights for the Y component.
    :param cb_coefficients:
        A list of 3 RGB weights for the Cb component.
    :param cr_coefficients:
        A list of 3 RGB weights for the Cr component.
    :param y_addition:
        A float or int representing the number to be added after weight-summing to get
        the Y component.
    :param cb_addition:
        A float or int representing the number to be added after weight-summing to get
        the Cb component.
    :param cr_addition:
        A float or int representing the number to be added after weight-summing to get
        the Cr component.

    :return:
        A np.ndarray of shape HxWx3: the image in YCbCr color space.
    """
    y_coefficients, cb_coefficients, cr_coefficients = (
        np.array(x) for x in (y_coefficients, cb_coefficients, cr_coefficients)
    )

    return np.array(
        [
            [
                [
                    y_coefficients @ pixel + y_addition,
                    cb_coefficients @ pixel + cb_addition,
                    cr_coefficients @ pixel + cr_addition,
                ]
                for pixel in row
            ]
            for row in pixel_data
        ]
    )


def shift_image_pixels(pixel_data: np.ndarray, value=-128) -> np.ndarray:
    """
    Shifts all values in an array by value.

    :param pixel_data:
        A np.ndarray of shape S (any shape).
    :param value:
        An int representing the value to be added to all array elements.

    :return:
        A np.ndarray of shape S.
    """
    return pixel_data + np.full(pixel_data.shape, value)


def divide_image_to_blocks(
    pixel_data: np.ndarray, block_width: int = 8, block_height: int = 8
) -> np.ndarray:
    """
    Divides an image into block_width x block_height blocks.

    :param pixel_data:
        A np.ndarray of shape HxWx3.
    :param block_width:
        An int representing the width of a block.
    :param block_height:
        An int representing the height of a block.

    :return:
        A np.ndarray of shape AxBx8x8x3, where A = H/8 and B = W/8
    """
    to_return = list()

    for h_offset in range(0, pixel_data.shape[0] - block_height + 1, block_height):
        current_row = list()

        for w_offset in range(0, pixel_data.shape[1] - block_width + 1, block_width):
            current_row.append(
                pixel_data[
                    h_offset : h_offset + block_height,
                    w_offset : w_offset + block_width,
                ]
            )

        to_return.append(current_row)

    return np.array(to_return)


def dct_2d_on_8x8_block(pixel_block: np.ndarray) -> np.ndarray:
    """
    Does 2D DCT on a single 8x8 block.

    :param pixel_block:
        A np.ndarray of shape 8x8x3.

    :return:
        A np.ndarray of shape 8x8x3: the 2D DCT result.
    """
    to_return = list()

    for i, pixel_block_row in enumerate(pixel_block):
        current_row = list()

        for j, pixel in enumerate(pixel_block_row):
            current_value = 0.0

            for u in range(8):
                for v in range(8):
                    c = (constants.DCT_C_ZERO_VAL if u == 0 else 1) * (
                        constants.DCT_C_ZERO_VAL if v == 0 else 1
                    )

                    current_value += (
                        c
                        * pixel
                        * np.cos(((2 * i + 1) * u * np.pi) / 16)
                        * np.cos(((2 * j + 1) * v * np.pi) / 16)
                    )

            current_row.append(current_value / 4)

        to_return.append(current_row)

    return np.array(to_return)


def dct_2d(pixel_blocks: np.ndarray, verbose: int = 0) -> np.ndarray:
    """
    Does 8x8 2D DCT on an image represented by pixel blocks.

    :param pixel_blocks:
        A np.ndarray of shape AxBx8x8x3, where A = H/8, B = W/8.
    :param verbose:
        An int; if greater than 0 will print out a tqdm progress bar.

    :return:
        A np.ndarray of shape AxBx8x8x3, where A = H/8, B = W/8.
    """
    to_return = list()

    if verbose > 0:
        pbar = tqdm(total=pixel_blocks.shape[0] * pixel_blocks.shape[1], file=stdout)

    for row in pixel_blocks:
        current_row = list()

        for pixel_block in row:
            current_row.append(dct_2d_on_8x8_block(pixel_block))

            if verbose > 0:
                pbar.update()

        to_return.append(current_row)

    if verbose > 0:
        pbar.close()

    return np.array(to_return)
