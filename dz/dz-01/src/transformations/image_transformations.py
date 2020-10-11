from sys import stdout

import numpy as np
from tqdm import tqdm

from . import constants


def rgb_to_ycbcr(
    pixel_data: np.ndarray,
    y_coefficients=constants.DEFAULT_Y_COEFFICIENTS,
    cb_coefficients=constants.DEFAULT_CB_COEFFICIENTS,
    cr_coefficients=constants.DEFAULT_CR_COEFFICIENTS,
    y_addition=constants.DEFAULT_Y_ADDITION,
    cb_addition=constants.DEFAULT_CB_ADDITION,
    cr_addition=constants.DEFAULT_CR_ADDITION,
):
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


def shift_image_pixels(pixel_data: np.ndarray, value=-128):
    return pixel_data + np.full(pixel_data.shape, value)


def divide_image_to_blocks(
    pixel_data: np.ndarray, block_width: int = 8, block_height: int = 8
):
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


def dct_2d_on_8x8_block(pixel_block: np.ndarray):
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


def dct_2d(pixel_blocks: np.ndarray, verbose: int = 0):
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
