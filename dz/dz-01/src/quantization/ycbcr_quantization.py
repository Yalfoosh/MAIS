import numpy as np

from . import constants


def get_quantization_tensor(
    y_table=constants.K1_TABLE, cb_table=constants.K2_TABLE, cr_table=constants.K2_TABLE
):
    return np.stack([y_table, cb_table, cr_table]).transpose((1, 2, 0))


def quantize_pixel_block(pixel_block: np.ndarray, quantization_tensor: np.ndarray):
    return np.rint(pixel_block / quantization_tensor).astype(int)


def quantize(pixel_blocks: np.ndarray, quantization_tensor: np.ndarray):
    to_return = list()

    for pixel_block_row in pixel_blocks:
        current_row = list()

        for pixel_block in pixel_block_row:
            current_row.append(quantize_pixel_block(pixel_block, quantization_tensor))

        to_return.append(current_row)

    return np.array(to_return)
