import numpy as np


def array_2d_to_zigzag(array_2d: np.ndarray):
    return np.concatenate(
        [
            np.diagonal(array_2d[::-1, :], k)[:: (2 * (k % 2) - 1)]
            for k in range(1 - array_2d.shape[0], array_2d.shape[0])
        ]
    )


def zigzag_pixel_blocks(pixel_blocks: np.ndarray):
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
