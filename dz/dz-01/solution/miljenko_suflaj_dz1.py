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

import argparse
from pathlib import Path
import re
from textwrap import dedent

import numpy as np

# region Constants
WHITESPACE_REGEX = re.compile(r"\s")

DEFAULT_Y_COEFFICIENTS = (0.299, 0.587, 0.114)
DEFAULT_CB_COEFFICIENTS = (-0.1687, -0.3313, 0.5)
DEFAULT_CR_COEFFICIENTS = (0.5, -0.4187, -0.0813)
DEFAULT_Y_ADDITION = 0
DEFAULT_CB_ADDITION = 128
DEFAULT_CR_ADDITION = 128

DCT_C_ZERO_VAL = np.reciprocal(np.square(2))

K1_TABLE = (
    (16, 11, 10, 16, 24, 40, 51, 61),
    (12, 12, 14, 19, 26, 58, 60, 55),
    (14, 13, 16, 24, 40, 57, 69, 56),
    (14, 17, 22, 29, 51, 87, 80, 62),
    (18, 22, 37, 56, 68, 109, 103, 77),
    (24, 35, 55, 64, 81, 104, 113, 92),
    (49, 64, 78, 87, 103, 121, 120, 101),
    (72, 92, 95, 98, 112, 100, 103, 99),
)

K2_TABLE = (
    (17, 18, 24, 47, 99, 99, 99, 99),
    (18, 21, 26, 66, 99, 99, 99, 99),
    (24, 26, 56, 99, 99, 99, 99, 99),
    (47, 66, 99, 99, 99, 99, 99, 99),
    (99, 99, 99, 99, 99, 99, 99, 99),
    (99, 99, 99, 99, 99, 99, 99, 99),
    (99, 99, 99, 99, 99, 99, 99, 99),
    (99, 99, 99, 99, 99, 99, 99, 99),
)

# endregion


# region Classes
class Ppm6Image:
    def __init__(self, image_path: Path or str):
        with open(image_path, mode="rb") as file:
            self.file_type = file.readline().decode("utf8").strip()
            self.width, self.height = (
                int(x)
                for x in WHITESPACE_REGEX.split(file.readline().decode("utf8").strip())
            )
            self.max_value = int(file.readline().decode("utf8").strip())

            bytes_per_color = 1 if self.max_value < 256 else 2
            bytes_per_pixel = 3 * bytes_per_color

            read_content = file.read()
            read_rows = [
                read_content[i : i + (self.width * bytes_per_pixel)]
                for i in range(0, len(read_content), self.width * bytes_per_pixel)
            ]
            read_pixels = [
                [
                    [int(element) for element in read_row[i : i + bytes_per_pixel]]
                    for i in range(0, len(read_row), bytes_per_pixel)
                ]
                for read_row in read_rows
            ]

            self.data = np.array(
                read_pixels, dtype=np.uint8 if bytes_per_color == 1 else np.uint16
            )


# endregion

# region Methods
def rgb_to_ycbcr(
    pixel_data: np.ndarray,
    y_coefficients=DEFAULT_Y_COEFFICIENTS,
    cb_coefficients=DEFAULT_CB_COEFFICIENTS,
    cr_coefficients=DEFAULT_CR_COEFFICIENTS,
    y_addition=DEFAULT_Y_ADDITION,
    cb_addition=DEFAULT_CB_ADDITION,
    cr_addition=DEFAULT_CR_ADDITION,
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
                    c = (DCT_C_ZERO_VAL if u == 0 else 1) * (
                        DCT_C_ZERO_VAL if v == 0 else 1
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


def dct_2d(pixel_blocks: np.ndarray):
    to_return = list()

    for row in pixel_blocks:
        current_row = list()

        for pixel_block in row:
            current_row.append(dct_2d_on_8x8_block(pixel_block))

        to_return.append(current_row)

    return np.array(to_return)


def get_quantization_tensor(y_table=K1_TABLE, cb_table=K2_TABLE, cr_table=K2_TABLE):
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


# endregion

# region Parsing
parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument(
    "image_path",
    type=str,
    metavar="STRING",
    default=None,
    help=dedent(
        f"""
        [Optional]
        A string representing the path to the PPM6 formatted file
        you want to work with.
        If not specified, nothing will run.
        """
    ),
)

parser.add_argument(
    "block_index",
    type=int,
    metavar="UINT",
    default=None,
    help=dedent(
        f"""
        [Optional]
        An unsigned integer representing the 0-based index of the
        block you wish to print after quantization.
        If not specified, this step will be skipped.
        """
    ),
)

parser.add_argument(
    "output_path",
    type=str,
    metavar="STRING",
    default=None,
    help=dedent(
        f"""
        [Optional]
        A string representing the path to the program output.
        If not specified, this step will be skipped.
        """
    ),
)

# endregion


def main():
    args = parser.parse_args()

    if args.image_path is None:
        exit(4)

    image = Ppm6Image(args.image_path)

    image_ycbcr = rgb_to_ycbcr(image.data)
    shifted_ycbcr = shift_image_pixels(image_ycbcr, -128)
    pixel_blocks = divide_image_to_blocks(shifted_ycbcr)

    dct_blocks = dct_2d(pixel_blocks)

    quantization_tensor = get_quantization_tensor()
    quantized_blocks = quantize(dct_blocks, quantization_tensor)

    if args.block_index is not None:
        if args.block_index < (quantized_blocks.shape[0] * quantized_blocks.shape[1]):
            print(
                quantized_blocks[args.block_index // quantized_blocks.shape[0]][
                    args.block_index % quantized_blocks.shape[0]
                ].transpose(2, 0, 1)
            )
        else:
            raise IndexError(
                dedent(
                    f"""\
                    Block index {args.block_index} is out of bounds for pixel block \
                    shape of {quantized_blocks.shape[0]} x {quantized_blocks.shape[1]}!\
                    """
                )
            )

    zigzagged_blocks = zigzag_pixel_blocks(quantized_blocks)

    blocks_to_write = zigzagged_blocks.reshape((-1, 3, 64)).transpose((1, 0, 2))

    if args.output_path is not None:
        with open(args.output_path, mode="w+", encoding="ascii") as file:
            to_write = f"{image.width} x {image.height}"

            for component_block_to_write in blocks_to_write:
                to_write += "\n"
                to_write += "\n".join(
                    [
                        " ".join([str(int(x)) for x in row])
                        for row in component_block_to_write
                    ]
                )
                to_write += "\n"

            file.write(to_write)


if __name__ == "__main__":
    main()
