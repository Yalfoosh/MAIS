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

from pathlib import Path

import numpy as np

from . import constants


class Ppm6Image:
    """
    A class for easier handling of PPM6 images.
    """

    def __init__(self, image_path: Path or str):
        with open(image_path, mode="rb") as file:
            self._file_type = file.readline().decode("utf8").strip()
            self._width, self._height = (
                int(x)
                for x in constants.WHITESPACE_REGEX.split(
                    file.readline().decode("utf8").strip()
                )
            )
            self._max_value = int(file.readline().decode("utf8").strip())

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

            self._data = np.array(
                read_pixels, dtype=np.uint8 if bytes_per_color == 1 else np.uint16
            )

    # region Properties
    @property
    def file_type(self) -> str:
        """
        The file type property.

        :return:
            A string representing the file type (should be only "P6").
        """
        return self._file_type

    @property
    def width(self) -> int:
        """
        The image width property.

        :return:
            An int representing the image width.
        """
        return self._width

    @property
    def height(self) -> int:
        """
        The image height property.

        :return:
            An int representing the image height.
        """
        return self._height

    @property
    def max_value(self) -> int:
        """
        The maximum value property.

        :return:
            An int representing the maximum value found in the image.
        """
        return self._max_value

    @property
    def data(self) -> np.ndarray:
        """
        The image data.

        :return:
            A np.ndarray of shape HxWx3 representing the image data.
        """
        return np.copy(self._data)

    # endregion
