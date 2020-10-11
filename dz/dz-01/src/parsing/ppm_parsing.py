from pathlib import Path

import numpy as np

from . import constants


class Ppm6Image:
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
    def file_type(self):
        return self._file_type

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def max_value(self):
        return self._max_value

    @property
    def data(self):
        return np.copy(self._data)

    # endregion
