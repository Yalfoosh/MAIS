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

DEFAULT_Y_COEFFICIENTS = (0.299, 0.587, 0.114)
DEFAULT_CB_COEFFICIENTS = (-0.1687, -0.3313, 0.5)
DEFAULT_CR_COEFFICIENTS = (0.5, -0.4187, -0.0813)
DEFAULT_Y_ADDITION = 0
DEFAULT_CB_ADDITION = 128
DEFAULT_CR_ADDITION = 128

DCT_C_ZERO_VAL = np.reciprocal(np.square(2))
