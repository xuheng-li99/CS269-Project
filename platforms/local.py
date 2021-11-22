# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import pathlib

from platforms import base


class Platform(base.Platform):
    @property
    def root(self):
        return os.path.join(pathlib.Path.home(), 'open_lth_data2')

    @property
    def dataset_root(self):
        return os.path.join(pathlib.Path.home(), 'open_lth_datasets2')

    @property
    def imagenet_root(self):
        raise NotImplementedError

    @property
    def tinyimagenet_root(self):
        raise NotImplementedError

