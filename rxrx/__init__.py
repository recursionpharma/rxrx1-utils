from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys


def _add_tpu_models_to_path():
    dir = os.path.dirname(os.path.realpath(__file__))
    tpu_models_dir = os.path.abspath(os.path.join(dir, '..', 'tpu', 'models'))
    if tpu_models_dir not in sys.path:
        sys.path.insert(0, tpu_models_dir)


_add_tpu_models_to_path()
