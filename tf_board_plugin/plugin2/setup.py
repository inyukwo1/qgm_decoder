from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import setuptools

# Specifying setup.py makes a plugin installable via a Python package manager.
# `entry_points` is an important field makes plugins discoverable by TensorBoard
# at runtime.
# See https://packaging.python.org/specifications/entry-points/
setuptools.setup(
    name="tensorboard_plugin_statistic",
    version="0.1.0",
    description="Sample TensorBoard plugin.",
    packages=["tensorboard_plugin_statistic"],
    package_data={"tensorboard_plugin_statistic": ["static/**"],},
    entry_points={
        "tensorboard_plugins": [
            "statistic = tensorboard_plugin_statistic.plugin:StatisticPlugin",
        ],
    },
)
