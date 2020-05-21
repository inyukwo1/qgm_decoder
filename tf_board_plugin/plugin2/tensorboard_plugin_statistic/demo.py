"""Demo code.
This generates summary logs viewable by the raw scalars example plugin.
After installing the plugin (`python setup.py develop`), you can run TensorBoard
with logdir set to the `demo_logs` directory.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from absl import app
import tensorflow as tf


tf.compat.v1.enable_eager_execution()
tf = tf.compat.v2


def main(unused_argv):
    writer = tf.summary.create_file_writer("demo_logs")
    with writer.as_default():
        for i in range(100):
            tf.summary.scalar("custom_tag", 100 * math.sin(i), step=i)


if __name__ == "__main__":
    app.run(main)