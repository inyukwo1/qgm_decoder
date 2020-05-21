# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
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
    dataset_name = "spider"
    writer = tf.summary.create_file_writer("demo_logs/testing1")
    with writer.as_default():
        tf.summary.write("datasets", dataset_name, step=0)
        tf.summary.write("{}_path".format(dataset_name), "/home/hkkang/{}".format(dataset_name), step=0)
        examples = [{
            "query": "what is your name?",
            "query_type": ["none", "none", "none", "none"],
            "db": ["academic"],
            "schema": ["student", "teacher", "school"],
            "gold": "SELECT * FROM table WHERE name=kyle",
            "pred": "SELECT * FROM table WHERE name=dave",
        }, {
            "query": "my name is kyle",
            "query_type": ["none", "none", "none", "none"],
            "db": ["student_1"],
            "schema": ["student", "teacher", "school"],
            "gold": "SELECT * FROM table WHERE name=kyle2",
            "pred": "SELECT * FROM table WHERE name=dave2",
        }]
        for i, example in enumerate(examples):
            tf.summary.write("{}_query".format(dataset_name), example["query"], step=i)
            tf.summary.write("{}_query_type".format(dataset_name), example["query_type"], step=i)
            tf.summary.write("{}_db".format(dataset_name), example["db"], step=i)
            tf.summary.write("{}_schema".format(dataset_name), example["schema"], step=i)
            tf.summary.write("{}_pred".format(dataset_name), example["pred"], step=i)
            tf.summary.write("{}_gold".format(dataset_name), example["gold"], step=i)
            tf.summary.scalar("custom_tag", 100 * math.sin(i), step=i)

if __name__ == "__main__":
    app.run(main)
