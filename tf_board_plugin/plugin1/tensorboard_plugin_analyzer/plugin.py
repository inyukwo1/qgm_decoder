# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""A sample plugin to demonstrate reading scalars."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import mimetypes
import os
import pickle
import base64

import six
from werkzeug import wrappers

from tensorboard import errors
from tensorboard.backend import http_util
from tensorboard.plugins import base_plugin
from tensorboard.plugins.scalar import metadata

_SCALAR_PLUGIN_NAME = metadata.PLUGIN_NAME
_PLUGIN_DIRECTORY_PATH_PART = "/data/plugin/QGM_analyzer/"


class AnalyzerPlugin(base_plugin.TBPlugin):
    """Raw summary example plugin for TensorBoard."""

    plugin_name = "QGM_analyzer"

    def __init__(self, context):
        """Instantiates ExampleRawScalarsPlugin.

        Args:
          context: A base_plugin.TBContext instance.
        """
        self._multiplexer = context.multiplexer
        self._logs = {}

    def get_plugin_apps(self):
        return {
            "/scalars": self.scalars_route,
            "/tags": self._serve_tags,
            "/models": self._serve_models,
            "/datasets": self.serve_datasets,
            "/data_indices": self.serve_data_indices,
            "/data": self.serve_data,
            "/image": self.serve_image,
            "/tensor": self.serve_tensor,
            "/inference": self.serve_inference,
            "/static/*": self._serve_static_file,
        }

    @wrappers.Request.application
    def serve_inference(self, request):
        model = request.args.get('model')
        dataset_name = request.args.get("dataset")
        data_idx = int(request.args.get("index"))

        try:
            log = self._logs[model]

            # To-Do: Check dataset for each data
            assert dataset_name in [tmp['name'] for tmp in log['dataset']]
            data = log['data'][data_idx]

            # Get inference score tensors
            inference = data['inference']

            body = {}
            for idx, image_path in enumerate(inference):
                with open(image_path, 'rb') as f:
                    encoded_img = base64.b64encode(f.read())
                    body['inference step {}'.format(idx)] = encoded_img

        except:
            raise RuntimeError("inference info not found")

        return http_util.Respond(request, body, "application/json")

    @wrappers.Request.application
    def serve_tensor(self, request):
        model = request.args.get("model")
        dataset_name = request.args.get("dataset")
        data_idx = int(request.args.get("index"))
        try:
            log = self._logs[model]

            # To-Do: Check dataset for each data
            assert dataset_name in [tmp['name'] for tmp in log['dataset']]
            data = log['data'][data_idx]

            # Get weight tensors
            weight_tensors = [value for key, value in data.items() if 'weight_tensors' in key and 'relation' not in key]
            relation_weight_tensors = [value for key, value in data.items() if 'relation_weight_tensors' in key]

            body = {}
            # Create image for weight tensors
            for layer_idx, layer in enumerate(weight_tensors):
                key = 'weight_tensor_layer_{}'.format(layer_idx)
                body[key] = []
                for head_idx, head_path in enumerate(layer):
                    # Read image
                    with open(head_path, "rb") as f:
                        encoded_img = base64.b64encode(f.read())
                        body[key] += [encoded_img]

            for layer_idx, layer in enumerate(relation_weight_tensors):
                key = 'relation_weight_tensor_layer_{}'.format(layer_idx)
                body[key] = []
                for head_idx, head_path in enumerate(layer):
                    # Read image
                    with open(head_path, "rb") as f:
                        encoded_img = base64.b64encode(f.read())
                        body[key] += [encoded_img]

        except:
            raise RuntimeError("No tensor found")

        return http_util.Respond(request, body, "application/json")

    @wrappers.Request.application
    def serve_image(self, request):
        model = request.args.get("requestedModel")
        dataset_name = request.args.get("requestedDataset")
        db = request.args.get("requestedDB")

        try:
            # Get dataset path
            log = self._logs[model]

            # To-Do: Check dataset for each data
            assert dataset_name in [tmp['name'] for tmp in log['dataset']]
            dataset_path = [dataset['path'] for dataset in log['dataset'] if dataset['name'] == dataset_name][0]

            # Read schema image
            image_path = "{}/schema_images/{}.png".format(dataset_path, db)

            with open(image_path, 'rb') as f:
                encoded_string = base64.b64encode(f.read())

            encoded_image = {
                "image": encoded_string
            }

        except:
            raise RuntimeError("No dataset path found")

        return http_util.Respond(request, encoded_image, "application/json")

    @wrappers.Request.application
    def _serve_tags(self, request):
        """Serves run to tag info.

        Frontend clients can use the Multiplexer's run+tag structure to request data
        for a specific run+tag. Responds with a map of the form:
        {runName: [tagName, tagName, ...]}
        """
        run_tag_mapping = self._multiplexer.PluginRunToTagToContent(
            _SCALAR_PLUGIN_NAME
        )
        run_info = {
            run: list(tags) for (run, tags) in six.iteritems(run_tag_mapping)
        }
        return http_util.Respond(request, run_info, "application/json")

    @wrappers.Request.application
    def _serve_models(self, request):
        paths = self._multiplexer._paths
        tag_names = {}
        for tag, path in paths.items():
            if 'result.pkl' in os.listdir(path):
                tag_names[tag] = tag
                # Read in
                with open(os.path.join(path, 'result.pkl'), "rb") as f:
                    self._logs[tag] = pickle.load(f)
        return http_util.Respond(request, tag_names, "application/json")

    @wrappers.Request.application
    def serve_datasets(self, request):
        """Serves example to indices info.

        Frontend clients can use the Multiplexer's index to request data
        for a specific example. Responds with a map of the form:
        {"nl": Str, "nl_type": str, "schema": image}
        """
        # Load datasets
        model = request.args.get("model")
        datasets = {}
        try:
            for key, log in self._logs.items():
                if key == model:
                    for dataset in log['dataset']:
                        dataset_name = dataset['name']
                        datasets[dataset_name] = dataset_name
        except:
            raise RuntimeError("No datasets for model: {}".format(model))

        return http_util.Respond(request, datasets, "application/json")

    @wrappers.Request.application
    def serve_data_indices(self, request):
        model = request.args.get("model")
        dataset = request.args.get("dataset")
        try:
            log = self._logs[model]
            # To-Do: Check dataset for each data
            assert dataset in [tmp['name'] for tmp in log['dataset']]

            # Get total number of datas
            num = len(log['data'])
            data_indices = {str(idx): str(idx) for idx in range(num)}
        except:
            raise RuntimeError("no data for model:{} dataset:{}",format(model, dataset))

        return http_util.Respond(request, data_indices, "application/json")


    @wrappers.Request.application
    def _serve_static_file(self, request):
        """Returns a resource file from the static asset directory.

        Requests from the frontend have a path in this form:
        /data/plugin/example_raw_scalars/static/foo
        This serves the appropriate asset: ./static/foo.

        Checks the normpath to guard against path traversal attacks.
        """
        static_path_part = request.path[len(_PLUGIN_DIRECTORY_PATH_PART) :]
        resource_name = os.path.normpath(
            os.path.join(*static_path_part.split("/"))
        )
        if not resource_name.startswith("static" + os.path.sep):
            return http_util.Respond(
                request, "Not found", "text/plain", code=404
            )

        resource_path = os.path.join(os.path.dirname(__file__), resource_name)
        with open(resource_path, "rb") as read_file:
            mimetype = mimetypes.guess_type(resource_path)[0]
            return http_util.Respond(
                request, read_file.read(), content_type=mimetype
            )

    @wrappers.Request.application
    def serve_data(self, request):
        model = request.args.get("model")
        dataset = request.args.get("dataset")
        data_idx = int(request.args.get("index"))

        try:
            log = self._logs[model]

            # To-Do: Check dataset for each data
            assert dataset in [tmp['name'] for tmp in log['dataset']], "dataset doesn't match"

            info = log['data'][data_idx]
            assert info['idx'] == data_idx, "data index doesn't match"

            query = info['query']
            columns = info['columns']
            tables = info['tables']
            db = info['db']
            gold = info['gold']
            pred = info['pred']
            sql = info['sql']

            body = {
                "query": query,
                "columns": columns,
                "tables": tables,
                "db": db,
                "sql": sql,
                "gold": gold,
                "pred": pred,
            }

        except:
            raise RuntimeError("No data found")

        return http_util.Respond(request, body, "application/json")

    @wrappers.Request.application
    def serve_dataset_path(self, request):
        model = request.args.get("model")
        dataset = request.args.get("dataset")
        # Get dataset path
        try:
            tensor_event = self._multiplexer.Tensors(model, "{}_path".format(dataset))[0]
            path = tensor_event.tensor_proto.string_val[0].decode("utd-8")
        except KeyError:
            raise errors.NotFoundError("No dataset path found")

        dataset_path = {
            'path': path
        }
        return http_util.Respond(request, dataset_path, "application/json")

    def is_active(self):
        """Returns whether there is relevant data for the plugin to process.

        When there are no runs with scalar data, TensorBoard will hide the plugin
        from the main navigation bar.
        """
        return bool(
            self._multiplexer.PluginRunToTagToContent(_SCALAR_PLUGIN_NAME)
        )

    def frontend_metadata(self):
        return base_plugin.FrontendMetadata(es_module_path="/static/index.js")

    def get_dataset_path(self, run, dataset):
        try:
            tensor_event = self._multiplexer.Tensors(run, "{}_path".format(dataset))[0]
            value = tensor_event.tensor_proto.string_val[0].decode("utf-8")
        except KeyError:
            raise errors.NotFoundError("No dataset recorded")
        return value

    @wrappers.Request.application
    def scalars_route(self, request):
        run = request.args.get("run")
        return http_util.Respond(request, [], "application/json")
