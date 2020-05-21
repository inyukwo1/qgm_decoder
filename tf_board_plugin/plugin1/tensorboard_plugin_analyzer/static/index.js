// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

import * as Model from './model.js';
import * as Views from './views.js';
import * as Dataset from './dataset.js';
import * as Data from './data.js';

/**
 * The main entry point of any TensorBoard iframe plugin.
 * It builds UI in this form:
 *   <link rel="stylesheet" href="./static/style.css" />
 *
 *   <h1>Example plugin - Select a run</h1>
 *   <select class="run-selector"></select>
 *   <div>${previewElements}</div>
 */
export async function render() {
  const stylesheet = document.createElement('link');
  stylesheet.rel = 'stylesheet';
  stylesheet.href = './static/style.css';
  document.body.appendChild(stylesheet);

  const header = document.createElement('h1');
  header.textContent = 'QGM Model Analyzer - Select model, dataset, and data_index';
  document.body.appendChild(header);

  // Model Selector
  const models = await Model.getModels();
  const modelSelector = Views.createModelSelector(models);

  // Datasets Selector
  const requestedModel = modelSelector.value;
  const datasets = await Dataset.getDatasets(requestedModel);
  const datasetSelector = Views.createDatasetSelector(datasets);

  // Data Selector
  const requestedDataset = datasetSelector.value;
  const datasetIndices = await Data.getDataIndices(requestedModel, requestedDataset);
  const dataIndexSelector = Views.createDataSelector(datasetIndices);

  // Data
  const previewContainer = document.createElement('div');
  updateDataText(modelSelector, datasetSelector, dataIndexSelector, previewContainer);

  // Binding
  const updateDataSelectorBound = updateDatasetSelector.bind(
      null,
      modelSelector,
      datasetSelector,
      dataIndexSelector,
      previewContainer,
  )
  const updateDataIndexSelectorBound = updateDataIndexSelector.bind(
      null,
      modelSelector,
      datasetSelector,
      dataIndexSelector,
      previewContainer,
  )

  const updatePreviewBound = updateDataText.bind(
      null,
      modelSelector,
      datasetSelector,
      dataIndexSelector,
      previewContainer
  );

  modelSelector.onchange = updateDataSelectorBound;
  datasetSelector.onchange = updateDataIndexSelectorBound;
  dataIndexSelector.onchange = updatePreviewBound;

  // Selector
  document.body.appendChild(modelSelector);
  document.body.appendChild(datasetSelector);
  document.body.appendChild(dataIndexSelector);

  // Body
  document.body.appendChild(previewContainer);

  // Append schema graph
  // document.body.appendChild(sth);
}

async function updateDatasetSelector(modelSelector, datasetSelector, dataIndexSelector, container){
  const requestedModel = modelSelector.value;
  const datasets = await Dataset.getDatasets(requestedModel);
  if(modelSelector.value !== requestedModel){
    return;
  }
  // Remove items if any
  removeOptions(datasetSelector);

  // Add new datasets
  for (const dataset of datasets) {
    datasetSelector.options.add(new Option(dataset, dataset));
  }
  await updateDataIndexSelector(modelSelector, datasetSelector, dataIndexSelector, container);
}

async function updateDataIndexSelector(modelSelector, datasetSelector, dataIndexSelector, container){
  const requestedModel = modelSelector.value;
  const requestedDataset = datasetSelector.value;
  const dataIndices = await Data.getDataIndices(requestedModel, requestedDataset);
  if (modelSelector.value !== requestedModel || datasetSelector.value !== requestedDataset){
    return;
  }
  // Remove items if any
  removeOptions(dataIndexSelector);
  // Add new data
  for (const dataIndex of dataIndices){
    dataIndexSelector.options.add(new Option(dataIndex, dataIndex));
  }
  await updateDataText(modelSelector, datasetSelector, dataIndexSelector, container);
}

async function updateDataText(modelSelector, datasetSelector, dataIndexSelector, container){
  container.textContent = "Loading data...";
  const requestedModel = modelSelector.value;
  const requestedDataset = datasetSelector.value;
  const requestedDataIndex = dataIndexSelector.value;
  const data = await Data.getDataInfo(requestedModel, requestedDataset, requestedDataIndex);
  const preview = Views.createPreviews(data)
  if(modelSelector.value !== requestedModel || datasetSelector.value !== requestedDataset || dataIndexSelector.value !== requestedDataIndex){
    return;
  }
  container.textContent = '';
  container.appendChild(preview);

  // Get schema image
  const requestedDB = data.db;
  const params = new URLSearchParams({requestedModel, requestedDataset, requestedDB});
  const path = `./image?${params}`;
  var img = document.createElement("img");
  const result = await fetch(path);
  let tmp = (await result.json()) || {};
  img.src = 'data:image/png;base64,'.concat(tmp["image"]);
  img.style.width= "1000px";
  img.style.height = "1000px";
  container.appendChild(img);

  // Get encoded image
  const tensors = await Data.getTensorInfo(requestedModel, requestedDataset, requestedDataIndex);
  for (let[tag, value_list] of Object.entries(tensors)){
    value_list.forEach(function(item, index){
        var img2 = document.createElement("img");
        img2.src = 'data:image/png;base64,'.concat(item);
        img2.style.width = "850px";
        img2.style.height = "850px";
        container.appendChild(img2);
    });
  }

  // Get inference scores
  const scores = await Data.getInferenceInfo(requestedModel, requestedDataset, requestedDataIndex);
  for (let[tag, value] of Object.entries(scores)){
    var img3 = document.createElement("img");
    img3.src = 'data:image/png;base64,'.concat(value);
    img3.style.width = "850px";
    img3.style.height = "850px";
    container.appendChild(img3);
  }
}

function removeOptions(selectElement) {
   var i, L = selectElement.options.length - 1;
   for(i = L; i >= 0; i--) {
      selectElement.remove(i);
   }
}
