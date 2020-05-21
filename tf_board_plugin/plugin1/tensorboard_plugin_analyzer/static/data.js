async function fetchJSON(url) {
  const response = await fetch(url);
  if (!response.ok) {
    return null;
  }
  return response.json();
}

// DataIndices
let dataIndicesToTagInfo = null;

async function updateDataIndicesInfo(model, dataset){
    const params = new URLSearchParams({model, dataset})
    dataIndicesToTagInfo = (await fetchJSON(`./data_indices?${params}`)) || {};
}

export async function getDataIndices(model, dataset){
    await updateDataIndicesInfo(model, dataset);
    return Object.keys(dataIndicesToTagInfo);
}

// Data
let dataInfo = null;

async function updateDataInfo(model, dataset, index){
    const params = new URLSearchParams({model, dataset, index});
    dataInfo = (await fetchJSON(`./data?${params}`)) || {};
}

export async function getDataInfo(model, dataset, index){
    await updateDataInfo(model, dataset, index);
    return dataInfo;
}

// Tensors
let tensorInfo = null;

async function updateTensorInfo(model, dataset, index){
    const params = new URLSearchParams({model, dataset, index});
    tensorInfo = (await fetchJSON(`./tensor?${params}`)) || {};
}

export async function getTensorInfo(model, dataset, index){
    await updateTensorInfo(model, dataset, index);
    return tensorInfo;
}

// Inference
let inferenceInfo = null;

async function updateInferenceInfo(model, dataset, index){
    const params = new URLSearchParams({model, dataset, index});
    inferenceInfo = (await fetchJSON(`./inference?${params}`)) || {};
}

export async function getInferenceInfo(model, dataset, index){
    await updateInferenceInfo(model, dataset, index);
    return inferenceInfo;
}
