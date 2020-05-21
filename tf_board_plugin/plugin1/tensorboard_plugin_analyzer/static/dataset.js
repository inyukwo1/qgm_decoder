let datasetToTagInfo = null;

async function fetchJSON(url) {
  const response = await fetch(url);
  if (!response.ok) {
    return null;
  }
  return response.json();
}

async function updateDatasetInfo(model){
    const params = new URLSearchParams({model});
    datasetToTagInfo = (await fetchJSON(`./datasets?${params}`)) || {};
}

export async function getDatasets(model){
    await updateDatasetInfo(model);
    return Object.keys(datasetToTagInfo);
}

let datasetPath = null;

async function updateDatasetPath(model, dataset){
    const params = new URLSearchParams({model, dataset});
    datasetPath = (await fetchJSON(`./dataset_path?${params}`)) || {};
}

export async function getDatasetPath(model, dataset){
    await updateDatasetPath(model, dataset);
    return datasetPath['path'];
}