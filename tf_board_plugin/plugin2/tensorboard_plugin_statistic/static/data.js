async function fetchJSON(url) {
  const response = await fetch(url);
  if (!response.ok) {
    return null;
  }
  return response.json();
}


let dataInfo = null;

async function updateDataInfo(run){
    const params = new URLSearchParams({run})
    dataInfo = (await fetchJSON(`./data?${params}`)) || {};
}

export async function getDataInfo(run){
    await updateDataInfo(run);
    return dataInfo;
}

