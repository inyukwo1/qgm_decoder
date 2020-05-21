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

// Generic view builders.


export function createPreviews(dataInfo){
  const fragment = document.createDocumentFragment();
  if (Object.keys(dataInfo).length == 0){
    const messageElement = createElement('h2');
    messageElement.textContent = 'No data selected.';
    fragment.appendChild(messageElement);
    return fragment;
  }
  // Create table
  const table = createElement('table', "table");

  for (let [tag, value] of Object.entries(dataInfo)){
    // create row
    const row = createElement('tr', 'tr_'.concat(tag));

    // tag
    const left_element = createElement('td', 'td_tag_'.concat(tag));
    left_element.textContent = tag.toUpperCase();
    row.appendChild(left_element);

    //create right value
    if (tag == 'columns' || tag == 'tables'){
      //create sub table
      const entities = ''.concat(value).split(',');
      const sub_table = createElement('table', "sub_table_".concat(tag));
      entities.forEach(function(item, index){
        const sub_row = createElement('tr', 'tr_'.concat(index));
        const sub_left_element = createElement('td', 'td_left');
        const sub_right_element = createElement('td', 'td_right');
        // Set value
        sub_left_element.textContent = index;
        sub_right_element.textContent = item;
        // Set size
        sub_left_element.width = '30';

        sub_row.appendChild(sub_left_element);
        sub_row.appendChild(sub_right_element);
        sub_table.appendChild(sub_row);
      });
      row.appendChild(sub_table);
    }
    else{
      const right_element = createElement('textarea', 'td_'.concat(tag));
      if (tag == 'query'){
        right_element.textContent = value.join(' ');
      }
      else if (tag == 'gold' || tag == 'pred'){
        const str = [];
        for (const item of value){
          str.push(item.join('(').concat(')'));
        }
        right_element.textContent = str.join(', ');
      }
      else{
        right_element.textContent = value;
      }
      right_element.rows = '2';
      right_element.cols = '127';
      row.appendChild(right_element);
    }

    table.appendChild(row);
  }
  fragment.appendChild(table);
  return fragment;
}


/**
 * @param {string} tag
 * @param {string=} className
 * @return {!Element}
 */
function createElement(tag, className) {
  const result = document.createElement(tag);
  if (className) {
    result.className = className;
  }
  return result;
}


// Appended codes from now on
export function createModelSelector(models){
    /**
   * Build a component in this form:
   *   <select class="model-selector">
   *     <option value="${run}">${run}</option>
   *     ...
   *   </select>
   */
  const element = createElement('select', 'model-selector');
  for (const model of models){
    element.options.add(new Option(model, model));
  }
  return element;
}

export function createDatasetSelector(datasets){
  const element = createElement('select', 'dataset-selector');
  for (const dataset of datasets){
    element.options.add(new Option(dataset, dataset));
  }
  return element;
}

export function createDataSelector(data){
  const element = createElement('select', 'data-selector');
  for (const item of data){
    element.options.add(new Option(item, item));
  }
  return element;
}
