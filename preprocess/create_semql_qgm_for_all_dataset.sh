#!/bin/bash

semql_path="./preprocess/"
qgm_path="./rule/qgm/"
#dataset_names=("wikisql" "wikitablequestions")
#dataset_names=("spider" "wikisql" "wikitablequestions")
dataset_names=("spider")
data_types=("train" "dev")
#data_types=("train" "dev" "test")


echo "Start download NLTK data"
python download_nltk.py
cd ../

for dataset_name in ${dataset_names[@]}
do
    # Skip if dataset doesn't exist
    if [ ! -d "./data/${dataset_name}" ]; then
        echo -e "\n\nDirectory not found. Skipping ${dataset_name}."
        continue
    fi

    echo -e "\n\nDataset: ${dataset_name}"
    for data_type in ${data_types[@]}
    do
        if [ "${dataset_name}" == "spider" ] && [ "${data_type}" == "test" ]; then
            continue
        fi

        echo -e "\nData_type: ${data_type}"
        data="../data/${dataset_name}/${data_type}_original.json"
        table_data="../data/${dataset_name}/tables.json"
        output="../data/${dataset_name}/${data_type}.json"

        cd ${semql_path}
        # Create SemQL
        echo "Start process the origin Spider dataset"
        python data_process.py --data_path ${data} --table_path ${table_data} --output ${output}
        echo "Start generate SemQL from SQL"
        python sql2SemQL.py --data_path ${output} --table_path ${table_data} --output ${output}
        cd ../

#        data="../../data/${dataset_name}/${data_type}_original.json"
#        table_data="../../data/${dataset_name}/tables.json"
#        output="../../data/${dataset_name}/${data_type}.json"
#
#        cd ${qgm_path}
#        # Create QGM
#        echo "Start generate QGM from SQL"
#        python sql2qgm.py --db ${table_data} --source ${output} --destin ${output}
#        # Pretty print
#        python -m json.tool ${output} > a && mv a ${output}
#        cd ../../
    done
done
