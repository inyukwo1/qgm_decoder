import json
dataset = "wikisql"


def read_write(inpath, outpath):
    with open(inpath, "r") as f:
        json_obj = json.load(f)

    sentences = [obj["question"] for obj in json_obj]
    with open(outpath, "w") as f:
        for sentence in sentences:
            print(sentence, file=f)


for data in ["spider", "spider_resplit_1", "spider_resplit_2", "spider_resplit_3"]:
    read_write("/mnt/disk2/qgm_decoder_datasets/{}/train_unsimplifiable.json".format(data),
               "/mnt/disk2/qgm_decoder_datasets/{}/train_unsimplifiable_sentences.txt".format(data))
    read_write("/mnt/disk2/qgm_decoder_datasets/{}/dev_unsimplifiable.json".format(data),
               "/mnt/disk2/qgm_decoder_datasets/{}/dev_unsimplifiable_sentences.txt".format(data))
    read_write("/mnt/disk2/qgm_decoder_datasets/{}/test_unsimplifiable.json".format(data),
               "/mnt/disk2/qgm_decoder_datasets/{}/test_unsimplifiable_sentences.txt".format(data))
    read_write("/mnt/disk2/qgm_decoder_datasets/{}/train_parsable.json".format(data),
               "/mnt/disk2/qgm_decoder_datasets/{}/train_parsable_sentences.txt".format(data))
    read_write("/mnt/disk2/qgm_decoder_datasets/{}/dev_parsable.json".format(data),
               "/mnt/disk2/qgm_decoder_datasets/{}/dev_parsable_sentences.txt".format(data))
    read_write("/mnt/disk2/qgm_decoder_datasets/{}/test_parsable.json".format(data),
               "/mnt/disk2/qgm_decoder_datasets/{}/test_parsable_sentences.txt".format(data))
