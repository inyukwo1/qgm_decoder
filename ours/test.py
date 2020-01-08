from universal_utils import download_file_from_google_drive
from ours.src import args as arg, utils
from ours.src.models.model import IRNet
from ours.src.rule import semQL
import torch
import os.path


SPIDER_MODEL_PATH="test_models/ours_spider.model"


def prepare_model_spider():
    if os.path.isfile(SPIDER_MODEL_PATH):
        return
    download_file_from_google_drive("1YPVNCzvn2CiQAWYYXV8dbYuocS7QuJY0", SPIDER_MODEL_PATH)


def test_evaluation_spider():
    arg_parser = arg.init_arg_parser()
    args = arg.init_config(arg_parser)
    grammar = semQL.Grammar()
    _, _, val_sql_data, \
    val_table_data = utils.load_dataset(args.dataset, use_small=True, use_eval_only=True)

    model = IRNet(args, grammar)

    if args.cuda: model.cuda()

    pretrained_model = torch.load(SPIDER_MODEL_PATH,
                                  map_location=lambda storage, loc: storage)
    import copy
    pretrained_modeled = copy.deepcopy(pretrained_model)
    for k in pretrained_model.keys():
        if k not in model.state_dict().keys():
            del pretrained_modeled[k]

    model.load_state_dict(pretrained_modeled)

    json_datas = utils.epoch_acc(model, args.batch_size, val_sql_data, val_table_data,
                                 beam_size=args.beam_size)
    best_acc, sketch_acc = utils.eval_acc(json_datas, val_sql_data, log=False)
    assert best_acc == 0.6375 and sketch_acc == 0.925


if __name__ == "__main__":
    prepare_model_spider()
    test_evaluation_spider()
