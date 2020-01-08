from universal_utils import download_file_from_google_drive
from irnet.src import args as arg, utils
from irnet.src.models.model import IRNet
from irnet.src.rule import semQL
from irnet.end2end import End2EndIRNet
import torch
import os.path


SPIDER_MODEL_PATH="test_models/irnet_spider.model"


def prepare_model_spider():
    if os.path.isfile(SPIDER_MODEL_PATH):
        return
    download_file_from_google_drive("14KvKDymTVwr14bTADXtfj33GNKK4AgWc", SPIDER_MODEL_PATH)


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

    model.word_emb = utils.load_word_emb(args.glove_embed_path)
    model.load_state_dict(pretrained_modeled)

    json_datas = utils.epoch_acc(model, args.batch_size, val_sql_data, val_table_data,
                                 beam_size=args.beam_size)
    acc = utils.eval_acc(json_datas, val_sql_data)
    assert acc == 0.425


def test_end2end_spider():
    end2end_irnet = End2EndIRNet()
    end2end_irnet.prepare_model("spider")
    q, _, _ = end2end_irnet.run_model("concert_singer", "How many singers do we have?")
    assert q == """SELECT count(*) FROM singer AS T1        """


if __name__ == "__main__":
    prepare_model_spider()
    test_evaluation_spider()
    test_end2end_spider()
