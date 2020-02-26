from allennlp import __version__
from allennlp.models.archival import load_archive
from allennlp.predictors.predictor import Predictor, JsonDict
from allennlp.common.util import import_submodules


class End2EndGNN:
    def prepare_model(self, dataset):
        import_submodules("gnn.models.semantic_parsing.spider_parser")
        import_submodules("gnn.dataset_readers.spider")
        import_submodules("gnn.predictors.spider_predictor")
        dataset_path = "test_models/gnn_{}.model".format(dataset)
        archive = load_archive(dataset_path, weights_file=dataset_path + "/best.th")
        self._predictor = Predictor.from_archive(archive, "spider")

    def run_model(self, db_id, nl_string):
        dataset_reader = self._predictor._dataset_reader
        ins = dataset_reader.text_to_instance(
            query_index=0, utterance=nl_string, db_id=db_id
        )
        result = self._predictor.predict_instance(ins)
        return result["predicted_sql_query"], None, nl_string


if __name__ == "__main__":
    model = End2EndGNN()
    model.prepare_model("spider")
    q, _, _ = model.run_model("imdb", "How many singers do we have?")
    print(q)
