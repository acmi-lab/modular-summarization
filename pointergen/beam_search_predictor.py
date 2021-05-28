from allennlp.predictors.predictor import  Predictor
from allennlp.models.model import Model
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data import Instance
from overrides import overrides
from allennlp.common.util import JsonDict, sanitize


class TextGen(Predictor):
    def __init__(self, model: Model, dataset_reader: DatasetReader, decode_strategy="greedy") -> None:
        super().__init__(model, dataset_reader)
        self.decode_strategy = decode_strategy

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"article_lines": ["...", "...", ...], "summary_lines": ["...", "...", ...]}``.
        """
        return self._dataset_reader.dict_to_instance(json_dict)

    @overrides
    def predict_instance(self, instance: Instance) -> JsonDict:
        outputs = self._model.forward_on_instance(instance, decode_strategy=self.decode_strategy)
        return sanitize(outputs)


    @overrides
    def predict_json(self, inputs: JsonDict) -> JsonDict:
        instance = self._json_to_instance(inputs)
        predicted = self.predict_instance(instance)
        ground_truth = " ".join(instance.fields["meta"]["target_tokens"])
        
        to_return = {"input": inputs["article_lines"],
                "ground_truth": ground_truth,
                "prediction": predicted}
    
        for extra_key in ["case_id", "index_in_note", "section", "orig_section"]:
            if extra_key in inputs.keys():
                to_return[extra_key] = inputs[extra_key]

        return to_return



@Predictor.register("beamsearch_constrained")
class Beamsearch(TextGen):
    def __init__(self, model: Model, dataset_reader: DatasetReader):
        super(Beamsearch, self).__init__(model, dataset_reader, decode_strategy="beamsearch_constrained")

@Predictor.register("beamsearch")
class BeamsearchConstrained(TextGen):
    def __init__(self, model: Model, dataset_reader: DatasetReader):
        super(BeamsearchConstrained, self).__init__(model, dataset_reader, decode_strategy="beamsearch_unconstrained")


