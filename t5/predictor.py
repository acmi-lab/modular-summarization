import pdb

from allennlp.predictors.predictor import  Predictor
from allennlp.models.model import Model
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data import Instance
from overrides import overrides
from allennlp.common.util import JsonDict, sanitize
from allennlp.common.util import START_SYMBOL, END_SYMBOL

from transformers import T5Tokenizer, T5Config


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

    def _join_tokens(self, token_list):
        return " ".join(token_list)

    @overrides
    def predict_json(self, inputs: JsonDict) -> JsonDict:
        instance = self._json_to_instance(inputs)
        predicted = self.predict_instance(instance)
        predicted_str = self._join_tokens(predicted["tokens"])

        ground_truth = self._join_tokens(instance.fields["meta"]["target_tokens"])

        to_return = {"input": inputs["article_lines"],
                "ground_truth": ground_truth,
                "prediction": predicted_str}

        for extra_key in inputs.keys():
            if extra_key not in to_return:
                to_return[extra_key] = inputs[extra_key]

        return to_return



class TextGenWordpiece(TextGen):
    def __init__(self, model: Model, dataset_reader: DatasetReader, decode_strategy="greedy") -> None:
        super(TextGenWordpiece, self).__init__(model, dataset_reader, decode_strategy)
        self._tokenizer = T5Tokenizer.from_pretrained("t5-base")
        self.new_unk_token = self._tokenizer.unk_token

    def _replace_toks_inplace(self, toklist, from_tok, to_tok):
        for _i, tok in enumerate(toklist):
            if tok==from_tok:
                toklist[_i]=to_tok

    @overrides
    def _join_tokens(self, token_list):
        self._replace_toks_inplace(token_list, from_tok="@@UNKNOWN@@", to_tok=self.new_unk_token)
        assert START_SYMBOL not in token_list
        assert END_SYMBOL not in token_list
        assert "@@PADDING@@" not in token_list
        return self._tokenizer.convert_tokens_to_string(token_list)


@Predictor.register("greedy")
class SimpleGreedyWordpiece(TextGenWordpiece):
    def __init__(self, model: Model, dataset_reader: DatasetReader):
        super(SimpleGreedyWordpiece, self).__init__(model, dataset_reader, decode_strategy="greedy")

@Predictor.register("beamsearch")
class BeamsearchWordpiece(TextGenWordpiece):
    def __init__(self, model: Model, dataset_reader: DatasetReader):
        super(BeamsearchWordpiece, self).__init__(model, dataset_reader, decode_strategy="beamsearch")

