import logging
import pdb
from typing import List, Dict

import numpy as np
from overrides import overrides
import pickle

import jsonlines

from allennlp.common.checks import ConfigurationError
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, ArrayField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

from overrides import overrides

from transformers import T5Tokenizer


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name



@DatasetReader.register("pretrained_cnndmail_dataset_reader")
class PretrainedCNNDmailDatasetReader(DatasetReader):
    def __init__(self,
                 max_source_length : int = 400,
                 max_target_length : int =100,
                 tokenizer: Tokenizer = None,
                 pretrained_model_name: str= 't5-base',
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lowercase_tokens : bool = False,
                 lazy: bool = False,
                 max_to_read = np.inf) -> None:
        super().__init__(lazy)
        self.lowercase_tokens = lowercase_tokens
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.max_to_read = max_to_read

        # REMEMBER : Your data file must not contain things like PAD, UNK, START, STOP explicitly
        self._tokenizer = T5Tokenizer.from_pretrained(pretrained_model_name)

        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        if "tokens" not in self._token_indexers or \
                not isinstance(self._token_indexers["tokens"], SingleIdTokenIndexer):
            raise ConfigurationError("CNNDmailDatasetReader expects 'token_indexers' to contain "
                                     "a 'single_id' token indexer called 'tokens'.")


    @overrides
    def _read(self, file_path):
        logger.info("Reading instances from lines in file at: %s", file_path)
        with jsonlines.open(file_path, "r") as reader:
            num_passed = 0
            for dp in reader:
                if num_passed == self.max_to_read:
                    return
                if len(" ".join(dp["article_lines"]))==0:    # if the input article has length 0 then there is a crash due to some NaN popping up. there are 114 such datapoins in cnndmail trainset
                    continue
                num_passed += 1
                yield self.dict_to_instance(dp)

    def dict_to_instance(self, dp):
        source_sequence = " ".join(dp["article_lines"])
        target_sequence = " ".join(dp["summary_lines"])
        source_words_truncated = source_sequence.split(" ")[:self.max_source_length]
        target_words_truncated = target_sequence.split(" ")[:self.max_target_length]
        source_sequence = " ".join(source_words_truncated)
        target_sequence = " ".join(target_words_truncated)
        return self.text_to_instance(source_sequence, target_sequence)


    @overrides
    def text_to_instance(self, source_string: str, target_string: str = None) -> Instance:  # type: ignore
        """
        Turn raw source string and target string into an ``Instance``.

        Parameters
        ----------
        source_string : ``str``, required
        target_string : ``str``, optional (default = None)

        Returns
        -------
        Instance
            See the above for a description of the fields that the instance will contain.
        """
        # pylint: disable=arguments-differ
        if self.lowercase_tokens:
            source_string = source_string.lower()
            target_string = target_string.lower()
        tokenized_source = self._tokenizer.tokenize(source_string)
        tokenized_source = [Token(w) for w in tokenized_source]
        source_field = TextField(tokenized_source, self._token_indexers)

        meta_fields = {"source_tokens": [x.text for x in tokenized_source]}
        fields_dict = {
                "source_tokens": source_field,
        }

        if target_string is not None:
            tokenized_target = self._tokenizer.tokenize(target_string)
            tokenized_target = [Token(w) for w in tokenized_target]
            meta_fields["target_tokens"] = [x.text for x in tokenized_target]
            tokenized_target.insert(0, Token(START_SYMBOL))
            tokenized_target.append(Token(END_SYMBOL))
            target_field = TextField(tokenized_target, self._token_indexers)
            fields_dict["target_tokens"] = target_field

        fields_dict["meta"] = MetadataField(meta_fields)

        return Instance(fields_dict)



@DatasetReader.register("conditioned_pretrained_cnndmail_dataset_reader")
class ConditionedPretrainedCNNDmailDatasetReader(PretrainedCNNDmailDatasetReader):
    @overrides
    def dict_to_instance(self, dp):
        source_sequence = " ".join(dp["article_lines"])
        target_sequence = " ".join(dp["summary_lines"])
        source_words_truncated = source_sequence.split(" ")[:self.max_source_length]
        target_words_truncated = target_sequence.split(" ")[:self.max_target_length]

        source_sequence = " ".join(source_words_truncated)
        section_name = dp["section"].replace("_"," ").strip().lower()
        source_sequence = section_name+" "+source_sequence

        target_sequence = " ".join(target_words_truncated)
        return self.text_to_instance(source_sequence, target_sequence)



