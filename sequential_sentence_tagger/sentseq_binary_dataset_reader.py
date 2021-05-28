import logging
from typing import List, Dict

import numpy as np
from overrides import overrides
import pickle

import jsonlines

from allennlp.common.checks import ConfigurationError
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, ArrayField, MetadataField, NamespaceSwappingField, ListField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter


from overrides import overrides


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("sentseq_binary_dataset_reader")
class SentSeqBinaryDatasetReader(DatasetReader):
    def __init__(self,
                 max_sent_length : int = np.inf,
                 max_sents_per_example : int = np.inf,                 
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lowercase_tokens : bool = False,
                 lazy: bool = False,
                 max_to_read = np.inf) -> None:
        super().__init__(lazy)
        self.lowercase_tokens = lowercase_tokens
        self.max_sent_length = max_sent_length
        self.max_sents_per_example = max_sents_per_example
        self.max_to_read = max_to_read
        self._tokenizer = tokenizer or WordTokenizer(word_splitter=JustSpacesWordSplitter())
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
        input_lines = dp["article_lines"]
        labels = dp["labels"]

        if len(input_lines)!=len(labels):
            print("error in", dp["case_id"])
        
        
        input_lines = input_lines[:self.max_sents_per_example]
        shortened_input_lines = []
        for line in input_lines:
            shortened_input_lines.append(  " ".join(line.split(" ")[:self.max_sent_length])  )
        labels = labels[:self.max_sents_per_example]
        return self.text_to_instance(shortened_input_lines, labels)

    @staticmethod
    def _tokens_to_ids(tokens: List[Token]) -> List[int]:
        ids: Dict[str, int] = {}
        out: List[int] = []
        for token in tokens:
            out.append(ids.setdefault(token.text.lower(), len(ids)))
        return out

    @overrides
    def text_to_instance(self, shortened_input_lines: List[str], labels: List[List[int]] = None) -> Instance:  # type: ignore
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
            for i, s in enumerate(shortened_input_lines):
                shortened_input_lines[i]=s.lower()

        tokenized_lines = [[Token(START_SYMBOL)]+self._tokenizer.tokenize(s)+[Token(END_SYMBOL)] for s in shortened_input_lines]
        indexed_line_fields = [TextField(tokenized_line, self._token_indexers) for tokenized_line in tokenized_lines]
        
        seq_of_textfields = ListField(indexed_line_fields)

        meta_fields = {"tokenized_lines": tokenized_lines}
        fields_dict = {
                "lines": seq_of_textfields,
        }

        if labels is not None:
            labelfield = ArrayField(np.max(np.array(labels), axis=1, keepdims=True))
            fields_dict["labels"] = labelfield

        fields_dict["meta"] = MetadataField(meta_fields)

        return Instance(fields_dict)


