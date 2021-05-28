
import torch.nn as nn

from allennlp.models.model import Model
from typing import Dict
from overrides import overrides
from allennlp.data.dataset import Batch
from allennlp.nn import util
from allennlp.common.util import START_SYMBOL, END_SYMBOL

from allennlp.data.instance import Instance

import torch
from torch.nn import LSTM
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.modules import TimeDistributed

from sequential_sentence_tagger.all_classification_metrics import AllClassificationMetric

EPS=1e-8

        
@Model.register("sequential_sentence_tagger")
class Seq2Seq(Model):
    def __init__(self, vocab, num_labels, hidden_size=256, emb_size=128, dataset:str="abridge"):
        super().__init__(vocab)
        # self.vocab=vocab

        ## vocab related setup begins
        self.vocab_size=vocab.get_vocab_size()
        self.PAD_ID = vocab.get_token_index(vocab._padding_token)
        self.OOV_ID = vocab.get_token_index(vocab._oov_token)
        self.START_ID = vocab.get_token_index(START_SYMBOL)
        self.END_ID = vocab.get_token_index(END_SYMBOL)
        ## vocab related setup ends

        self.emb_size=emb_size
        self.hidden_size=hidden_size
        self.num_labels=num_labels

        self.emb_layer = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.emb_size)        
        lstm_layer = LSTM(input_size=self.emb_size, hidden_size=self.hidden_size, bidirectional=True, batch_first=True)
        self.sentence_lstm_encoder = TimeDistributed(PytorchSeq2SeqWrapper(lstm_layer))
        
        lstm_layer2 = LSTM(input_size=2*self.hidden_size, hidden_size=self.hidden_size, bidirectional=True, batch_first=True)
        self.contextual_encoder = PytorchSeq2SeqWrapper(lstm_layer2)
        self.projection_layer = nn.Linear(2*self.hidden_size, self.num_labels)
        
        self.loss =nn.BCEWithLogitsLoss(reduction='none')

        self._metric = AllClassificationMetric(num_classes=num_labels, dataset=dataset)
        
        
        # buffers because these dont need grads. These are placed here because they will be replicated across gpus
        self.register_buffer("true_rep", torch.tensor(1.0))
        self.register_buffer("false_rep", torch.tensor(0.0))
    

    def forward(self, lines, labels=None, meta=None, only_predict_probs=False):
#         print(self.PAD_ID)
        input_tokens = lines["tokens"]
        input_pad_mask= input_tokens!=self.PAD_ID
#         print(input_pad_mask)
        embedded_seq = self.emb_layer(input_tokens)
        sentenced_encoded = self.sentence_lstm_encoder(embedded_seq, input_pad_mask)  # batchxnumsentxsentlenxhidden_size
        
        embedding_summed = torch.sum(sentenced_encoded, axis=-2, keepdim=False)
        to_divide = torch.sum(input_pad_mask, axis=-1, keepdim=True)+EPS
        meanpooled_sent_reps = embedding_summed/to_divide       # batchxnumsentx2*hidden_size
        
        sentence_level_pad_mask = input_pad_mask[:,:,0]
#         print(sentence_level_pad_mask)
        contextual_embedded = self.contextual_encoder(meanpooled_sent_reps, sentence_level_pad_mask)
    
        logits = self.projection_layer(contextual_embedded)
        
        loss_without_mask = self.loss(logits, labels)
        loss_with_mask = loss_without_mask*sentence_level_pad_mask.unsqueeze(-1)
        total_loss = torch.sum(loss_with_mask)
        numpreds = torch.sum(sentence_level_pad_mask)
        avg_loss = total_loss/numpreds


        if only_predict_probs:
            probs = torch.nn.functional.sigmoid(logits)
            return probs.detach().cpu().numpy()

        if labels is not None:
            probs = torch.nn.functional.sigmoid(logits)
            self._metric(gold_labels=labels.reshape(-1,self.num_labels), predictions=probs.reshape(-1,self.num_labels), mask=sentence_level_pad_mask.reshape(-1))

        return {
            "loss": avg_loss
            }

            
    @overrides
    def forward_on_instance(self, instance: Instance) -> Dict[str, str]:
        """
        Takes an :class:`~allennlp.data.instance.Instance`, which typically has raw text in it,
        converts that text into arrays using this model's :class:`Vocabulary`, passes those arrays
        through :func:`self.forward()` and :func:`self.decode()` (which by default does nothing)
        and returns the result.  Before returning the result, we convert any
        ``torch.Tensors`` into numpy arrays and remove the batch dimension.
        """
        cuda_device = self._get_prediction_device()
        dataset = Batch([instance])
        dataset.index_instances(self.vocab)
        model_input = util.move_to_device(dataset.as_tensor_dict(), cuda_device)
        return self.forward(**model_input, only_predict_probs=True)
        

    def get_metrics(self, reset: bool = False) -> Dict[str, any]:
        metrics = self._metric.get_metric(reset)
        return metrics

