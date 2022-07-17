# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch RoBERTa model. """


import logging

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss

from transformers import RobertaConfig
from transformers.file_utils import add_start_docstrings, add_start_docstrings_to_callable
from .modeling_bert import BertEmbeddings, BertLayerNorm, BertModel, BertPreTrainedModel, gelu
from transformers.modeling_utils import create_position_ids_from_input_ids

# Modify code here
from knowledge_injection_layer.coref_triple_enc import Coref_triple_enc
from knowledge_injection_layer.config import Config
from knowledge_injection_layer.modules import Combineloss
from knowledge_injection_layer.kg_injection import Kg_Injection
# from transformers.modeling_bert import
import numpy as np

from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

logger = logging.getLogger(__name__)

ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP = {
    "roberta-base": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-pytorch_model.bin",
    "roberta-large": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-pytorch_model.bin",
    "roberta-large-mnli": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-mnli-pytorch_model.bin",
    "distilroberta-base": "https://s3.amazonaws.com/models.huggingface.co/bert/distilroberta-base-pytorch_model.bin",
    "roberta-base-openai-detector": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-openai-detector-pytorch_model.bin",
    "roberta-large-openai-detector": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-openai-detector-pytorch_model.bin",
}


class RobertaEmbeddings(BertEmbeddings):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """

    def __init__(self, config, with_naive_feature):
        super().__init__(config, with_naive_feature)
        self.padding_idx = 1
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=self.padding_idx)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, ner_ids=None, ent_ids=None):
        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx).to(input_ids.device)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        return super().forward(
            input_ids, token_type_ids=token_type_ids, position_ids=position_ids, inputs_embeds=inputs_embeds, ner_ids=ner_ids, ent_ids=ent_ids
        )

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """ We are provided embeddings directly. We cannot infer which are padded so just generate
        sequential position ids.

        :param torch.Tensor inputs_embeds:
        :return torch.Tensor:
        """
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        return position_ids.unsqueeze(0).expand(input_shape)


ROBERTA_START_DOCSTRING = r"""

    This model is a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`_ sub-class.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general
    usage and behavior.

    Parameters:
        config (:class:`~transformers.RobertaConfig`): Model configuration class with all the parameters of the
            model. Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""

ROBERTA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`transformers.RobertaTokenizer`.
            See :func:`transformers.PreTrainedTokenizer.encode` and
            :func:`transformers.PreTrainedTokenizer.encode_plus` for details.

            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.

            `What are attention masks? <../glossary.html#attention-mask>`__
        token_type_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Segment token indices to indicate first and second portions of the inputs.
            Indices are selected in ``[0, 1]``: ``0`` corresponds to a `sentence A` token, ``1``
            corresponds to a `sentence B` token

            `What are token type IDs? <../glossary.html#token-type-ids>`_
        position_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Indices of positions of each input sequence tokens in the position embeddings.
            Selected in the range ``[0, config.max_position_embeddings - 1]``.

            `What are position IDs? <../glossary.html#position-ids>`_
        head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`, defaults to :obj:`None`):
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:
            :obj:`1` indicates the head is **not masked**, :obj:`0` indicates the head is **masked**.
        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`, defaults to :obj:`None`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert `input_ids` indices into associated vectors
            than the model's internal embedding lookup matrix.
"""


@add_start_docstrings(
    "The bare RoBERTa Model transformer outputting raw hidden-states without any specific head on top.",
    ROBERTA_START_DOCSTRING,
)
class RobertaModel(BertModel):
    """
    This class overrides :class:`~transformers.BertModel`. Please check the
    superclass for the appropriate documentation alongside usage examples.
    """

    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config, with_naive_feature, entity_structure):
        super().__init__(config, with_naive_feature, entity_structure)

        self.embeddings = RobertaEmbeddings(config, with_naive_feature)
        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value


@add_start_docstrings(
    """Roberta Model for DocRED, this is a multi entity-pair cls setting""",
    ROBERTA_START_DOCSTRING,
)
class RobertaForDocRED(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config, num_labels, max_ent_cnt, with_naive_feature=False, entity_structure=False):
        super().__init__(config)
        self.config = config
        self.num_labels = num_labels
        self.max_ent_cnt = max_ent_cnt
        self.with_naive_feature = with_naive_feature
        self.reduced_dim = 128
        self.roberta = RobertaModel(config, with_naive_feature, entity_structure)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dim_reduction = nn.Linear(config.hidden_size, self.reduced_dim)
        self.feature_size = self.reduced_dim
        if self.with_naive_feature:
            self.feature_size += 20
            self.distance_emb = nn.Embedding(20, 20, padding_idx=10)
        self.bili = torch.nn.Bilinear(self.feature_size, self.feature_size, self.num_labels)
        self.hidden_size = config.hidden_size

        self.init_weights()

        # Modify code here, add Knowledge_injection_layer
        if Config.add_kg_flag:
            self.kg_injection = Kg_Injection(Config.ent_hidden_dim, Config.rel_hidden_dim,
                                             Config.gcn_layer_nums, Config.gcn_head_nums,
                                             Config.gcn_in_drop, Config.kg_num_hidden_layers,
                                             Config.hidden_dim,
                                             Config.kg_intermediate_size,
                                             Config.kg_num_attention_heads,
                                             Config.kg_attention_probs_dropout_prob)
        else:
            self.kg_injection = None

        if Config.add_coref_flag:
            self.coref_injection = Coref_triple_enc(config.hidden_size)
        else:
            self.coref_injection = None

        self.combineloss = Combineloss(Config)

    def encode( self, input_ids, attention_mask, subword_indexs,
                kg_ent_attrs, kg_ent_attr_nums, kg_ent_attr_lens, kg_adj, kg_adj_edges, kg_ent_labels, kg_ent_mask):

        config = self.config
        start_tokens = [config.bos_token_id]
        end_tokens = [config.eos_token_id]
        '''
        print("shape of config:")
        print("input_ids: {0}".format(input_ids.size()))
        print("attention_mask: {0}".format(attention_mask.size()))
        print("kg_ent_attr_nums: {0}".format(np.array(kg_ent_attr_nums).shape))
        print("kg_ent_attrs: {0}".format(kg_ent_attrs.size()))
        print("kg_ent_attr_lens: {0}".format(kg_ent_attr_lens.size()))

        print("kg_ent_labels: {0}".format(kg_ent_labels.size()))
        print("kg_ent_mask: {0}".format(kg_ent_mask.size()))
        '''


        

        new_input_ids, new_attention_mask, new_input_ids_kg, loss_kg, num_seg, seq_len, n, c,  len_start, len_end, bert_strats= self.process(input_ids, attention_mask, subword_indexs, start_tokens, end_tokens,
                                                                            self.kg_injection, kg_ent_attrs, kg_ent_attr_nums, kg_ent_attr_lens, kg_adj, kg_adj_edges, kg_ent_labels, kg_ent_mask)
        return new_input_ids, new_attention_mask, new_input_ids_kg, loss_kg, num_seg, seq_len, n, c,  len_start, len_end, bert_strats

    def process(self, input_ids, attention_mask, subword_indexs, start_tokens, end_tokens,
                kg_injection, kg_ent_attrs, kg_ent_attr_nums, kg_ent_attr_lens, kg_adj, kg_adj_edges, kg_ent_labels, kg_ent_mask):
        n, c = input_ids.size()
        if(c>512):
            print("wawawa")
            exit()
        bert_starts = torch.zeros((n,c)).to(input_ids)
        for i in range(n):
            token_start_idxs = [x + 1 for x in subword_indexs[i]] # np.cumsum([0] + subword_lengths[i][:-1])
            for x in token_start_idxs:
                if x<c:
                    bert_starts[i, x] = 1
        new_input_ids, new_attention_mask, new_input_ids_kg, loss_kg, num_seg, seq_len, c,  len_start, len_end = self.bert_forward(kg_injection, input_ids, attention_mask, start_tokens, end_tokens, bert_starts,
                                                                                           kg_ent_attrs, kg_ent_attr_nums, kg_ent_attr_lens, kg_adj, kg_adj_edges, kg_ent_labels, kg_ent_mask)
        return new_input_ids, new_attention_mask, new_input_ids_kg, loss_kg, num_seg, seq_len, n, c,  len_start, len_end, bert_starts
    
    def bert_forward(self, kg_injection, input_ids, attention_mask, start_tokens, end_tokens, bert_starts, 
                    kg_ent_attrs, kg_ent_attr_nums, kg_ent_attr_lens, kg_adj, kg_adj_edges, kg_ent_labels, kg_ent_mask):
        n, c = input_ids.size()
        start_tokens = torch.tensor(start_tokens).to(input_ids)
        end_tokens = torch.tensor(end_tokens).to(input_ids)
        len_start = start_tokens.size(0)
        len_end = end_tokens.size(0)


        embedding_output = self.roberta.embeddings.word_embeddings(input_ids)
        loss_kg = None
        if Config.add_kg_flag or Config.add_coref_flag:
            context_masks = [mask[starts.nonzero().squeeze(1)] for mask, starts in
                                zip(attention_mask, bert_starts)]
            for i in range(len(context_masks)):
                tmp0 = torch.tensor([0]*(512-len(context_masks[i])),device = torch.cuda.current_device())
                context_masks[i] = torch.cat((context_masks[i],tmp0),0)
                del tmp0
            context_masks = pad_sequence(context_masks, batch_first=True, padding_value=0)

            embedding_output_kg = embedding_output.detach().clone()

            if Config.add_kg_flag:
                word_vec_kg = [layer[starts.nonzero().squeeze(1)] for layer, starts in zip(embedding_output_kg, bert_starts)]
                for i in range(len(word_vec_kg)):
                    arr = np.zeros((512-len(word_vec_kg[i]), 1024))
                    tmp1 = torch.tensor((arr),device = torch.cuda.current_device())
                    word_vec_kg[i] = torch.cat((word_vec_kg[i], tmp1),0)
                    del arr,tmp1
                word_vec_kg = pad_sequence(word_vec_kg, batch_first=True, padding_value=0)
                
                #context_masks = context_masks.to(torch.float32)
                context_masks = context_masks.bool()
                #print(word_vec_kg.dtype)
                word_vec_kg = word_vec_kg.to(torch.float32)
                word_vec_kg, loss_kg = kg_injection(kg_ent_attrs, kg_ent_attr_nums,
                                                    kg_ent_attr_lens,
                                                    kg_adj, kg_ent_labels,
                                                    word_vec_kg, context_masks, kg_ent_mask,
                                                    kg_ent_labels, kg_adj_edges)

                batch_size = embedding_output_kg.size(0)
                embedding_output_kg_result = []
                for i in range(batch_size):
                    layer = embedding_output_kg[i]
                    starts = bert_starts[i]

                    starts = torch.cumsum(starts, dim=0) - 1
                    new_layer = torch.cat((layer[0].unsqueeze(0), word_vec_kg[i][starts[1:]]), dim=0)
                    embedding_output_kg_result.append(new_layer)

                embedding_output_kg = torch.stack(embedding_output_kg_result, dim=0)

        new_input_ids, new_attention_mask, num_seg = [], [], []
        if Config.add_kg_flag or Config.add_coref_flag:
            new_input_ids_kg = []
        seq_len = attention_mask.sum(1).cpu().numpy().astype(np.int32).tolist()
        for i, l_i in enumerate(seq_len):
            if l_i > 512:
                print("aaaaaa")
                exit()
            bert_indexs = [(bert_s, min(l_i - len_end, bert_s + 512-(len_start+len_end))) for bert_s in
                            range(len_start, l_i - len_end, 512-(len_start+len_end))]
            num_seg.append(len(bert_indexs))
            for j, (bert_s, bert_e) in enumerate(bert_indexs):
                if j == len(bert_indexs) - 1:
                    if l_i <= 512:
                        if Config.add_kg_flag or Config.add_coref_flag:
                            new_input_ids_kg.append(
                                        torch.cat([self.roberta.embeddings.word_embeddings(start_tokens),
                                                embedding_output_kg[i, len_start: min(512-len_end, c-len_end)],
                                                self.roberta.embeddings.word_embeddings(end_tokens)], dim=0))
                        new_attention_mask.append(attention_mask[i, :512])
                else:
                    if Config.add_kg_flag or Config.add_coref_flag:
                        new_input_ids_kg.append(
                                    torch.cat([self.roberta.embeddings.word_embeddings(start_tokens),
                                            embedding_output_kg[i, bert_s: bert_e],
                                            self.roberta.embeddings.word_embeddings(end_tokens)], dim=0))
                    new_attention_mask.append(attention_mask[i, bert_s - len_start:bert_e + len_end])
        
        return new_input_ids, new_attention_mask, new_input_ids_kg, loss_kg, num_seg, seq_len, c,  len_start, len_end
    
    @add_start_docstrings_to_callable(ROBERTA_INPUTS_DOCSTRING)
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, ent_mask=None, position_ids=None, head_mask=None, inputs_embeds=None, ent_ner=None, ent_pos=None, ent_distance=None, structure_mask=None, label=None, label_mask=None,
                subword_indexs = None, kg_ent_attrs=None, kg_ent_attr_nums=None, kg_ent_attr_lens=None, kg_adj=None, kg_adj_edges=None, kg_ent_labels=None, kg_ent_mask=None,
                coref_h_mapping=None, coref_t_mapping=None, coref_dis=None, coref_lens=None, coref_mention_position=None, coref_label=None, coref_label_mask=None
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for computing the token classification loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.RobertaConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when ``labels`` is provided) :
            Classification loss.
        scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.num_labels)`)
            Classification scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        from transformers import RobertaTokenizer, RobertaForTokenClassification
        import torch

        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaForTokenClassification.from_pretrained('roberta-base')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1] * input_ids.size(1)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, scores = outputs[:2]

        """
        if (Config.add_kg_flag or Config.add_coref_flag):
            new_input_ids, new_attention_mask, new_input_ids_kg, loss_kg, num_seg, seq_len, n, c,  len_start, len_end, bert_starts= self.encode(input_ids, attention_mask, subword_indexs,
                                                                                kg_ent_attrs, kg_ent_attr_nums, kg_ent_attr_lens, kg_adj, kg_adj_edges, kg_ent_labels, kg_ent_mask)
            
            new_attention_mask = torch.stack(new_attention_mask, dim=0)
            tmp_embedding_output_kg = torch.stack(new_input_ids_kg)
            output_kg_list = []
                
            for i in range(len(tmp_embedding_output_kg)):
                arr = np.zeros((512-len(tmp_embedding_output_kg[i]), 1024))
                tmp1 = torch.tensor((arr),device = torch.cuda.current_device())
                output_kg_list.append(torch.cat((tmp_embedding_output_kg[i], tmp1),0))
            
            embedding_output_kg = torch.stack(output_kg_list)
            embedding_output_kg = embedding_output_kg.to(torch.float32)


        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            ner_ids=ent_ner,
            ent_ids=ent_pos,
            structure_mask=structure_mask.float(),
        )
        # get sequence outputs
        sequence_output = outputs[0]
        if (Config.add_kg_flag or Config.add_coref_flag):
            outputs_kg = self.roberta(
                attention_mask=new_attention_mask,
                structure_mask=structure_mask.float(),
                ner_ids=ent_ner,
                ent_ids=ent_pos,
                inputs_embeds=embedding_output_kg
            )
            sequence_output_kg = outputs_kg[0]
            #sequence_output_kg = self.re_cz(num_seg, seq_len, c, sequence_output_kg, attention_mask, len_start, len_end)


            doc_lens = [len(x) for x in subword_indexs]
            loss_coref = 0
            if Config.add_coref_flag:
                word_vec_kg = [layer[starts.nonzero().squeeze(1)] for layer, starts in
                            zip(sequence_output_kg, bert_starts)]
                for i in range(len(word_vec_kg)):
                    arr = np.zeros((512-len(word_vec_kg[i]), 1024))
                    tmp1 = torch.tensor((arr),device = torch.cuda.current_device())
                    word_vec_kg[i] = torch.cat((word_vec_kg[i], tmp1),0)
                word_vec_kg = pad_sequence(word_vec_kg, batch_first=True, padding_value=0)
                word_vec_kg = word_vec_kg.to(torch.float32)
                sequence_output_kg, loss_coref = self.coref_injection(coref_h_mapping, coref_t_mapping, coref_dis,
                                                                    coref_lens, word_vec_kg,
                                                                    coref_mention_position,
                                                                    coref_label,
                                                                    coref_label_mask)
                embedding_output_kg_result = []
                for i in range(n):
                    layer = word_vec_kg[i]
                    starts = bert_starts[i]
                    doc_len = doc_lens[i]

                    starts = torch.cumsum(starts, dim=0) - 1
                    new_layer = torch.cat((layer[0].unsqueeze(0), sequence_output_kg[i][starts[1:]]), dim=0)
                    new_layer[doc_len:].zero_()
                    embedding_output_kg_result.append(new_layer)
                sequence_output_kg = torch.stack(embedding_output_kg_result, dim=0)

            tmp_output_kg_list=[]    
            for i in range(len(sequence_output_kg)):
                arr = np.zeros((512-len(sequence_output_kg[i]), 1024))
                tmp1 = torch.tensor((arr),device = torch.cuda.current_device())
                tmp_output_kg_list.append(torch.cat((sequence_output_kg[i], tmp1),0))
            
            sequence_output_kg = torch.stack(tmp_output_kg_list)

            if Config.add_kg_flag or Config.add_coref_flag:
                sequence_output = torch.max(torch.stack([sequence_output, sequence_output_kg]), dim=0)[0]
        outputs = sequence_output
        
        

        # projection: dim reduction

        outputs = outputs.to(torch.float32)
        outputs = torch.relu(self.dim_reduction(outputs))
        ent_rep = torch.matmul(ent_mask, outputs)

        # prepare entity rep
        ent_rep_h = ent_rep.unsqueeze(2).repeat(1, 1, self.max_ent_cnt, 1)
        ent_rep_t = ent_rep.unsqueeze(1).repeat(1, self.max_ent_cnt, 1, 1)

        # concate distance feature
        if self.with_naive_feature:
            ent_rep_h = torch.cat([ent_rep_h, self.distance_emb(ent_distance)], dim=-1)
            ent_rep_t = torch.cat([ent_rep_t, self.distance_emb(20 - ent_distance)], dim=-1)

        ent_rep_h = self.dropout(ent_rep_h)
        ent_rep_t = self.dropout(ent_rep_t)
        logits = self.bili(ent_rep_h, ent_rep_t)
        loss_fct = BCEWithLogitsLoss(reduction='none')

        loss_all_ent_pair = loss_fct(logits.view(-1, self.num_labels), label.float().view(-1, self.num_labels))
        # loss_all_ent_pair: [bs, max_ent_cnt, max_ent_cnt]
        # label_mask: [bs, max_ent_cnt, max_ent_cnt]
        loss_all_ent_pair = loss_all_ent_pair.view(-1, self.max_ent_cnt, self.max_ent_cnt, self.num_labels)
        loss_all_ent_pair = torch.mean(loss_all_ent_pair, dim=-1)
        loss_per_example = torch.sum(loss_all_ent_pair * label_mask, dim=[1, 2]) / torch.sum(label_mask, dim=[1, 2])
        loss = torch.mean(loss_per_example)

        logits = torch.sigmoid(logits)
        return (loss, logits), loss_kg, loss_coref  # (loss), logits