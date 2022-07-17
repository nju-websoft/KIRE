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
"""PyTorch BERT model. """


import logging
import math
import os

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
import numpy as np

from transformers.activations import gelu, gelu_new, swish
from transformers import BertConfig
from transformers.file_utils import add_start_docstrings, add_start_docstrings_to_callable
from transformers.modeling_utils import PreTrainedModel, prune_linear_layer

# Modify code here
from knowledge_injection_layer.coref_injection import Coref_Injection
from knowledge_injection_layer.coref_triple_enc import Coref_triple_enc
from knowledge_injection_layer.config import Config
from knowledge_injection_layer.modules import Combineloss
from knowledge_injection_layer.kg_injection import Kg_Injection
# from transformers.modeling_bert import
import numpy as np

from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F


logger = logging.getLogger(__name__)

BERT_PRETRAINED_MODEL_ARCHIVE_MAP = {
    "bert-base-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-pytorch_model.bin",
    "bert-large-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-pytorch_model.bin",
    "bert-base-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-pytorch_model.bin",
    "bert-large-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-pytorch_model.bin",
    "bert-base-multilingual-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased-pytorch_model.bin",
    "bert-base-multilingual-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-pytorch_model.bin",
    "bert-base-chinese": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-pytorch_model.bin",
    "bert-base-german-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-cased-pytorch_model.bin",
    "bert-large-uncased-whole-word-masking": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-pytorch_model.bin",
    "bert-large-cased-whole-word-masking": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-pytorch_model.bin",
    "bert-large-uncased-whole-word-masking-finetuned-squad": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-finetuned-squad-pytorch_model.bin",
    "bert-large-cased-whole-word-masking-finetuned-squad": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-finetuned-squad-pytorch_model.bin",
    "bert-base-cased-finetuned-mrpc": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-finetuned-mrpc-pytorch_model.bin",
    "bert-base-german-dbmdz-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-dbmdz-cased-pytorch_model.bin",
    "bert-base-german-dbmdz-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-dbmdz-uncased-pytorch_model.bin",
    "bert-base-japanese": "https://s3.amazonaws.com/models.huggingface.co/bert/cl-tohoku/bert-base-japanese-pytorch_model.bin",
    "bert-base-japanese-whole-word-masking": "https://s3.amazonaws.com/models.huggingface.co/bert/cl-tohoku/bert-base-japanese-whole-word-masking-pytorch_model.bin",
    "bert-base-japanese-char": "https://s3.amazonaws.com/models.huggingface.co/bert/cl-tohoku/bert-base-japanese-char-pytorch_model.bin",
    "bert-base-japanese-char-whole-word-masking": "https://s3.amazonaws.com/models.huggingface.co/bert/cl-tohoku/bert-base-japanese-char-whole-word-masking-pytorch_model.bin",
    "bert-base-finnish-cased-v1": "https://s3.amazonaws.com/models.huggingface.co/bert/TurkuNLP/bert-base-finnish-cased-v1/pytorch_model.bin",
    "bert-base-finnish-uncased-v1": "https://s3.amazonaws.com/models.huggingface.co/bert/TurkuNLP/bert-base-finnish-uncased-v1/pytorch_model.bin",
    "bert-base-dutch-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/wietsedv/bert-base-dutch-cased/pytorch_model.bin",
}


def load_tf_weights_in_bert(model, config, tf_checkpoint_path):
    """ Load tf checkpoints in a pytorch model.
    """
    try:
        import re
        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info("Converting TensorFlow checkpoint from {}".format(tf_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        logger.info("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        name = name.split("/")
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(
            n in ["adam_v", "adam_m", "AdamWeightDecayOptimizer", "AdamWeightDecayOptimizer_1", "global_step"]
            for n in name
        ):
            logger.info("Skipping {}".format("/".join(name)))
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                scope_names = re.split(r"_(\d+)", m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] == "kernel" or scope_names[0] == "gamma":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "output_bias" or scope_names[0] == "beta":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "output_weights":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "squad":
                pointer = getattr(pointer, "classifier")
            else:
                try:
                    pointer = getattr(pointer, scope_names[0])
                except AttributeError:
                    logger.info("Skipping {}".format("/".join(name)))
                    continue
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        if m_name[-11:] == "_embeddings":
            pointer = getattr(pointer, "weight")
        elif m_name == "kernel":
            array = np.transpose(array)
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        logger.info("Initialize PyTorch weight {}".format(name))
        pointer.data = torch.from_numpy(array)
    return model


def mish(x):
    return x * torch.tanh(nn.functional.softplus(x))


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish, "gelu_new": gelu_new, "mish": mish}


BertLayerNorm = torch.nn.LayerNorm


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config, with_naive_feature, dataset):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        #  add ner and coref embeds
        self.with_naive_feature = with_naive_feature
        if self.with_naive_feature:
            if dataset == "DocRED":
                self.ner_emb = nn.Embedding(7, config.hidden_size, padding_idx=0)
                self.ent_emb = nn.Embedding(42+1, config.hidden_size, padding_idx=0)
            elif dataset == "DWIE":
                self.ner_emb = nn.Embedding(19, config.hidden_size, padding_idx=0)
                self.ent_emb = nn.Embedding(96+1, config.hidden_size, padding_idx=0)

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, ner_ids=None, ent_ids=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + position_embeddings + token_type_embeddings

        #  add ner and coref embeds
        if self.with_naive_feature:
            ner_embeddings = self.ner_emb(ner_ids)
            embeddings += ner_embeddings
            ent_embddings = self.ent_emb(ent_ids)
            embeddings += ent_embddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(self, config, entity_structure):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        # ==================SSAN==================
        self.entity_structure = entity_structure
        if entity_structure != 'none':
            num_structural_dependencies = 5  # 5 distinct dependencies of entity structure, please refer to our paper.
            if entity_structure == 'decomp':
                self.bias_layer_k = nn.ParameterList(
                    [nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.num_attention_heads, self.attention_head_size))) for _ in range(num_structural_dependencies)])
                self.bias_layer_q = nn.ParameterList(
                    [nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.num_attention_heads, self.attention_head_size))) for _ in range(num_structural_dependencies)])
            elif entity_structure == 'biaffine':
                self.bili = nn.ParameterList(
                    [nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.num_attention_heads, self.attention_head_size, self.attention_head_size)))
                     for _ in range(num_structural_dependencies)])
            self.abs_bias = nn.ParameterList(
                [nn.Parameter(torch.zeros(self.num_attention_heads)) for _ in range(num_structural_dependencies)])

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        layer_idx,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        structure_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        # ==================SSAN==================
        # add attentive bias according to structure prior
        # query_layer / key_layer:         [bs, n_head, seq, hidden_per_head]
        # structure_mask[i]:               [bs, n_head, seq, seq]
        # if layer_idx >= 4: # you can set specific layers here with entity structure if you want to.
        if self.entity_structure != 'none':
            for i in range(5):
                if self.entity_structure == 'decomp':
                    attention_bias_q = torch.einsum("bnid,nd->bni", query_layer, self.bias_layer_k[i]).unsqueeze(-1).repeat(1, 1, 1, query_layer.size(2))
                    attention_bias_k = torch.einsum("nd,bnjd->bnj", self.bias_layer_q[i], key_layer).unsqueeze(-2).repeat(1, 1, key_layer.size(2), 1)
                    attention_scores += (attention_bias_q + attention_bias_k + self.abs_bias[i][None, :, None, None]) * structure_mask[i]
                elif self.entity_structure == 'biaffine':
                    attention_bias = torch.einsum("bnip,npq,bnjq->bnij", query_layer, self.bili[i], key_layer)
                    attention_scores += (attention_bias + self.abs_bias[i][None, :, None, None]) * structure_mask[i]

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        return outputs


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config, entity_structure):
        super().__init__()
        self.self = BertSelfAttention(config, entity_structure)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.self.num_attention_heads, self.self.attention_head_size)
        heads = set(heads) - self.pruned_heads  # Convert to set and remove already pruned heads
        for head in heads:
            # Compute how many pruned heads are before the head and move the index accordingly
            head = head - sum(1 if h < head else 0 for h in self.pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        layer_idx,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        structure_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        self_outputs = self.self(
            layer_idx, hidden_states, attention_mask, head_mask, structure_mask, encoder_hidden_states, encoder_attention_mask
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

# TO DO NOW, understand the code
class BertLayer(nn.Module):
    def __init__(self, config, entity_structure):
        super().__init__()
        self.attention = BertAttention(config, entity_structure)
        self.is_decoder = config.is_decoder
        if self.is_decoder:
            self.crossattention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        layer_idx,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        structure_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        self_attention_outputs = self.attention(layer_idx, hidden_states, attention_mask, head_mask, structure_mask)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        if self.is_decoder and encoder_hidden_states is not None:
            cross_attention_outputs = self.crossattention(
                attention_output, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:]  # add cross attentions if we output attention weights

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs
        return outputs


class BertEncoder(nn.Module):
    def __init__(self, config, entity_structure):
        super().__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([BertLayer(config, entity_structure) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        structure_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                i, hidden_states, attention_mask, head_mask[i], structure_mask, encoder_hidden_states, encoder_attention_mask
            )
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)


class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertOnlyNSPHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class BertPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class BertPreTrainedModel(PreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for downloading and loading pretrained models.
    """

    config_class = BertConfig
    pretrained_model_archive_map = BERT_PRETRAINED_MODEL_ARCHIVE_MAP
    load_tf_weights = load_tf_weights_in_bert
    base_model_prefix = "bert"

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


BERT_START_DOCSTRING = r"""
    This model is a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`_ sub-class.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general
    usage and behavior.

    Parameters:
        config (:class:`~transformers.BertConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""

BERT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`transformers.BertTokenizer`.
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
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`, defaults to :obj:`None`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
            if the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask
            is used in the cross-attention if the model is configured as a decoder.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
"""


@add_start_docstrings(
    "The bare Bert Model transformer outputting raw hidden-states without any specific head on top.",
    BERT_START_DOCSTRING,
)
class BertModel(BertPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well
    as a decoder, in which case a layer of cross-attention is added between
    the self-attention layers, following the architecture described in `Attention is all you need`_ by Ashish Vaswani,
    Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the
    :obj:`is_decoder` argument of the configuration set to :obj:`True`; an
    :obj:`encoder_hidden_states` is expected as an input to the forward pass.

    .. _`Attention is all you need`:
        https://arxiv.org/abs/1706.03762

    """

    def __init__(self, config, with_naive_feature, entity_structure, dataset):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config, with_naive_feature, dataset)
        self.encoder = BertEncoder(config, entity_structure)
        self.pooler = BertPooler(config)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        ner_ids=None,
        ent_ids=None,
        structure_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        r"""
    Return:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (:obj:`torch.FloatTensor`: of shape :obj:`(batch_size, hidden_size)`):
            Last layer hidden-state of the first token of the sequence (classification token)
            further processed by a Linear layer and a Tanh activation function. The Linear
            layer weights are trained from the next sentence prediction (classification)
            objective during pre-training.

            This output is usually *not* a good summary
            of the semantic content of the input, you're often better with averaging or pooling
            the sequence of hidden-states for the whole input sequence.
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

        from transformers import BertModel, BertTokenizer
        import torch

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')

        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)

        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple

        """

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if self.config.is_decoder:
                batch_size, seq_length = input_shape
                seq_ids = torch.arange(seq_length, device=device)
                causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
                causal_mask = causal_mask.to(
                    attention_mask.dtype
                )  # causal and attention masks must have same type with pytorch version < 1.3
                extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                    input_shape, attention_mask.shape
                )
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # prepare for split/multi -head attention
        # [bs, n_structure, seq, seq] -> [bs, n_structure, n_head, seq, seq]
        structure_mask = structure_mask.transpose(0, 1)[:, :, None, :, :]

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)

            if encoder_attention_mask.dim() == 3:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
            elif encoder_attention_mask.dim() == 2:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
            else:
                raise ValueError(
                    "Wrong shape for encoder_hidden_shape (shape {}) or encoder_attention_mask (shape {})".format(
                        encoder_hidden_shape, encoder_attention_mask.shape
                    )
                )

            encoder_extended_attention_mask = encoder_extended_attention_mask.to(
                dtype=next(self.parameters()).dtype
            )  # fp16 compatibility
            encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -10000.0
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = (
                    head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                )  # We can specify head_mask for each layer
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype
            )  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds,
            ner_ids=ner_ids, ent_ids=ent_ids
        )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            structure_mask=structure_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[
            1:
        ]  # add hidden_states and attentions if they are here
        return outputs # sequence_output, pooled_output, (hidden_states), (attentions)


@add_start_docstrings(
    """Structured Self-Attention Network""",
    BERT_START_DOCSTRING,
)
class BertForDocRED(BertPreTrainedModel):
    def __init__(self, config, num_labels, max_ent_cnt, dataset, with_naive_feature=False, entity_structure=False):
        super().__init__(config)
        self.dataset = dataset
        self.num_labels = num_labels
        self.max_ent_cnt = max_ent_cnt
        self.with_naive_feature = with_naive_feature
        self.reduced_dim = 128
        self.bert = BertModel(config, with_naive_feature, entity_structure, dataset)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dim_reduction = nn.Linear(config.hidden_size, self.reduced_dim)
        self.feature_size = self.reduced_dim
        if self.with_naive_feature:
            if dataset == "DWIE":
                self.feature_size += 26
                self.distance_emb = nn.Embedding(26, 26, padding_idx=10)
            else:
                self.feature_size += 20
                self.distance_emb = nn.Embedding(20, 20, padding_idx=10)
        self.bili = torch.nn.Bilinear(self.feature_size, self.feature_size, self.num_labels)
        self.hidden_size = config.hidden_size
        self.init_weights()
        # Modify code here, add Knowledge_injection_layer
        if Config.add_kg_flag:
            self.kg_injection = Kg_Injection(torch.from_numpy(np.load('./data/vec.npy')), Config.kg_freeze_words,
                                             Config.ent_hidden_dim, Config.gpuid,
                                             Config.gcn_layer_nums, Config.gcn_in_drop, 
                                             Config.gcn_out_drop, Config.hidden_dim,
                                             Config.kg_intermediate_size,
                                             Config.kg_num_attention_heads,
                                             Config.kg_attention_probs_dropout_prob,
                                             Config.adaption_type, Config.kg_align_loss, Config.gcn_type)
        else:
            self.kg_injection = None
        if Config.add_coref_flag:
            self.coref_injection = Coref_Injection(config.hidden_size)
        else:
            self.coref_injection = None
        self.combineloss = Combineloss(Config)

        if Config.add_kg_flag or Config.add_coref_flag:
            # self.kg_linear_transfer = nn.Linear(params['lstm_dim']*2, params['lstm_dim'])
            if Config.kg_transfer == 'linear':
                self.kg_linear_transfer = nn.Sequential(nn.Linear(config.hidden_size * 2, config.hidden_size),
                                                        nn.ReLU(),
                                                        nn.Linear(config.hidden_size, config.hidden_size))
            else:
                self.kg_linear_transfer = None
        else:
            self.kg_linear_transfer = None

    def encode( self, input_ids, attention_mask, subword_indexs, token_type_ids, ent_ner, ent_pos,
                kg_ent_attrs, kg_ent_attr_nums, kg_ent_attr_lens, kg_adj, kg_adj_edges,kg_radj, kg_ent_labels, kg_ent_mask,
                coref_h_mapping, coref_t_mapping, coref_dis,coref_lens, coref_mention_position, coref_label, coref_label_mask):

        config = self.config
        start_tokens = [config.cls_token_id]
        end_tokens = [config.sep_token_id]
        new_input_ids, new_attention_mask, new_token_type_ids, new_ent_ner, new_ent_pos, new_input_ids_kg, loss_kg, num_seg, seq_len, n, c,  len_start, len_end, bert_strats= self.process(input_ids, attention_mask, subword_indexs, token_type_ids, ent_ner, ent_pos, start_tokens, end_tokens,
                                                                            self.kg_injection, kg_ent_attrs, kg_ent_attr_nums, kg_ent_attr_lens, kg_adj, kg_adj_edges, kg_radj,kg_ent_labels, kg_ent_mask,
                                                                            self.coref_injection, coref_h_mapping,coref_t_mapping,coref_dis, coref_lens,
                                                                            coref_mention_position,coref_label, coref_label_mask,self.kg_linear_transfer)
        return new_input_ids, new_attention_mask, new_token_type_ids, new_ent_ner, new_ent_pos, new_input_ids_kg, loss_kg, num_seg, seq_len, n, c,  len_start, len_end, bert_strats
    
    def process(self, input_ids, attention_mask, subword_indexs, token_type_ids, ent_ner, ent_pos, start_tokens, end_tokens,
                kg_injection, kg_ent_attrs, kg_ent_attr_nums, kg_ent_attr_lens, kg_adj, kg_adj_edges, kg_radj,kg_ent_labels, kg_ent_mask,
                coref_injection, coref_h_mapping,coref_t_mapping, coref_dis,coref_lens,
                coref_mention_position,coref_label, coref_label_mask,kg_linear_transfer):
        n, c = input_ids.size()
        bert_starts = torch.zeros((n,c)).to(input_ids)
        for i in range(n):
            token_start_idxs = [x + 1 for x in subword_indexs[i]] 
            for x in token_start_idxs:
                if x<c:
                    bert_starts[i, x] = 1
        new_input_ids, new_attention_mask, new_token_type_ids, new_ent_ner, new_ent_pos, new_input_ids_kg, loss_kg, num_seg, seq_len, c,  len_start, len_end = self.bert_forward(kg_injection, input_ids, attention_mask, subword_indexs, start_tokens, end_tokens, bert_starts, token_type_ids, ent_ner, ent_pos, 
                                                                                           kg_ent_attrs, kg_ent_attr_nums, kg_ent_attr_lens, kg_adj, kg_adj_edges, kg_radj, kg_ent_labels, kg_ent_mask)
        return new_input_ids, new_attention_mask, new_token_type_ids, new_ent_ner, new_ent_pos, new_input_ids_kg, loss_kg, num_seg, seq_len, n, c,  len_start, len_end, bert_starts
    
    def bert_forward(self, kg_injection, input_ids, attention_mask, subword_indexs, start_tokens, end_tokens, bert_starts, token_type_ids, ent_ner, ent_pos,
                    kg_ent_attrs, kg_ent_attr_nums, kg_ent_attr_lens, kg_adj, kg_adj_edges,kg_radj, kg_ent_labels, kg_ent_mask):
        n, c = input_ids.size()
        start_tokens = torch.tensor(start_tokens).to(input_ids)
        end_tokens = torch.tensor(end_tokens).to(input_ids)
        len_start = start_tokens.size(0)
        len_end = end_tokens.size(0)

        embedding_output = self.bert.embeddings.word_embeddings(input_ids)
        loss_kg = None
        if Config.add_kg_flag or Config.add_coref_flag:
            context_masks = [mask[starts.nonzero().squeeze(1)] for mask, starts in
                                zip(attention_mask, bert_starts)]
            for i in range(len(context_masks)):
                if self.dataset == "DocRED":
                    tmp0 = torch.tensor([0]*(512-len(context_masks[i])),device = torch.cuda.current_device())
                    #tmp0 = torch.tensor([0]*(512-len(context_masks[i])),device = 'cpu')
                else:
                    tmp0 = torch.tensor([0]*(1800-len(context_masks[i])),device = torch.cuda.current_device())
                    #tmp0 = torch.tensor([0]*(1800-len(context_masks[i])),device = 'cpu')
                context_masks[i] = torch.cat((context_masks[i],tmp0),0)
                del tmp0
            context_masks = pad_sequence(context_masks, batch_first=True, padding_value=0)
            embedding_output_kg = embedding_output.detach().clone()

            if Config.add_kg_flag:
                word_vec_kg = [layer[starts.nonzero().squeeze(1)] for layer, starts in zip(embedding_output_kg, bert_starts)]
                for i in range(len(word_vec_kg)):
                    if self.dataset == "DocRED":
                        arr = np.zeros((512-len(word_vec_kg[i]), 768))
                    else:
                        arr = np.zeros((1800-len(word_vec_kg[i]), 768))
                    tmp1 = torch.tensor((arr),device = torch.cuda.current_device())
                    #tmp1 = torch.tensor((arr),device = 'cpu')
                    word_vec_kg[i] = torch.cat((word_vec_kg[i], tmp1),0)
                    del arr,tmp1
                word_vec_kg = pad_sequence(word_vec_kg, batch_first=True, padding_value=0)
                
                context_masks = context_masks.bool()
                doc_rep = torch.max(word_vec_kg, dim=1)[0]
                word_vec_kg = word_vec_kg.to(torch.float32)
            

                word_vec_kg, loss_kg = kg_injection(kg_ent_attrs, kg_ent_attr_nums,
                                                    kg_ent_attr_lens,
                                                    kg_adj, kg_ent_labels,
                                                    word_vec_kg, doc_rep, context_masks, kg_ent_mask,
                                                    kg_ent_labels, kg_adj_edges,kg_radj)

                batch_size = embedding_output_kg.size(0)
                embedding_output_kg_result = []
                for i in range(batch_size):
                    layer = embedding_output_kg[i]
                    starts = bert_starts[i]
                    starts = torch.cumsum(starts, dim=0) - 1
                    new_layer = torch.cat((layer[0].unsqueeze(0), word_vec_kg[i][starts[1:]]), dim=0)
                    embedding_output_kg_result.append(new_layer)

                embedding_output_kg = torch.stack(embedding_output_kg_result, dim=0)
        new_input_ids, new_attention_mask, new_token_type_ids, new_ent_ner, new_ent_pos, num_seg = [], [], [], [], [], []
        
        new_input_ids_kg = []
        seq_len = attention_mask.sum(1).cpu().numpy().astype(np.int32).tolist()
        for i, l_i in enumerate(seq_len):
            bert_indexs = [(bert_s, min(l_i - len_end, bert_s + 512-(len_start+len_end))) for bert_s in
                            range(len_start, l_i - len_end, 512-(len_start+len_end))]
            num_seg.append(len(bert_indexs))
            for j, (bert_s, bert_e) in enumerate(bert_indexs):
                if j == len(bert_indexs) - 1:
                    if l_i <= 512:
                        
                        if Config.add_kg_flag or Config.add_coref_flag:
                            new_input_ids_kg.append(
                                        torch.cat([self.bert.embeddings.word_embeddings(start_tokens),
                                                embedding_output_kg[i, len_start: min(512-len_end, c-len_end)],
                                                self.bert.embeddings.word_embeddings(end_tokens)], dim=0))
                        
                        new_input_ids.append(torch.cat([self.bert.embeddings.word_embeddings(start_tokens),
                                                    embedding_output[i, len_start: min(512-len_end, c-len_end)],
                                                    self.bert.embeddings.word_embeddings(end_tokens)],
                                                    dim=0))
                        new_attention_mask.append(attention_mask[i, :512])
                        new_token_type_ids.append(token_type_ids[i, :512])
                        new_ent_ner.append(ent_ner[i, :512])
                        new_ent_pos.append(ent_pos[i, :512])
                    else:
                        
                        if Config.add_kg_flag or Config.add_coref_flag:
                            new_input_ids_kg.append(
                                    torch.cat([self.bert.embeddings.word_embeddings(start_tokens),
                                               embedding_output_kg[i, (bert_e - 512 + len_start + len_end): bert_e],
                                               self.bert.embeddings.word_embeddings(end_tokens)], dim=0))
                        
                        new_input_ids.append(torch.cat([self.bert.embeddings.word_embeddings(start_tokens),
                                                    embedding_output[i, bert_e - 512 + len_start + len_end: bert_e],
                                                    self.bert.embeddings.word_embeddings(end_tokens)],
                                                    dim=0))
                        new_attention_mask.append(attention_mask[i, bert_e - 512 + len_end:bert_e + len_end])
                        new_token_type_ids.append(token_type_ids[i, bert_e - 512 + len_end:bert_e + len_end])
                        new_ent_ner.append(ent_ner[i, :512])
                        new_ent_pos.append(ent_pos[i, :512])

                else:
                    
                    if Config.add_kg_flag or Config.add_coref_flag:
                        new_input_ids_kg.append(
                                    torch.cat([self.bert.embeddings.word_embeddings(start_tokens),
                                            embedding_output_kg[i, bert_s: bert_e],
                                            self.bert.embeddings.word_embeddings(end_tokens)], dim=0))
                    
                    new_attention_mask.append(attention_mask[i, bert_s - len_start:bert_e + len_end])
                    new_token_type_ids.append(token_type_ids[i, bert_s - len_start:bert_e + len_end])
                    new_ent_ner.append(ent_ner[i, :512])
                    new_ent_pos.append(ent_pos[i, :512])
                    new_input_ids.append(torch.cat([self.bert.embeddings.word_embeddings(start_tokens),
                                                embedding_output[i, bert_s: bert_e],
                                                self.bert.embeddings.word_embeddings(end_tokens)],
                                                dim=0))
        
        embedding_output = torch.stack(new_input_ids, dim=0)
        attention_mask = torch.stack(new_attention_mask, dim=0)
        token_type_ids = torch.stack(new_token_type_ids, dim=0)
        ent_ner = torch.stack(new_ent_ner, dim=0)
        ent_pos = torch.stack(new_ent_pos, dim=0)
        return embedding_output, new_attention_mask, token_type_ids, ent_ner, ent_pos, new_input_ids_kg, loss_kg, num_seg, seq_len, c,  len_start, len_end
    def re_cz(self, num_seg, seq_len, c, outputs, attention_mask, len_start, len_end):
        i = 0
        re_context_output = []
        #re_attention = []
        for n_seg, l_i in zip(num_seg, seq_len):
            if l_i <= 512:
                assert n_seg == 1
                if c <= 512:
                    re_context_output.append(outputs[i])
                    #re_attention.append(attention[i])
                else:
                    context_output1 = F.pad(outputs[i, :512, :], (0, 0, 0, c-512))
                    re_context_output.append(context_output1)
                    #attention1 = F.pad(attention[i][:, :512, :512], (0, c-512, 0, c-512))
                    #re_attention.append(attention1)
            else:
                context_output1 = []
                attention1 = None
                mask1 = []
                for j in range(i, i + n_seg - 1):
                    if j == i:
                        context_output1.append(outputs[j][:512 - len_end, :])
                        #attention1 = F.pad(attention[j][:, :512-len_end, :512-len_end], (0, c-(512-len_end), 0, c-(512-len_end)))
                        mask1.append(attention_mask[j][:512 - len_end])
                    else:
                        context_output1.append(outputs[j][len_start:512 - len_end, :])
                        #attention2 = F.pad(attention[j][:, len_start:512-len_end, len_start:512-len_end],
                        #                               (512-len_end+(j-i-1)*(512-len_end-len_start), c-(512-len_end+(j-i)*(512-len_end-len_start)),
                        #                                512-len_end+(j-i-1)*(512-len_end-len_start), c-(512-len_end+(j-i)*(512-len_end-len_start))))
                        #if attention1 is None:
                        #    attention1 = attention2
                        #else:
                        #    attention1 = attention1 + attention2
                        mask1.append(attention_mask[j][len_start:512 - len_end])
                # for x in context_output1:
                #     print(x.size())
                # print(c)
                # print(c - (n_seg - 1) * (512 - len_end) + (n_seg - 2) * len_start)
                context_output1 = F.pad(torch.cat(context_output1, dim=0),
                                            (0, 0, 0, c - (n_seg - 1) * (512 - len_end) + (n_seg - 2) * len_start))
                #att = attention1 + F.pad(attention[i + n_seg - 1][:, len_start:, len_start:],
                #                         (l_i - 512 + len_start, c - l_i, l_i - 512 + len_start, c - l_i))

                context_output2 = outputs[i + n_seg - 1][len_start:]
                context_output2 = F.pad(context_output2, (0, 0, l_i - 512 + len_start, c - l_i))

                mask1 = F.pad(torch.cat(mask1, dim=0), (0, c - (n_seg - 1) * (512 - len_end) + (n_seg - 2) * len_start))
                mask2 = attention_mask[i + n_seg - 1][len_start:]
                mask2 = F.pad(mask2, (l_i - 512 + len_start, c - l_i))
                mask = mask1 + mask2 + 1e-10
                context_output1 = (context_output1 + context_output2) / mask.unsqueeze(-1)
                re_context_output.append(context_output1)
                #att = att / (att.sum(-1, keepdim=True) + 1e-10)
                #re_attention.append(att)

            i += n_seg
        #attention = torch.stack(re_attention, dim=0)
        context_output = torch.stack(re_context_output, dim=0)
        return context_output
    
    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING)
    def forward(
            self, input_ids=None, attention_mask=None, token_type_ids=None, ent_mask=None, position_ids=None, head_mask=None, inputs_embeds=None, ent_ner=None, ent_pos=None, ent_distance=None, structure_mask=None, label=None, label_mask=None,
                subword_indexs = None, kg_ent_attrs=None, kg_ent_attr_nums=None, kg_ent_attr_lens=None, kg_ent_attr_indexs=None, kg_adj=None,kg_adj_edges=None,  kg_radj=None, kg_ent_labels=None, kg_ent_mask=None,
                coref_h_mapping=None, coref_t_mapping=None, coref_dis=None, coref_lens=None, coref_mention_position=None, coref_label=None, coref_label_mask=None
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`label` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
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

        from transformers import BertTokenizer, BertForSequenceClassification
        import torch

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)

        loss, logits = outputs[:2]

        """
        new_input_ids, new_attention_mask, new_token_type_ids, new_ent_ner, new_ent_pos, new_input_ids_kg, loss_kg, num_seg, seq_len, n, c,  len_start, len_end, bert_starts= self.encode(input_ids, attention_mask, subword_indexs, token_type_ids, ent_ner, ent_pos,
                                                                                kg_ent_attrs, kg_ent_attr_nums, kg_ent_attr_lens, kg_adj, kg_adj_edges,kg_radj, kg_ent_labels, kg_ent_mask,
                                                                                coref_h_mapping, coref_t_mapping, coref_dis,coref_lens, coref_mention_position, coref_label, coref_label_mask)
            
        new_attention_mask = torch.stack(new_attention_mask, dim=0)
        if (Config.add_kg_flag or Config.add_coref_flag):    
            tmp_embedding_output_kg = torch.stack(new_input_ids_kg)

            output_list = []
            output_kg_list = []

            for i in range(len(tmp_embedding_output_kg)):
                arr = np.zeros((512-len(tmp_embedding_output_kg[i]), 768))
                tmp1 = torch.tensor((arr),device = torch.cuda.current_device())
                #tmp1 = torch.tensor((arr),device = 'cpu')
                output_kg_list.append(torch.cat((tmp_embedding_output_kg[i], tmp1),0))
            embedding_output_kg = torch.stack(output_kg_list)


            embedding_output_kg = embedding_output_kg.to(torch.float32)
        outputs= self.bert(
            input_ids=None, 
            attention_mask=new_attention_mask,
            token_type_ids=new_token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=new_input_ids,
            ner_ids=new_ent_ner,
            ent_ids=new_ent_pos,
            structure_mask=structure_mask.float(),
        )
        # get sequence outputs
        sequence_output = outputs[0]
        attention = outputs[-1][-1]
        outputs = sequence_output
        outputs = outputs.to(torch.float32)
        # projection: dim reduction
        outputs= self.re_cz(num_seg, seq_len, c, outputs, new_attention_mask, len_start, len_end)
        loss_coref=0
        if (Config.add_kg_flag or Config.add_coref_flag):
            outputs_kg = self.bert(
                input_ids=None, 
                attention_mask=new_attention_mask,
                token_type_ids=new_token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                structure_mask=structure_mask.float(),
                ner_ids=new_ent_ner,
                ent_ids=new_ent_pos,
                inputs_embeds=embedding_output_kg
            )
            sequence_output_kg = outputs_kg[0]
            attention_kg = outputs_kg[-1][-1]
            sequence_output_kg = self.re_cz(num_seg, seq_len, c, sequence_output_kg, new_attention_mask, len_start, len_end)


            doc_lens = [len(x) for x in subword_indexs]
            if Config.add_coref_flag:
                word_vec_kg = [layer[starts.nonzero().squeeze(1)] for layer, starts in
                            zip(sequence_output_kg, bert_starts)]
                
                for i in range(len(word_vec_kg)):
                    if self.dataset == "DocRED":
                        arr = np.zeros((512-len(word_vec_kg[i]), 768))
                    else:
                        arr = np.zeros((1800-len(word_vec_kg[i]), 768))
                    tmp1 = torch.tensor((arr),device = torch.cuda.current_device())
                    #tmp1 = torch.tensor((arr),device = 'cpu')
                    word_vec_kg[i] = torch.cat((word_vec_kg[i], tmp1),0)
                
                word_vec_kg = pad_sequence(word_vec_kg, batch_first=True, padding_value=0)
                word_vec_kg = word_vec_kg.to(torch.float32)
                sequence_output_kg, loss_coref = self.coref_injection(coref_h_mapping, coref_t_mapping,coref_dis,
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
            '''
            tmp_output_kg_list=[]    
            for i in range(len(sequence_output_kg)):
                if self.dataset == "DocRED":
                    arr = np.zeros((512-len(sequence_output_kg[i]), 768))
                else:
                    arr = np.zeros((1800-len(sequence_output_kg[i]), 768))
                tmp1 = torch.tensor((arr),device = torch.cuda.current_device())
                tmp_output_kg_list.append(torch.cat((sequence_output_kg[i], tmp1),0))
            
            sequence_output_kg = torch.stack(tmp_output_kg_list)
            '''
            if Config.add_kg_flag or Config.add_coref_flag:
                if self.kg_linear_transfer is None:  # 默认max
                    sequence_output = torch.max(torch.stack([outputs, sequence_output_kg]), dim=0)[0]
                else:  # linear方式
                    sequence_output = self.kg_linear_transfer(torch.cat([outputs, sequence_output_kg], dim=-1))
        outputs = torch.relu(self.dim_reduction(outputs))
        ent_rep = torch.matmul(ent_mask, outputs)

        # prepare entity rep
        ent_rep_h = ent_rep.unsqueeze(2).repeat(1, 1, self.max_ent_cnt, 1)
        ent_rep_t = ent_rep.unsqueeze(1).repeat(1, self.max_ent_cnt, 1, 1)

        # concate distance feature
        if self.with_naive_feature:
            ent_rep_h = torch.cat([ent_rep_h, self.distance_emb(ent_distance)], dim=-1)
            if self.dataset == "DWIE":
                ent_rep_t = torch.cat([ent_rep_t, self.distance_emb(26 - ent_distance)], dim=-1)
            else:
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
        return (loss, logits),loss_kg, loss_coref # (loss), logits
