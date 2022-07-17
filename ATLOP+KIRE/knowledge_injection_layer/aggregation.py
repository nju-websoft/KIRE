import torch
from torch import nn
import torch.nn.functional as F


class Aggregation(nn.Module):
    def __init__(self, kg_num_hidden_layers, hidden_dim, ent_hidden_dim, kg_intermediate_dim, kg_num_attention_heads, kg_attention_probs_dropout_prob):

        super(Aggregation, self).__init__()
        self.kg_num_attention_heads = kg_num_attention_heads

        layers = []
        for i in range(kg_num_hidden_layers):
            layers.append(adaption_layer(hidden_dim, ent_hidden_dim, kg_intermediate_dim, kg_num_attention_heads, kg_attention_probs_dropout_prob))
        self.layers = nn.ModuleList(layers)

        self.predict_dense = nn.Linear(hidden_dim, ent_hidden_dim)
        self.LayerNorm = nn.LayerNorm(ent_hidden_dim, eps=1e-5)
        self.cross_loss = nn.CrossEntropyLoss(ignore_index=-100)

    def _forward_loss(self, doc_hidden_states, ent_hidden_states, attention_mask, attention_mask_ent):
        all_encoder_layers = []
        for layer_module in self.layers:
            doc_hidden_states, ent_hidden_states = layer_module(doc_hidden_states, ent_hidden_states, attention_mask, attention_mask_ent)
            all_encoder_layers.append(doc_hidden_states)
        return all_encoder_layers[-1], ent_hidden_states

    def get_kg_loss(self, hidden_states, kg_candicate, kg_ent_labels):
        """

        :param hidden_states: new_doc_hidden_states
        :param kg_candicate: <batch, node_size, -1>
        :param kg_ent_labels
        :return:
        """
        hidden_states = self.predict_dense(hidden_states)
        hidden_states = F.gelu(hidden_states)
        ent_hidden_states = self.LayerNorm(hidden_states) # batch_size, max_len, -1
        predict_ent = torch.bmm(ent_hidden_states, kg_candicate.permute(0, 2, 1)).reshape(-1, kg_candicate.size(1))  # batch_size, max_len, node_size
        loss_ent = self.cross_loss(predict_ent, kg_ent_labels.view(-1))
        return loss_ent

    def forward(self, doc_hidden_states, ent_hidden_states, attention_mask, attention_mask_ent):
        """
        :param doc_hidden_states: <batch, max_doc_len,-1>
        :param doc_reps: <batch,-1>
        :param ent_hidden_states: <batch, max_doc_len, -1>
        :param attention_mask_ent <batch, max_doc_len>
        :param attention_mask
        :return:
        """
        new_doc_hidden_states, _ = self._forward_loss(doc_hidden_states, ent_hidden_states, attention_mask, attention_mask_ent)
        return new_doc_hidden_states


class adaption_layer(nn.Module):
    def __init__(self, hidden_dim, ent_hidden_dim, kg_intermediate_dim, kg_num_attention_heads, kg_attention_probs_dropout_prob):
        super(adaption_layer, self).__init__()
        self.word_self_att = nn.MultiheadAttention(embed_dim=hidden_dim,
                                                   num_heads=kg_num_attention_heads,
                                                   dropout=kg_attention_probs_dropout_prob)
        self.word_self_output = SelfOutput(hidden_dim, hidden_dim)

        self.ent_self_att = nn.MultiheadAttention(embed_dim=ent_hidden_dim,  # ent_hidden_dim = 100
                                                  num_heads=kg_num_attention_heads,
                                                  dropout=kg_attention_probs_dropout_prob)
        self.ent_self_output = SelfOutput(ent_hidden_dim, ent_hidden_dim)

        self.intermediate_dense = nn.Linear(hidden_dim, kg_intermediate_dim)
        self.intermediate_dense_ent = nn.Linear(ent_hidden_dim, kg_intermediate_dim)

        self.final_word_self_output = SelfOutput(kg_intermediate_dim, hidden_dim)
        self.final_ent_self_output = SelfOutput(kg_intermediate_dim, ent_hidden_dim)
        self.kg_num_attention_heads = kg_num_attention_heads

    def _attention_mask_expand(self, attention_mask):
        """

        :param attention_mask: <batch, max_doc_len>
        :return: extended_attention_mask <batch*head_nums, max_doc_len, max_doc_len>
        """
        max_doc_len = attention_mask.size(1)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (~extended_attention_mask) * -10000.0
        extended_attention_mask = extended_attention_mask.repeat(1, self.kg_num_attention_heads, max_doc_len, 1) \
                                  .reshape(-1, max_doc_len, max_doc_len)
        return extended_attention_mask

    def forward(self, doc_hidden_states, ent_hidden_states, attention_mask, attention_mask_ent):
        # 1. self_attn_layer
        _doc_hidden_states = doc_hidden_states.permute(1, 0, 2)
        _attention_mask = self._attention_mask_expand(attention_mask)

        word_attn_output, _ = self.word_self_att(_doc_hidden_states, _doc_hidden_states, _doc_hidden_states,
                                                 attn_mask=_attention_mask)
        word_attn_output = word_attn_output.permute(1, 0, 2)
        word_attn_output = self.word_self_output(word_attn_output, doc_hidden_states)

        _ent_hidden_states = ent_hidden_states.permute(1, 0, 2)
        _ent_attention_mask = self._attention_mask_expand(attention_mask_ent)

        ent_attn_output, _ = self.ent_self_att(_ent_hidden_states, _ent_hidden_states, _ent_hidden_states,
                                               attn_mask=_ent_attention_mask)
        ent_attn_output = ent_attn_output.permute(1, 0, 2)
        ent_attn_output = self.ent_self_output(ent_attn_output, ent_hidden_states)
        ent_attn_output = ent_attn_output * attention_mask_ent.unsqueeze(-1)
        word_hidden_states = self.intermediate_dense(word_attn_output)
        ent_hidden_states_ent = self.intermediate_dense_ent(ent_attn_output)
        hidden_states = F.gelu(word_hidden_states + ent_hidden_states_ent)

        # 3.output
        layer_output = self.final_word_self_output(hidden_states, word_attn_output)
        layer_output_ent = self.final_ent_self_output(hidden_states, ent_attn_output)
        return layer_output, layer_output_ent

class SelfOutput(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_prob=0.5):
        super(SelfOutput, self).__init__()
        self.dense = nn.Linear(input_dim, output_dim)
        self.LayerNorm = nn.LayerNorm(output_dim, eps=1e-5)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states