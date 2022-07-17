"""
kg 融合层
1. ERNIE
2. gate
"""
import torch
from torch import nn
import torch.nn.functional as F

from knowledge_injection_layer.config import Config


class Kg_Adaption(nn.Module):
    def __init__(self, hidden_dim, ent_hidden_dim, kg_intermediate_dim, kg_num_attention_heads, kg_attention_probs_dropout_prob, adaption_type='gate', kg_align_loss=False):
        """

        :param hidden_dim:
        :param ent_hidden_dim:
        :param kg_intermediate_dim:
        :param kg_num_attention_heads:
        :param kg_attention_probs_dropout_prob:
        :param adaption_type:
        :param kg_align_loss: 是否使用kg align loss
        """
        super(Kg_Adaption, self).__init__()
        self.kg_num_attention_heads = kg_num_attention_heads
        self.adaption_type = adaption_type

        if adaption_type == 'gate':  # gate 方式进行融合
            self.w0 = nn.Linear(ent_hidden_dim, hidden_dim)  # 维护维度一致
            self.w = nn.Linear(hidden_dim*3, 1)
        else:  # ERNIE方式
            layers = []
            for i in range(Config.kg_num_hidden_layers):
                layers.append(adaption_layer(hidden_dim, ent_hidden_dim, kg_intermediate_dim, kg_num_attention_heads, kg_attention_probs_dropout_prob))
            self.layers = nn.ModuleList(layers)

        if kg_align_loss:
            # kg loss prediction
            self.predict_dense = nn.Linear(hidden_dim, ent_hidden_dim)
            self.LayerNorm = nn.LayerNorm(ent_hidden_dim, eps=1e-5)

            self.cross_loss = nn.CrossEntropyLoss(ignore_index=-100)

        # 加self-attention层
        # nn.MultiheadAttention

    def _forward_gate(self, doc_hidden_states, doc_reps, ent_hidden_states, attention_mask_ent):
        """
        :param doc_hidden_states: <batch, max_doc_len,-1> 上下文编码的token embedding
        :param doc_reps: <batch,-1> 文档表示
        :param ent_hidden_states: <batch, max_doc_len, -1> 实体kg embedding, 每个token映射的
        :param attention_mask_ent: <> 1表示不是mask的
        :return:
        """
        max_doc_len = doc_hidden_states.size(1)
        ent_hidden_states = self.w0(ent_hidden_states)
        doc_reps = doc_reps.unsqueeze(1).repeat(1, max_doc_len, 1)
        gate = F.sigmoid(self.w(torch.cat([doc_reps, doc_hidden_states, ent_hidden_states], dim=-1)))
        ent_hidden_states = torch.where(attention_mask_ent.unsqueeze(2).repeat(1, 1, doc_hidden_states.size(2)).byte(), ent_hidden_states, doc_hidden_states)
        new_doc_hidden_states = gate * ent_hidden_states + (1 - gate) * doc_hidden_states
        return new_doc_hidden_states

    def _attention_mask_expand(self, attention_mask):
        """

        :param attention_mask: <batch, max_doc_len> 1 表示需要的
        :return: extended_attention_mask <batch*head_nums, max_doc_len, max_doc_len> 1表示需要mask的
        """
        max_doc_len = attention_mask.size(1)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (~extended_attention_mask) * -10000.0
        extended_attention_mask = extended_attention_mask.repeat(1, self.kg_num_attention_heads, max_doc_len, 1) \
                                  .reshape(-1, max_doc_len, max_doc_len)
        return extended_attention_mask

    def _forward_loss(self, doc_hidden_states, ent_hidden_states, attention_mask, attention_mask_ent):
        """

        :param doc_hidden_states:
        :param ent_hidden_states:
        :param attention_mask:
        :param attention_mask_ent:
        :return: layer_output： 融合知识后实体表示
                 layer_output_ent: 重构后实体表示
        """
        # 1. self_attn_layer
        _doc_hidden_states = doc_hidden_states.permute(1, 0, 2)
        _attention_mask = self._attention_mask_expand(attention_mask)
        word_attn_output, _ = self.word_self_att(_doc_hidden_states, _doc_hidden_states, _doc_hidden_states, attn_mask=_attention_mask)
        word_attn_output = word_attn_output.permute(1, 0, 2)
        word_attn_output = self.word_self_output(word_attn_output, doc_hidden_states)  # 加入残差链接

        _ent_hidden_states = ent_hidden_states.permute(1, 0, 2)
        _ent_attention_mask = self._attention_mask_expand(attention_mask_ent)
        ent_attn_output, _ = self.ent_self_att(_ent_hidden_states, _ent_hidden_states, _ent_hidden_states, attn_mask=_ent_attention_mask)
        ent_attn_output = ent_attn_output.permute(1, 0, 2)
        ent_attn_output = self.ent_self_output(ent_attn_output, ent_hidden_states)  # 加入残差链接
        ent_attn_output = ent_attn_output * attention_mask_ent.unsqueeze(-1)

        # 2. Intermediate layer
        word_hidden_states = self.intermediate_dense(word_attn_output)
        ent_hidden_states_ent = self.intermediate_dense_ent(ent_attn_output)
        hidden_states = F.gelu(word_hidden_states + ent_hidden_states_ent)

        # 3.output
        layer_output = self.final_word_self_output(hidden_states, word_attn_output)
        layer_output_ent = self.final_ent_self_output(hidden_states, ent_attn_output)
        return layer_output, layer_output_ent

    def _new_forward_loss(self, doc_hidden_states, ent_hidden_states, attention_mask, attention_mask_ent):
        all_encoder_layers = []
        for layer_module in self.layers:
            doc_hidden_states, ent_hidden_states = layer_module(doc_hidden_states, ent_hidden_states, attention_mask, attention_mask_ent)
            all_encoder_layers.append(doc_hidden_states)
        return all_encoder_layers[-1], ent_hidden_states

    def get_kg_loss(self, hidden_states, kg_candicate, kg_ent_labels):
        """

        :param hidden_states: new_doc_hidden_states 融合kg的token表示
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

    def forward(self, doc_hidden_states, doc_reps, ent_hidden_states, attention_mask, attention_mask_ent):
        """
        :param doc_hidden_states: <batch, max_doc_len,-1> 上下文编码的token embedding
        :param doc_reps: <batch,-1> 文档表示
        :param ent_hidden_states: <batch, max_doc_len, -1> 实体kg embedding, 每个token映射的
        :param attention_mask_ent <batch, max_doc_len> 1 表示是实体，针对每个token进行单独判断
        :param attention_mask
        :return:
        """
        if self.adaption_type == 'gate':
            new_doc_hidden_states = self._forward_gate(doc_hidden_states, doc_reps, ent_hidden_states, attention_mask_ent)
            return new_doc_hidden_states
        else:
            new_doc_hidden_states, _ = self._new_forward_loss(doc_hidden_states, ent_hidden_states, attention_mask, attention_mask_ent)
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

        :param attention_mask: <batch, max_doc_len> 1 表示需要的
        :return: extended_attention_mask <batch*head_nums, max_doc_len, max_doc_len> 1表示需要mask的
        """
        max_doc_len = attention_mask.size(1)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (~extended_attention_mask) * -10000.0
        extended_attention_mask = extended_attention_mask.repeat(1, self.kg_num_attention_heads, max_doc_len, 1) \
                                  .reshape(-1, max_doc_len, max_doc_len)
        return extended_attention_mask

    def forward(self, doc_hidden_states, ent_hidden_states, attention_mask, attention_mask_ent):
        """

                :param doc_hidden_states:
                :param ent_hidden_states:
                :param attention_mask:
                :param attention_mask_ent:
                :return: layer_output： 融合知识后实体表示
                         layer_output_ent: 重构后实体表示
                """
        # 1. self_attn_layer
        _doc_hidden_states = doc_hidden_states.permute(1, 0, 2)
        _attention_mask = self._attention_mask_expand(attention_mask)
        # print(_doc_hidden_states.size())
        # print(_attention_mask.size())

        word_attn_output, _ = self.word_self_att(_doc_hidden_states, _doc_hidden_states, _doc_hidden_states,
                                                 attn_mask=_attention_mask)
        word_attn_output = word_attn_output.permute(1, 0, 2)
        word_attn_output = self.word_self_output(word_attn_output, doc_hidden_states)  # 加入残差链接

        _ent_hidden_states = ent_hidden_states.permute(1, 0, 2)
        # print(1, _ent_hidden_states.size())
        _ent_attention_mask = self._attention_mask_expand(attention_mask_ent)
        # print(_ent_hidden_states.size())
        # print(_ent_attention_mask.size())

        ent_attn_output, _ = self.ent_self_att(_ent_hidden_states, _ent_hidden_states, _ent_hidden_states,
                                               attn_mask=_ent_attention_mask)
        # print(3, ent_attn_output.size())
        ent_attn_output = ent_attn_output.permute(1, 0, 2)
        ent_attn_output = self.ent_self_output(ent_attn_output, ent_hidden_states)  # 加入残差链接
        # print(5, ent_attn_output.size())
        ent_attn_output = ent_attn_output * attention_mask_ent.unsqueeze(-1)
        # print(6, ent_attn_output)
        # 2. Intermediate layer
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