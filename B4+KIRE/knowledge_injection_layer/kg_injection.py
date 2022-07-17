from torch import nn

from knowledge_injection_layer.kg_adaption import Kg_Adaption
from knowledge_injection_layer.kg_encoder import Kg_Encoder
import time
import torch

class Kg_Injection(nn.Module):
    def __init__(self, pembeds, kg_freeze_words, ent_hidden_dim, gpuid, gcn_layer_nums, gcn_in_drop, gcn_out_drop,
                 hidden_dim, kg_intermediate_dim, kg_num_attention_heads, kg_attention_probs_dropout_prob, adaption_type='gate', kg_align_loss=False, gcn_type='N_DS'):
        """

        :param pembeds:
        :param kg_freeze_words:
        :param ent_hidden_dim:
        :param gpuid:
        :param gcn_in_drop:
        :param gcn_out_drop:
        :param hidden_dim:
        :param kg_intermediate_dim:
        :param kg_num_attention_heads:
        :param kg_attention_probs_dropout_prob:
        :param adaption_type:
        :param kg_align_loss: 是否使用kg align loss
        """
        super(Kg_Injection, self).__init__()
        self.kg_encoder = Kg_Encoder(pembeds, kg_freeze_words, ent_hidden_dim, gpuid, gcn_layer_nums, gcn_in_drop, gcn_out_drop, gcn_type)
        self.kg_adaption = Kg_Adaption(hidden_dim, ent_hidden_dim, kg_intermediate_dim, kg_num_attention_heads, kg_attention_probs_dropout_prob, adaption_type, kg_align_loss)
        self.kg_align_loss = kg_align_loss

    def forward(self, entity_attrs, attrs_nums, entity_attr_lens, kg_adj, entity_doc_nodeid, doc_hidden_states, doc_reps, attention_mask, attention_mask_ent, kg_ent_labels, kg_adj_edges):
        """

        :param  entity_attrs: <batch_attr_nums, max_attr_len>
        :param   attrs_nums: [[2,8], [3,3,3],[4,4]]
        :param   entity_attr_lens: CPU 1D tensor
        :param   kg_adj: list [[sparse_tensor]] [batch_size, <node_size, node_size>>  # node_size = entity_size(top 顺序维持一致) + others
        :param  entity_doc_nodeid: [batch, max_len, nid]  每个token对应nid编码
        :param doc_hidden_states: <batch, max_doc_len,-1> 上下文编码的token embedding
        :param doc_reps: <batch,-1> 文档表示
        :param attention_mask
        :param attention_mask_ent <batch, max_doc_len> 1 表示是实体，针对每个token进行单独判断
        :param kg_ent_labels ent align的真值
        :return: new_doc_hidden_states： 融合知识后实体表示
                  loss_kg： align 损失
        """

        # kg 实体表示学习
        # t1 = time.time()
        kg_input_ent, kg_candicate = self.kg_encoder(entity_attrs, attrs_nums,
                                                     entity_attr_lens, kg_adj,
                                                     entity_doc_nodeid, kg_adj_edges)
        # t2 = time.time()
        # print("kg_encoder", t2-t1)  # 3.79s
        # kg 实体表示和token 表示进行融合
        new_doc_hidden_states = self.kg_adaption(doc_hidden_states, doc_reps, kg_input_ent, attention_mask, attention_mask_ent)
        # t3 = time.time()
        # print("kg_adaption", t3-t2)  # 0.0007s

        loss_kg = None
        if self.kg_align_loss:
            loss_kg = self.kg_adaption.get_kg_loss(new_doc_hidden_states, kg_candicate, kg_ent_labels)
        # print("kg_align_loss", time.time() - t3)  # 0.000004s
        return new_doc_hidden_states, loss_kg