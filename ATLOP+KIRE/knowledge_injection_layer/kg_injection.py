from torch import nn

from knowledge_injection_layer.aggregation import Aggregation
from knowledge_injection_layer.kg_encoder import Rela_triple_enc

class Kg_Injection(nn.Module):
    def __init__(self, ent_hidden_dim, rel_hidden_dim, gcn_layer_nums, gcn_head_nums, gcn_in_drop,
                 kg_num_hidden_layers, hidden_dim, kg_intermediate_dim, kg_num_attention_heads, kg_attention_probs_dropout_prob):

        super(Kg_Injection, self).__init__()
        self.kg_encoder = Rela_triple_enc(ent_hidden_dim, rel_hidden_dim, gcn_layer_nums, gcn_head_nums, gcn_in_drop)
        self.aggragation = Aggregation(kg_num_hidden_layers, hidden_dim, ent_hidden_dim, kg_intermediate_dim, kg_num_attention_heads, kg_attention_probs_dropout_prob)

    def forward(self, entity_attrs, attrs_nums, entity_attr_lens, kg_adj, entity_doc_nodeid, doc_hidden_states, attention_mask, attention_mask_ent, kg_ent_labels, kg_adj_edges):


        kg_input_ent, kg_candicate = self.kg_encoder(entity_attrs, attrs_nums,
                                                     entity_attr_lens, kg_adj,
                                                     entity_doc_nodeid, kg_adj_edges)

        new_doc_hidden_states = self.aggragation(doc_hidden_states, kg_input_ent, attention_mask, attention_mask_ent)

        loss_kg = self.aggragation.get_kg_loss(new_doc_hidden_states, kg_candicate, kg_ent_labels)

        return new_doc_hidden_states, loss_kg