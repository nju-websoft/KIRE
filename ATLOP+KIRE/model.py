import torch
import torch.nn as nn
from opt_einsum import contract
from long_seq import process_long_input
from losses import ATLoss
from knowledge_injection_layer.coref_triple_enc import Coref_triple_enc
from knowledge_injection_layer.config import Config
from knowledge_injection_layer.modules import Combineloss
from knowledge_injection_layer.kg_injection import Kg_Injection
# from transformers.modeling_bert import
import numpy as np

class DocREModel(nn.Module):
    def __init__(self, config, model, emb_size=768, block_size=64, num_labels=-1):
        super().__init__()
        self.config = config
        self.model = model
        self.hidden_size = config.hidden_size
        self.loss_fnt = ATLoss()

        self.head_extractor = nn.Linear(2 * config.hidden_size, emb_size)
        self.tail_extractor = nn.Linear(2 * config.hidden_size, emb_size)
        self.bilinear = nn.Linear(emb_size * block_size, config.num_labels)

        self.emb_size = emb_size
        self.block_size = block_size
        self.num_labels = num_labels

        # Knowledge_injection_layer
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


    def encode(self, input_ids, attention_mask, subword_indexs,kg_ent_attrs, kg_ent_attr_nums, kg_ent_attr_lens, kg_adj, kg_adj_edges, kg_ent_labels, kg_ent_mask
               ,coref_h_mapping, coref_t_mapping, coref_dis, coref_lens, coref_mention_position, coref_label, coref_label_mask):
        config = self.config
        if config.transformer_type == "bert":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id]
        elif config.transformer_type == "roberta":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id, config.sep_token_id]
        sequence_output, attention, loss_kg, loss_coref = process_long_input(self.model, input_ids, attention_mask, subword_indexs, start_tokens, end_tokens,
            self.kg_injection, kg_ent_attrs, kg_ent_attr_nums, kg_ent_attr_lens, kg_adj, kg_adj_edges, kg_ent_labels, kg_ent_mask,
                                                                                     self.coref_injection, coref_h_mapping,
                                                                                     coref_t_mapping, coref_dis, coref_lens,
                                                                                     coref_mention_position, coref_label, coref_label_mask)

        return sequence_output, attention, loss_kg, loss_coref

    def get_hrt(self, sequence_output, attention, entity_pos, hts):
        offset = 1 if self.config.transformer_type in ["bert", "roberta"] else 0
        n, h, _, c = attention.size()
        hss, tss, rss = [], [], []
        for i in range(len(entity_pos)):
            entity_embs, entity_atts = [], []
            for e in entity_pos[i]:
                if len(e) > 1:
                    e_emb, e_att = [], []
                    for start, end in e:
                        if start + offset < c:
                            # In case the entity mention is truncated due to limited max seq length.
                            e_emb.append(sequence_output[i, start + offset])
                            e_att.append(attention[i, :, start + offset])
                    if len(e_emb) > 0:
                        e_emb = torch.logsumexp(torch.stack(e_emb, dim=0), dim=0)
                        e_att = torch.stack(e_att, dim=0).mean(0)
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                else:
                    start, end = e[0]
                    if start + offset < c:
                        e_emb = sequence_output[i, start + offset]
                        e_att = attention[i, :, start + offset]
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                entity_embs.append(e_emb)
                entity_atts.append(e_att)

            entity_embs = torch.stack(entity_embs, dim=0)  # [n_e, d]
            entity_atts = torch.stack(entity_atts, dim=0)  # [n_e, h, seq_len]

            ht_i = torch.LongTensor(hts[i]).to(sequence_output.device)
            hs = torch.index_select(entity_embs, 0, ht_i[:, 0])
            ts = torch.index_select(entity_embs, 0, ht_i[:, 1])

            h_att = torch.index_select(entity_atts, 0, ht_i[:, 0])
            t_att = torch.index_select(entity_atts, 0, ht_i[:, 1])
            ht_att = (h_att * t_att).mean(1)
            ht_att = ht_att / (ht_att.sum(1, keepdim=True) + 1e-5)
            rs = contract("ld,rl->rd", sequence_output[i], ht_att)
            hss.append(hs)
            tss.append(ts)
            rss.append(rs)
        hss = torch.cat(hss, dim=0)
        tss = torch.cat(tss, dim=0)
        rss = torch.cat(rss, dim=0)
        return hss, rss, tss

    def forward(self,
                input_ids=None,
                attention_mask=None,
                subword_indexs=None,
                labels=None,
                entity_pos=None,
                hts=None,
                instance_mask=None,
                kg_ent_attrs=None, kg_ent_attr_nums=None, kg_ent_attr_lens=None, kg_adj=None, kg_adj_edges=None, kg_ent_labels=None, kg_ent_mask=None,
                coref_h_mapping=None, coref_t_mapping=None, coref_dis=None, coref_lens=None, coref_mention_position=None, coref_label=None, coref_label_mask=None
                ):

        sequence_output, attention, loss_kg, loss_coref = self.encode(input_ids, attention_mask, subword_indexs,kg_ent_attrs, kg_ent_attr_nums, kg_ent_attr_lens, kg_adj, kg_adj_edges, kg_ent_labels, kg_ent_mask,
                                         coref_h_mapping, coref_t_mapping, coref_dis,coref_lens, coref_mention_position, coref_label, coref_label_mask)

        hs, rs, ts = self.get_hrt(sequence_output, attention, entity_pos, hts)

        hs = torch.tanh(self.head_extractor(torch.cat([hs, rs], dim=1)))
        ts = torch.tanh(self.tail_extractor(torch.cat([ts, rs], dim=1)))
        b1 = hs.view(-1, self.emb_size // self.block_size, self.block_size)
        b2 = ts.view(-1, self.emb_size // self.block_size, self.block_size)
        bl = (b1.unsqueeze(3) * b2.unsqueeze(2)).view(-1, self.emb_size * self.block_size)
        logits = self.bilinear(bl)

        output = (self.loss_fnt.get_label(logits, num_labels=self.num_labels),)
        if labels is not None:
            labels = [torch.tensor(label) for label in labels]
            labels = torch.cat(labels, dim=0).to(logits)
            loss = self.loss_fnt(logits.float(), labels.float())
            output = (loss.to(sequence_output),) + output
        return output, loss_kg, loss_coref
