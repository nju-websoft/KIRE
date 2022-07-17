import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import *
import numpy as np

from knowledge_injection_layer.kg_injection import Kg_Injection
from models.basemodel import BaseModel
from nnet.attention import SelfAttention
from nnet.modules import Classifier, EncoderLSTM, EmbedLayer
from nnet.rgcn import RGCN_Layer
from utils.tensor_utils import rm_pad, split_n_pad
from nnet.attention import MultiHeadAttention
from knowledge_injection_layer.coref_injection import Coref_Injection
from knowledge_injection_layer.config import Config
from knowledge_injection_layer.modules import Combineloss


class GLRE(BaseModel):
    def __init__(self, params, pembeds, loss_weight=None, sizes=None, maps=None, lab2ign=None):
        super(GLRE, self).__init__(params, pembeds, loss_weight, sizes, maps, lab2ign)
        if params['dataset'] == 'docred':
            self.bert_max_len = 512
        elif params['dataset'] == 'dwie':
            self.bert_max_len = 2500

        if params['encoder'] == 'lstm':  # only lstm
            lstm_input = params['word_dim'] + params['type_dim']
            pretrain_hidden_size = params['lstm_dim'] * 2
        elif params['encoder'] == 'plm':  # only plm
            pretrain_hidden_size = params['plm_dim'] #+ params['type_dim']
        else:  # plm+lstm
            lstm_input = params['plm_dim']
            pretrain_hidden_size = params['lstm_dim'] * 2 #+ params['type_dim']

        # 加载bert
        if params['encoder'] != 'lstm':
            if params['dataset'] in ['docred', 'dwie']:
                if 'bert-large' in params['pretrain_l_m']:
                    self.pretrain_lm = BertModel.from_pretrained('../bert_large/')
                elif 'bert-base' in params['pretrain_l_m']:
                    self.pretrain_lm = BertModel.from_pretrained('../bert_base/')
            else:
                self.pretrain_lm = BertModel.from_pretrained('../biobert_large/')


        # 加载lstm
        if params['encoder'] != 'plm':
            self.encoder = EncoderLSTM(input_size=lstm_input,
                                       num_units=params['lstm_dim'],
                                       nlayers=params['bilstm_layers'],
                                       bidir=True,
                                       dropout=params['drop_i'])

        self.pretrain_l_m_linear_re = nn.Linear(pretrain_hidden_size, params['lstm_dim'])

        # global node rep
        self.type_embed = EmbedLayer(num_embeddings=3,
                                     embedding_dim=params['type_dim'],
                                     dropout=0.0)
        rgcn_input_dim = params['lstm_dim'] + params['type_dim']

        self.rgcn_layer = RGCN_Layer(params, rgcn_input_dim, params['rgcn_hidden_dim'], params['rgcn_num_layers'],
                                     relation_cnt=5)
        self.rgcn_linear_re = nn.Linear(params['rgcn_hidden_dim'] * 2, params['rgcn_hidden_dim'])

        if params['rgcn_num_layers'] == 0:
            input_dim = rgcn_input_dim * 2
        else:
            input_dim = params['rgcn_hidden_dim'] * 2

        if params['local_rep']:
            self.local_rep_layer = Local_rep_layer(params)
            if not params['global_rep']:
                input_dim = params['lstm_dim'] * 2
            else:
                input_dim += params['lstm_dim'] * 2

        input_dim += params['dist_dim'] * 2

        if params['context_att']:
            self.self_att = SelfAttention(input_dim, 1.0)
            input_dim = input_dim * 2

        self.mlp_layer = params['mlp_layers']
        if self.mlp_layer > -1:
            hidden_dim = params['mlp_dim']
            layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
            for _ in range(params['mlp_layers'] - 1):
                layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
            self.out_mlp = nn.Sequential(*layers)
            input_dim = hidden_dim

        self.classifier = Classifier(in_size=input_dim,
                                     out_size=sizes['rel_size'],
                                     dropout=params['drop_o'])

        self.rel_size = sizes['rel_size']
        self.context_att = params['context_att']
        self.pretrain_l_m = params['pretrain_l_m']
        self.local_rep = params['local_rep']
        self.global_rep = params['global_rep']
        self.params = params

        # Knowledge_injection_layer
        if Config.add_kg_flag:
            assert self.word_embed.pret_embeds is not None
            self.kg_injection = Kg_Injection(self.word_embed.pret_embeds, Config.kg_freeze_words, Config.ent_hidden_dim, Config.gpuid,
                                             Config.gcn_layer_nums, Config.gcn_in_drop, Config.gcn_out_drop, Config.hidden_dim,
                                             Config.kg_intermediate_size, Config.kg_num_attention_heads, Config.kg_attention_probs_dropout_prob,
                                             Config.adaption_type, Config.kg_align_loss, Config.gcn_type)

        if Config.add_coref_flag:
            if Config.coref_place == 'afterRnn':
                self.coref_injection = Coref_Injection(params['lstm_dim'])
            else:
                exit(-1)

        if Config.add_kg_flag or Config.add_coref_flag:
            # self.kg_linear_transfer = nn.Linear(params['lstm_dim']*2, params['lstm_dim'])
            self.kg_linear_transfer = nn.Sequential(nn.Linear(params['lstm_dim']*2, params['lstm_dim']),
                                                    nn.ReLU(),
                                                    nn.Linear(params['lstm_dim'], params['lstm_dim']))
            # nn.init.eye_(self.kg_linear_transfer.weight.data)
            # nn.init.zeros_(self.kg_linear_transfer.bias.data)

        self.combineloss = Combineloss(Config)


    def encoding_layer(self, word_vec, seq_lens):
        """
        Encoder Layer -> Encode sequences using BiLSTM.
        @:param word_sec [list]
        @:param seq_lens [list]
        """
        ys, doc_rep = self.encoder(torch.split(word_vec, seq_lens.tolist(), dim=0), seq_lens)  # 20, 460, 128
        return ys, doc_rep

    def graph_layer(self, nodes, info, section):
        """
        Graph Layer -> Construct a document-level graph
        The graph edges hold representations for the connections between the nodes.
        Args:
            nodes:
            info:        (Tensor, 5 columns) entity_id, entity_type, start_wid, end_wid, sentence_id
            section:     (Tensor <B, 3>) #entities/#mentions/#sentences per batch
            positions:   distances between nodes (only M-M and S-S)

        Returns: (Tensor) graph, (Tensor) tensor_mapping, (Tensors) indices, (Tensor) node information
        """

        # all nodes in order: entities - mentions - sentences
        nodes = torch.cat(nodes, dim=0)  # e + m + s (all)
        nodes_info = self.node_info(section, info)  # info/node: node type | semantic type | sentence ID

        nodes = torch.cat((nodes, self.type_embed(nodes_info[:, 0])), dim=1)

        # re-order nodes per document (batch)
        nodes = self.rearrange_nodes(nodes, section)
        nodes = split_n_pad(nodes, section.sum(dim=1))  # torch.Size([4, 76, 210]) batch_size * node_size * node_emb

        nodes_info = self.rearrange_nodes(nodes_info, section)
        nodes_info = split_n_pad(nodes_info, section.sum(dim=1),
                                 pad=-1)  # torch.Size([4, 76, 3]) batch_size * node_size * node_type_size

        return nodes, nodes_info

    def node_layer(self, encoded_seq, info, word_sec):
        # SENTENCE NODES
        sentences = torch.mean(encoded_seq, dim=1)  # sentence nodes (avg of sentence words)

        # MENTION & ENTITY NODES
        encoded_seq_token = rm_pad(encoded_seq, word_sec)
        mentions = self.merge_tokens(info, encoded_seq_token)
        entities = self.merge_mentions(info, mentions)  # entity nodes
        return (entities, mentions, sentences)

    def node_info(self, section, info):
        """
        info:        (Tensor, 5 columns) entity_id, entity_type, start_wid, end_wid, sentence_id
        Col 0: node type | Col 1: semantic type | Col 2: sentence id
        """
        typ = torch.repeat_interleave(torch.arange(3).to(self.device), section.sum(dim=0))  # node types (0,1,2)
        rows_ = torch.bincount(info[:, 0]).cumsum(dim=0)
        rows_ = torch.cat([torch.tensor([0]).to(self.device), rows_[:-1]]).to(self.device)  #

        stypes = torch.neg(torch.ones(section[:, 2].sum())).to(self.device).long()  # semantic type sentences = -1
        all_types = torch.cat((info[:, 1][rows_], info[:, 1], stypes), dim=0)
        sents_ = torch.arange(section.sum(dim=0)[2]).to(self.device)
        sent_id = torch.cat((info[:, 4][rows_], info[:, 4], sents_), dim=0)  # sent_id
        return torch.cat((typ.unsqueeze(-1), all_types.unsqueeze(-1), sent_id.unsqueeze(-1)), dim=1)

    @staticmethod
    def rearrange_nodes(nodes, section):
        """
        Re-arrange nodes so that they are in 'Entity - Mention - Sentence' order for each document (batch)
        """
        tmp1 = section.t().contiguous().view(-1).long().to(nodes.device)
        tmp3 = torch.arange(section.numel()).view(section.size(1),
                                                  section.size(0)).t().contiguous().view(-1).long().to(nodes.device)
        tmp2 = torch.arange(section.sum()).to(nodes.device).split(tmp1.tolist())
        tmp2 = pad_sequence(tmp2, batch_first=True, padding_value=-1)[tmp3].view(-1)
        tmp2 = tmp2[(tmp2 != -1).nonzero().squeeze()]  # remove -1 (padded)

        nodes = torch.index_select(nodes, 0, tmp2)
        return nodes

    # bert 长度重组
    def _re_cz(self, num_seg, seq_len, c, context_output, doc_rep, attention_mask, len_start, len_end):
        i = 0
        re_context_output = []
        re_doc_rep = []
        for n_seg, l_i in zip(num_seg, seq_len):
            if l_i <= 512:
                assert n_seg == 1
                if c <= 512:
                    context_output1 = context_output[i]
                else:
                    context_output1 = F.pad(context_output[i][:512], (0, 0, 0, c-512))
                re_context_output.append(context_output1)
            else:
                context_output1 = []
                mask1 = []
                for j in range(i, i + n_seg - 1):
                    if j == i:
                        context_output1.append(context_output[j][:512 - len_end])
                        mask1.append(attention_mask[j][:512 - len_end])
                    else:
                        context_output1.append(context_output[j][len_start:512 - len_end])
                        mask1.append(attention_mask[j][len_start:512 - len_end])

                context_output1 = F.pad(torch.cat(context_output1, dim=0),
                                            (0, 0, 0, c - (n_seg - 1) * (512 - len_end) + (n_seg - 2) * len_start))
                mask1 = F.pad(torch.cat(mask1, dim=0),
                              (0, c - (n_seg - 1) * (512 - len_end) + (n_seg - 2) * len_start))

                context_output2 = context_output[i + n_seg - 1][len_start:]

                # context_output2 = F.pad(context_output2, (0, 0, (n_seg-1)*(512-len_end)-(n_seg-2)*len_start, c-(n_seg-1)*(512-len_end)+(n_seg-2)*len_start-(512-len_start)))
                context_output2 = F.pad(context_output2, (0, 0, l_i - 512 + len_start, c - l_i))
                mask2 = attention_mask[i + n_seg - 1][len_start:]
                mask2 = F.pad(mask2, (l_i - 512 + len_start, c - l_i))
                # mask2 = F.pad(mask2, ((n_seg - 1) * (512 - len_end) - (n_seg - 2) * len_start,
                #                                           c - (n_seg - 1) * (512 - len_end) + (n_seg - 2) * len_start - (512 - len_start)))
                mask = mask1 + mask2 + 1e-10
                context_output1 = (context_output1 + context_output2) / mask.unsqueeze(-1)
                re_context_output.append(context_output1)
            re_doc_rep.append(torch.max(doc_rep[i: i + n_seg], dim=0)[0])
            i += n_seg
        doc_rep = torch.stack(re_doc_rep, dim=0)
        context_output = torch.stack(re_context_output, dim=0)
        return context_output, doc_rep

    def bert_forward(self, batch):
        len_start = 1
        len_end = 1
        start_tokens = [self.params['cls']]  # 仅适用BERT
        end_tokens = [self.params['sep']]
        start_tokens = torch.tensor(start_tokens).to(batch['bert_token'])
        end_tokens = torch.tensor(end_tokens).to(batch['bert_token'])

        n, c = batch['bert_token'].size()
        embedding_output  = self.pretrain_lm.embeddings.word_embeddings(batch['bert_token'])  # 仅词向量 batch*max_leng*768
        loss_kg = None
        if Config.add_kg_flag or Config.add_coref_flag:
            # for context_mask
            context_masks = [mask[starts.nonzero().squeeze(1)] for mask, starts in
                             zip(batch['bert_mask'], batch['bert_starts'])]
            context_masks_pad = []
            for output, word_len in zip(context_masks, batch['section'][:, 3]):
                if output.size(0) < word_len:
                    padding = Variable(output.data.new(1).zero_())
                    output = torch.cat([output, padding.expand(word_len - output.size(0))], dim=0)
                context_masks_pad.append(output)
            context_masks = torch.cat(context_masks_pad, dim=0)
            context_masks = split_n_pad(context_masks, batch['section'][:, 3])

            embedding_output_kg = embedding_output.detach().clone()
            # 取出
            if Config.add_kg_flag:
                word_vec_kg = [layer[starts.nonzero().squeeze(1)] for layer, starts in
                                  zip(embedding_output_kg, batch['bert_starts'])]
                word_vec_kg_pad = []
                for output, word_len in zip(word_vec_kg, batch['section'][:, 3]):
                    if output.size(0) < word_len:
                        padding = Variable(output.data.new(1, 1).zero_())
                        output = torch.cat([output, padding.expand(word_len - output.size(0), output.size(1))], dim=0)
                    word_vec_kg_pad.append(output)
                word_vec_kg = torch.cat(word_vec_kg_pad, dim=0)
                word_vec_kg = split_n_pad(word_vec_kg, batch['section'][:, 3])

                doc_rep = torch.max(word_vec_kg, dim=1)[0]
                # print("context_masks", context_masks.size())
                # print("word_vec_kg", word_vec_kg.size())

                word_vec_kg, loss_kg = self.kg_injection(batch['kg_ent_attrs'], batch['kg_ent_attr_nums'],
                                                         batch['kg_ent_attr_lens'],
                                                         batch['kg_adj'], batch['kg_radj'], batch['kg_ent_labels'],
                                                         word_vec_kg, doc_rep, context_masks, batch['kg_ent_mask'],
                                                         batch['kg_ent_labels'], batch['kg_adj_edges'])

                # 返回
                batch_size = embedding_output_kg.size(0)
                embedding_output_kg_result = []
                for i in range(batch_size):
                    layer = embedding_output_kg[i]
                    starts = batch['bert_starts'][i]
                    doc_len = batch['section'][:, 3][i]
                    # layer[starts.nonzero().squeeze(1)] = word_vec_kg[i][:doc_len]  # 只置换head subtoken

                    starts = torch.cumsum(starts, dim=0) - 1
                    new_layer = torch.cat((layer[0].unsqueeze(0), word_vec_kg[i][starts[1:]]), dim=0)
                    new_layer[doc_len:].zero_()
                    embedding_output_kg_result.append(new_layer)
                embedding_output_kg = torch.stack(embedding_output_kg_result, dim=0)

        # bert 长度分割
        attention_mask = batch['bert_mask']
        new_input_ids, new_attention_mask, num_seg = [], [], []
        if Config.add_kg_flag or Config.add_coref_flag:
            new_input_ids_kg = []
        seq_len = attention_mask.sum(1).cpu().numpy().astype(np.int32).tolist()
        for i, l_i in enumerate(seq_len):
            bert_indexs = [(bert_s, min(l_i-len_end, bert_s+512-len_start-len_end)) for bert_s in range(len_start, l_i-len_end, 512-len_end-len_start)] # 不包含CLS SEP标识符
            num_seg.append(len(bert_indexs))
            for j, (bert_s, bert_e) in enumerate(bert_indexs):
                if j == len(bert_indexs)-1: # 最后一段
                    if l_i <=512:
                        new_input_ids.append(torch.cat([self.pretrain_lm.embeddings.word_embeddings(start_tokens),
                                                        embedding_output[i, len_start: min(512-len_end, c-len_end)],
                                                        self.pretrain_lm.embeddings.word_embeddings(end_tokens)],
                                                       dim=0))
                        if Config.add_kg_flag or Config.add_coref_flag:
                            new_input_ids_kg.append(
                                torch.cat([self.pretrain_lm.embeddings.word_embeddings(start_tokens),
                                           embedding_output_kg[i, len_start: min(512-len_end, c-len_end)],
                                           self.pretrain_lm.embeddings.word_embeddings(end_tokens)], dim=0))
                        new_attention_mask.append(attention_mask[i, :512])

                    else:
                        new_input_ids.append(torch.cat([self.pretrain_lm.embeddings.word_embeddings(start_tokens),
                                                        embedding_output[i, bert_e-512+len_start + len_end: bert_e],
                                                        self.pretrain_lm.embeddings.word_embeddings(end_tokens)], dim=0))
                        if Config.add_kg_flag or Config.add_coref_flag:
                            new_input_ids_kg.append(torch.cat([self.pretrain_lm.embeddings.word_embeddings(start_tokens),
                                                        embedding_output_kg[i, (bert_e-512+len_start+len_end): bert_e],
                                                            self.pretrain_lm.embeddings.word_embeddings(end_tokens)], dim=0))
                        new_attention_mask.append(attention_mask[i, bert_e-512+len_end:bert_e+len_end])
                else:
                    new_input_ids.append(torch.cat([self.pretrain_lm.embeddings.word_embeddings(start_tokens),
                                                    embedding_output[i, bert_s: bert_e],
                                                    self.pretrain_lm.embeddings.word_embeddings(end_tokens)], dim=0))
                    if Config.add_kg_flag or Config.add_coref_flag:
                        new_input_ids_kg.append(torch.cat([self.pretrain_lm.embeddings.word_embeddings(start_tokens),
                                                           embedding_output_kg[i, bert_s: bert_e],
                                                           self.pretrain_lm.embeddings.word_embeddings(end_tokens)], dim=0))
                    new_attention_mask.append(attention_mask[i, bert_s-len_start:bert_e+len_end])

        embedding_output = torch.stack(new_input_ids, dim=0)
        attention_mask = torch.stack(new_attention_mask, dim=0)
        if Config.add_kg_flag or Config.add_coref_flag:
            embedding_output_kg = torch.stack(new_input_ids_kg)

        context_output, doc_rep = self.pretrain_lm(attention_mask=attention_mask, inputs_embeds=embedding_output)[:2]

        # bert 长度重组
        context_output, doc_rep = self._re_cz(num_seg, seq_len, c, context_output, doc_rep, attention_mask, len_start, len_end)
        context_output = [layer[starts.nonzero().squeeze(1)] for layer, starts in
                          zip(context_output, batch['bert_starts'])]
        context_output_pad = []
        for output, word_len in zip(context_output, batch['section'][:, 3]):
            if output.size(0) < word_len:
                padding = Variable(output.data.new(1, 1).zero_())
                output = torch.cat([output, padding.expand(word_len - output.size(0), output.size(1))], dim=0)
            context_output_pad.append(output)
        context_output = torch.cat(context_output_pad, dim=0)

        if Config.add_kg_flag or Config.add_coref_flag:
            context_output_kg, doc_rep_kg = self.pretrain_lm(attention_mask=attention_mask,
                                                       inputs_embeds=embedding_output_kg)[:2]
            # bert 长度重组
            context_output_kg, doc_rep_kg = self._re_cz(num_seg, seq_len, c, context_output_kg, doc_rep_kg, attention_mask,
                                                  len_start, len_end)
            context_output_kg = [layer[starts.nonzero().squeeze(1)] for layer, starts in
                              zip(context_output_kg, batch['bert_starts'])]
            context_output_pad = []
            for output, word_len in zip(context_output_kg, batch['section'][:, 3]):
                if output.size(0) < word_len:
                    padding = Variable(output.data.new(1, 1).zero_())
                    output = torch.cat([output, padding.expand(word_len - output.size(0), output.size(1))], dim=0)
                context_output_pad.append(output)
            context_output_kg = torch.cat(context_output_pad, dim=0)

        if Config.add_kg_flag or Config.add_coref_flag:
            return context_output, doc_rep, context_output_kg, doc_rep_kg, loss_kg
        else:
            return context_output, doc_rep, None, None, None

    def forward(self, batch):

        # encoder layer
        loss_kg = None
        if self.params['encoder'] == 'lstm':
            context_masks = batch['words'].detach().clone().bool()
            context_masks[context_masks > 0] = 1
            context_masks = split_n_pad(context_masks, batch['section'][:, 3])
            if Config.add_kg_flag and Config.onlywc:
                word_vec = self.word_embed(batch['words'])
                word_vec = split_n_pad(word_vec, batch['section'][:, 3])
                word_vec_kg = word_vec.detach().clone()  # 不影响词向量
                doc_rep = torch.max(word_vec_kg, dim=1)[0]
                word_vec_kg, loss_kg = self.kg_injection(batch['kg_ent_attrs'], batch['kg_ent_attr_nums'],
                                                         batch['kg_ent_attr_lens'],
                                                         batch['kg_adj'], batch['kg_radj'], batch['kg_ent_labels'],
                                                         word_vec_kg, doc_rep, context_masks, batch['kg_ent_mask'],
                                                         batch['kg_ent_labels'], batch['kg_adj_edges'])
                word_vec = rm_pad(word_vec, batch['section'][:, 3])
                word_vec_kg = rm_pad(word_vec_kg, batch['section'][:, 3])
                ner_vec = self.ner_emb(batch['ners'])
                ner_vec_kg = ner_vec.detach().clone()
                input_vec = torch.cat([word_vec, ner_vec], dim=-1)
                input_vec_kg = torch.cat([word_vec_kg, ner_vec_kg], dim=-1)
            else:
                input_vec = self.input_layer(batch['words'], batch['ners'])
                if Config.add_kg_flag or Config.add_coref_flag:
                    input_vec_kg = self.input_layer(batch['words'], batch['ners'])
            encoded_seq, doc_rep = self.encoding_layer(input_vec, batch['section'][:, 3])
            if Config.add_kg_flag or Config.add_coref_flag:
                encoded_seq_kg, doc_rep_kg = self.encoding_layer(input_vec_kg, batch['section'][:, 3])

        elif self.params['encoder'] in ['plm', 'plm+lstm']:
            # context_output, doc_rep = self.pretrain_lm(batch['bert_token'], attention_mask=batch['bert_mask'])[:2]
            encoded_seq, doc_rep, encoded_seq_kg, doc_rep_kg, loss_kg = self.bert_forward(batch)

            if self.params['encoder'] == 'plm+lstm':
                encoded_seq, _ = self.encoding_layer(encoded_seq, batch['section'][:, 3])
                encoded_seq = rm_pad(encoded_seq, batch['section'][:, 3])
                if Config.add_kg_flag or Config.add_coref_flag:
                    encoded_seq_kg, _ = self.encoding_layer(encoded_seq_kg, batch['section'][:, 3])
                    encoded_seq_kg = rm_pad(encoded_seq_kg, batch['section'][:, 3])

            encoded_seq = split_n_pad(encoded_seq, batch['section'][:, 3])  # 文档为单位
            if Config.add_kg_flag or Config.add_coref_flag:
                encoded_seq_kg = split_n_pad(encoded_seq_kg, batch['section'][:, 3])

        encoded_seq = self.pretrain_l_m_linear_re(encoded_seq)
        # doc_rep = self.pretrain_l_m_linear_re(doc_rep)
        if Config.add_kg_flag or Config.add_coref_flag:
            encoded_seq_kg = self.pretrain_l_m_linear_re(encoded_seq_kg)
        # doc_rep_kg = self.pretrain_l_m_linear_re(doc_rep_kg)

        # Knowledge Injection layer
        '''======加入共指信息=======(更新指代词表示)'''
        loss_coref = None
        if Config.add_coref_flag and Config.coref_place == 'afterRnn':
            encoded_seq_kg, loss_coref = self.coref_injection(batch['coref_h_mapping'], batch['coref_t_mapping'],
                                                            batch['coref_lens'], encoded_seq_kg,
                                                            batch['coref_mention_position'], batch['coref_label'], batch['coref_label_mask'])

        '''======加入kg信息======（更新实体mention表示）'''
        # if not Config.onlywc and Config.add_kg_flag:
        #     encoded_seq_kg, loss_kg = self.kg_injection(batch['kg_ent_attrs'], batch['kg_ent_attr_nums'], batch['kg_ent_attr_lens'],
        #                                              batch['kg_adj'], batch['kg_ent_labels'],
        #                                              encoded_seq_kg, doc_rep, context_masks, batch['kg_ent_mask'], batch['kg_ent_labels'], batch['kg_adj_edges'])
        if Config.add_kg_flag or Config.add_coref_flag:
            # encoded_seq = torch.mean(torch.stack([encoded_seq, encoded_seq_kg]), dim=0)
            encoded_seq = self.kg_linear_transfer(torch.cat([encoded_seq, encoded_seq_kg], dim=-1))

        # Task layer
        encoded_seq = rm_pad(encoded_seq, batch['section'][:, 3])
        encoded_seq = split_n_pad(encoded_seq, batch['word_sec'])  # 句子为单位

        nodes = self.node_layer(encoded_seq, batch['entities'], batch['word_sec'])
        init_nodes = nodes

        nodes, nodes_info = self.graph_layer(nodes, batch['entities'], batch['section'][:, 0:3])

        nodes, _ = self.rgcn_layer(nodes, batch['rgcn_adjacency'], batch['section'][:, 0:3])
        entity_size = batch['section'][:, 0].max()
        r_idx, c_idx = torch.meshgrid(torch.arange(entity_size).to(self.device),
                                      torch.arange(entity_size).to(self.device))
        relation_rep_h = nodes[:, r_idx]
        relation_rep_t = nodes[:, c_idx]

        if self.local_rep:
            entitys_pair_rep_h, entitys_pair_rep_t = self.local_rep_layer(batch['entities'], batch['section'],
                                                                          init_nodes, nodes)
            if not self.global_rep:
                relation_rep_h = entitys_pair_rep_h
                relation_rep_t = entitys_pair_rep_t
            else:
                relation_rep_h = torch.cat((relation_rep_h, entitys_pair_rep_h), dim=-1)
                relation_rep_t = torch.cat((relation_rep_t, entitys_pair_rep_t), dim=-1)

        dis_h_2_t = batch['distances_dir'] + 10
        dis_t_2_h = -batch['distances_dir'] + 10
        dist_dir_h_t_vec = self.dist_embed_dir(dis_h_2_t)
        dist_dir_t_h_vec = self.dist_embed_dir(dis_t_2_h)

        relation_rep_h = torch.cat((relation_rep_h, dist_dir_h_t_vec), dim=-1)
        relation_rep_t = torch.cat((relation_rep_t, dist_dir_t_h_vec), dim=-1)

        graph_select = torch.cat((relation_rep_h, relation_rep_t), dim=-1)

        if self.context_att:
            relation_mask = torch.sum(torch.ne(batch['multi_relations'], 0), -1).gt(0)
            graph_select = self.self_att(graph_select, graph_select, relation_mask)

        ## Classification
        r_idx, c_idx = torch.meshgrid(torch.arange(nodes_info.size(1)).to(self.device),
                                      torch.arange(nodes_info.size(1)).to(self.device))
        select, _ = self.select_pairs(nodes_info, (r_idx, c_idx), self.dataset)
        graph_select = graph_select[select]
        if self.mlp_layer > -1:
            graph_select = self.out_mlp(graph_select)
        graph = self.classifier(graph_select)

        loss_re, stats, preds, pred_pairs, multi_truth, mask, truth, output = self.estimate_loss(graph,
                                                                                         batch['relations'][select],
                                                                                         batch['multi_relations'][select])

        if not Config.coref_kl_loss:
            loss_coref = None

        if not Config.kg_align_loss:
            loss_kg = None

        loss = self.combineloss(loss_re, loss_coref, loss_kg)

        return loss, stats, preds, select, pred_pairs, multi_truth, mask, truth, loss_re, loss_coref, loss_kg, output


class Local_rep_layer(nn.Module):
    def __init__(self, params):
        super(Local_rep_layer, self).__init__()
        input_dim = params['rgcn_hidden_dim']
        self.device = torch.device("cuda" if params['gpu'] != -1 else "cpu")

        self.multiheadattention = MultiHeadAttention(input_dim, num_heads=params['att_head_num'],
                                                     dropout=params['att_dropout'])
        self.multiheadattention1 = MultiHeadAttention(input_dim, num_heads=params['att_head_num'],
                                                      dropout=params['att_dropout'])

    def forward(self, info, section, nodes, global_nodes):
        """
            :param info: mention_size * 5  <entity_id, entity_type, start_wid, end_wid, sentence_id, origin_sen_id, node_type>
            :param section batch_size * 3 <entity_size, mention_size, sen_size>
            :param nodes <batch_size * node_size>
        """
        entities, mentions, sentences = nodes  # entity_size * dim
        entities = global_nodes

        entity_size = section[:, 0].max()
        mentions = split_n_pad(mentions, section[:, 1])

        mention_sen_rep = F.embedding(info[:, 4], sentences)  # mention_size * sen_dim
        mention_sen_rep = split_n_pad(mention_sen_rep, section[:, 1])

        eid_ranges = torch.arange(0, max(info[:, 0]) + 1).to(self.device)
        eid_ranges = split_n_pad(eid_ranges, section[:, 0], pad=-2)  # batch_size * men_size

        r_idx, c_idx = torch.meshgrid(torch.arange(entity_size).to(self.device),
                                      torch.arange(entity_size).to(self.device))
        query_1 = entities[:, r_idx]  # 2 * 30 * 30 * 128
        query_2 = entities[:, c_idx]

        info = split_n_pad(info, section[:, 1], pad=-1)
        m_ids, e_ids = torch.broadcast_tensors(info[:, :, 0].unsqueeze(1), eid_ranges.unsqueeze(-1))
        index_m = torch.ne(m_ids, e_ids).to(self.device)  # batch_size * entity_size * mention_size
        index_m_h = index_m.unsqueeze(2).repeat(1, 1, entity_size, 1)\
                                        .reshape(index_m.shape[0], entity_size * entity_size, -1)\
                                        .to(self.device)

        index_m_t = index_m.unsqueeze(1).repeat(1, entity_size, 1, 1)\
                                        .reshape(index_m.shape[0], entity_size * entity_size, -1)\
                                        .to(self.device)

        entitys_pair_rep_h, h_score = self.multiheadattention(mention_sen_rep, mentions, query_2, index_m_h)
        entitys_pair_rep_t, t_score = self.multiheadattention1(mention_sen_rep, mentions, query_1, index_m_t)
        return entitys_pair_rep_h, entitys_pair_rep_t


class BERT(nn.Module):
    def __init__(self, dataset):
        if dataset == 'docred':
            self.bert_max_len = 512  # 预设的bert最大长度
        elif dataset == 'dwie':
            self.bert_max_len = 2500