import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch import nn
import numpy as np
import math
from torch.nn import init
from torch.nn.utils import rnn
from knowledge_injection_layer.coref_injection import Coref_Injection
from knowledge_injection_layer.config import Config
from knowledge_injection_layer.modules import Combineloss
from knowledge_injection_layer.kg_injection import Kg_Injection


class LSTM(nn.Module):
    def __init__(self, config):
        super(LSTM, self).__init__()
        self.config = config

        word_vec_size = config.data_word_vec.shape[0]
        self.word_emb = nn.Embedding(word_vec_size, config.data_word_vec.shape[1])
        self.word_emb.weight.data.copy_(torch.from_numpy(config.data_word_vec))
        # self.word_emb.weight.requires_grad = False

        # self.char_emb = nn.Embedding(config.data_char_vec.shape[0], config.data_char_vec.shape[1])
        # self.char_emb.weight.data.copy_(torch.from_numpy(config.data_char_vec))
        # char_dim = config.data_char_vec.shape[1]
        # char_hidden = 100
        # self.char_cnn = nn.Conv1d(char_dim,  char_hidden, 5)
        self.coref_embed = nn.Embedding(config.max_length, config.coref_size, padding_idx=0)
        self.ner_emb = nn.Embedding(config.ner_size, config.entity_type_size, padding_idx=0)

        input_size = config.data_word_vec.shape[1] + config.coref_size + config.entity_type_size  # + char_hidden
        hidden_size = 128

        # self.rnn = EncoderLSTM(input_size, hidden_size, 1, True, True, 1 - config.keep_prob, False)
        # self.linear_re = nn.Linear(hidden_size*2, hidden_size)  # *4 for 2layer

        self.rnn = EncoderLSTM(input_size, hidden_size, 1, True, False, 1 - config.keep_prob, False)
        self.linear_re = nn.Linear(hidden_size, hidden_size)  # *4 for 2layer

        self.bili = torch.nn.Bilinear(hidden_size + config.dis_size, hidden_size + config.dis_size, config.relation_num)

        self.dis_embed = nn.Embedding(20, config.dis_size, padding_idx=10)

        # Knowledge_injection_layer
        self.kiconfig = Config()
        if self.kiconfig.add_kg_flag:

            self.kg_injection = Kg_Injection(torch.from_numpy(config.data_word_vec), self.kiconfig.kg_freeze_words,
                                             self.kiconfig.ent_hidden_dim, self.kiconfig.gpuid,
                                             self.kiconfig.gcn_layer_nums, self.kiconfig.gcn_in_drop,
                                             self.kiconfig.gcn_out_drop, self.kiconfig.hidden_dim,
                                             self.kiconfig.kg_intermediate_size, self.kiconfig.kg_num_attention_heads,
                                             self.kiconfig.kg_attention_probs_dropout_prob,
                                             self.kiconfig.adaption_type, self.kiconfig.kg_align_loss, self.kiconfig.gcn_type)

        if Config.add_coref_flag:
            if Config.coref_place == 'afterRnn':
                self.coref_injection = Coref_Injection(hidden_size)
            elif Config.coref_place == 'afterWordvec1':
                self.coref_injection = Coref_Injection(input_size)
            else:
                self.coref_injection = Coref_Injection(Config.hidden_dim)

        self.combineloss = Combineloss(self.kiconfig)
        if Config.add_kg_flag or Config.add_coref_flag:
            # self.kg_linear_transfer = nn.Linear(params['lstm_dim']*2, params['lstm_dim'])
            self.kg_linear_transfer = nn.Sequential(nn.Linear(hidden_size * 2, hidden_size),
                                                    nn.ReLU(),
                                                    nn.Linear(hidden_size, hidden_size))

    def forward(self, context_idxs, pos, context_ner, context_char_idxs, context_lens, h_mapping, t_mapping,
                relation_mask, dis_h_2_t, dis_t_2_h, kg_ent_attrs, kg_ent_attr_nums, kg_ent_attr_lens, kg_adj, kg_adj_edges, kg_ent_labels, kg_ent_mask,
				coref_h_mapping, coref_t_mapping, coref_lens, coref_mention_position, coref_label, coref_label_mask):
        # para_size, char_size, bsz = context_idxs.size(1), context_char_idxs.size(2), context_idxs.size(0)
        # context_ch = self.char_emb(context_char_idxs.contiguous().view(-1, char_size)).view(bsz * para_size, char_size, -1)
        # context_ch = self.char_cnn(context_ch.permute(0, 2, 1).contiguous()).max(dim=-1)[0].view(bsz, para_size, -1)
        sent = self.word_emb(context_idxs)
        if Config.add_kg_flag or Config.add_coref_flag:
            sent_kg = sent.detach().clone()
        # Knowledge Injection layer
        '''======加入共指信息=======(更新指代词表示)'''
        loss_coref = None
        if Config.add_coref_flag and Config.coref_place == 'afterWordvec0':
            sent_kg, loss_coref = self.coref_injection(coref_h_mapping, coref_t_mapping,
                                                    coref_lens, sent_kg,
                                                    coref_mention_position, coref_label,
                                                    coref_label_mask)

        '''======加入kg信息======（更新实体mention表示）'''
        loss_kg = None
        if Config.add_kg_flag:
            context_masks = context_idxs.clone().bool()
            context_masks[context_masks > 0] = 1
            # print(context_masks.size())
            doc_rep = torch.max(sent_kg, dim=1)[0]
            sent_kg, loss_kg = self.kg_injection(kg_ent_attrs, kg_ent_attr_nums,
                                              kg_ent_attr_lens,
                                              kg_adj, kg_ent_labels,
                                              sent_kg, doc_rep, context_masks, kg_ent_mask,
                                              kg_ent_labels, kg_adj_edges)

        sent_pos = self.coref_embed(pos)
        sent_ner = self.ner_emb(context_ner)
        sent = torch.cat([sent, sent_pos, sent_ner], dim=-1)
        if Config.add_kg_flag or Config.add_coref_flag:
            sent_pos_kg = sent_pos.detach().clone()
            sent_ner_kg = sent_ner.detach().clone()
            sent_kg = torch.cat([sent_kg, sent_pos_kg, sent_ner_kg], dim=-1)
        # sent = torch.cat([self.word_emb(context_idxs), context_ch], dim=-1)

        # Knowledge Injection layer
        '''======加入共指信息=======(更新指代词表示)'''
        if Config.add_coref_flag and Config.coref_place == 'afterWordvec1':
            sent_kg, loss_coref = self.coref_injection(coref_h_mapping, coref_t_mapping,
                                                    coref_lens, sent_kg,
                                                    coref_mention_position, coref_label,
                                                    coref_label_mask)
        # '''======加入kg信息======（更新实体mention表示）'''
        # loss_kg = None
        # if self.kiconfig.add_kg_flag:
        #     context_masks = context_idxs.clone().bool()
        #     context_masks[context_masks > 0] = 1
        #     doc_rep = torch.max(sent, dim=-1)[0]
        #     sent, loss_kg = self.kg_injection(kg_ent_attrs, kg_ent_attr_nums,
        #                                              kg_ent_attr_lens,
        #                                              kg_adj, kg_ent_labels,
        #                                              sent, doc_rep, context_masks, kg_ent_mask,
        #                                              kg_ent_labels)

        # context_mask = (context_idxs > 0).float()
        context_output = self.rnn(sent, context_lens)
        if Config.add_kg_flag or Config.add_coref_flag:
            context_output_kg = self.rnn(sent_kg, context_lens)

        if Config.add_coref_flag and Config.coref_place == 'afterRnn':
            context_output_kg, loss_coref = self.coref_injection(coref_h_mapping, coref_t_mapping,
                                                              coref_lens, context_output_kg,
                                                              coref_mention_position, coref_label,
                                                              coref_label_mask)
        if Config.add_kg_flag or Config.add_coref_flag:
            context_output = self.kg_linear_transfer(torch.cat([context_output, context_output_kg], dim=-1))
            # context_output = torch.mean(torch.stack([context_output, context_output_kg]), dim=0)
        context_output = torch.relu(self.linear_re(context_output))

        start_re_output = torch.matmul(h_mapping, context_output)
        end_re_output = torch.matmul(t_mapping, context_output)
        # predict_re = self.bili(start_re_output, end_re_output)

        s_rep = torch.cat([start_re_output, self.dis_embed(dis_h_2_t)], dim=-1)
        t_rep = torch.cat([end_re_output, self.dis_embed(dis_t_2_h)], dim=-1)
        predict_re = self.bili(s_rep, t_rep)

        return predict_re, loss_kg, loss_coref


class LockedDropout(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = dropout

    def forward(self, x):
        dropout = self.dropout
        if not self.training:
            return x
        m = x.data.new(x.size(0), 1, x.size(2)).bernoulli_(1 - dropout)
        mask = Variable(m.div_(1 - dropout), requires_grad=False)
        mask = mask.expand_as(x)
        return mask * x


class EncoderRNN(nn.Module):
    def __init__(self, input_size, num_units, nlayers, concat, bidir, dropout, return_last):
        super().__init__()
        self.rnns = []
        for i in range(nlayers):
            if i == 0:
                input_size_ = input_size
                output_size_ = num_units
            else:
                input_size_ = num_units if not bidir else num_units * 2
                output_size_ = num_units
            self.rnns.append(nn.GRU(input_size_, output_size_, 1, bidirectional=bidir, batch_first=True))
        self.rnns = nn.ModuleList(self.rnns)
        self.init_hidden = nn.ParameterList(
            [nn.Parameter(torch.Tensor(2 if bidir else 1, 1, num_units).zero_()) for _ in range(nlayers)])
        self.dropout = LockedDropout(dropout)
        self.concat = concat
        self.nlayers = nlayers
        self.return_last = return_last

    # self.reset_parameters()

    def reset_parameters(self):
        for rnn in self.rnns:
            for name, p in rnn.named_parameters():
                if 'weight' in name:
                    p.data.normal_(std=0.1)
                else:
                    p.data.zero_()

    def get_init(self, bsz, i):
        return self.init_hidden[i].expand(-1, bsz, -1).contiguous()

    def forward(self, input, input_lengths=None):
        bsz, slen = input.size(0), input.size(1)
        output = input
        outputs = []
        if input_lengths is not None:
            lens = input_lengths.data.cpu().numpy()
        for i in range(self.nlayers):
            hidden = self.get_init(bsz, i)
            output = self.dropout(output)
            if input_lengths is not None:
                output = rnn.pack_padded_sequence(output, lens, batch_first=True)

            output, hidden = self.rnns[i](output, hidden)

            if input_lengths is not None:
                output, _ = rnn.pad_packed_sequence(output, batch_first=True)
                if output.size(1) < slen:  # used for parallel
                    padding = Variable(output.data.new(1, 1, 1).zero_())
                    output = torch.cat([output, padding.expand(output.size(0), slen - output.size(1), output.size(2))],
                                       dim=1)
            if self.return_last:
                outputs.append(hidden.permute(1, 0, 2).contiguous().view(bsz, -1))
            else:
                outputs.append(output)
        if self.concat:
            return torch.cat(outputs, dim=2)
        return outputs[-1]


class EncoderLSTM(nn.Module):
    def __init__(self, input_size, num_units, nlayers, concat, bidir, dropout, return_last):
        super().__init__()
        self.rnns = []
        for i in range(nlayers):
            if i == 0:
                input_size_ = input_size
                output_size_ = num_units
            else:
                input_size_ = num_units if not bidir else num_units * 2
                output_size_ = num_units
            self.rnns.append(nn.LSTM(input_size_, output_size_, 1, bidirectional=bidir, batch_first=True))
        self.rnns = nn.ModuleList(self.rnns)

        self.init_hidden = nn.ParameterList(
            [nn.Parameter(torch.Tensor(2 if bidir else 1, 1, num_units).zero_()) for _ in range(nlayers)])
        self.init_c = nn.ParameterList(
            [nn.Parameter(torch.Tensor(2 if bidir else 1, 1, num_units).zero_()) for _ in range(nlayers)])

        self.dropout = LockedDropout(dropout)
        self.concat = concat
        self.nlayers = nlayers
        self.return_last = return_last

    # self.reset_parameters()

    def reset_parameters(self):
        for rnn in self.rnns:
            for name, p in rnn.named_parameters():
                if 'weight' in name:
                    p.data.normal_(std=0.1)
                else:
                    p.data.zero_()

    def get_init(self, bsz, i):
        return self.init_hidden[i].expand(-1, bsz, -1).contiguous(), self.init_c[i].expand(-1, bsz, -1).contiguous()

    def forward(self, input, input_lengths=None):
        bsz, slen = input.size(0), input.size(1)
        output = input
        outputs = []
        if input_lengths is not None:
            lens = input_lengths.data.cpu().numpy()

        for i in range(self.nlayers):
            hidden, c = self.get_init(bsz, i)

            output = self.dropout(output)
            if input_lengths is not None:
                output = rnn.pack_padded_sequence(output, lens, batch_first=True)

            output, hidden = self.rnns[i](output, (hidden, c))

            if input_lengths is not None:
                output, _ = rnn.pad_packed_sequence(output, batch_first=True)
                if output.size(1) < slen:  # used for parallel
                    padding = Variable(output.data.new(1, 1, 1).zero_())
                    output = torch.cat([output, padding.expand(output.size(0), slen - output.size(1), output.size(2))],
                                       dim=1)
            if self.return_last:
                outputs.append(hidden.permute(1, 0, 2).contiguous().view(bsz, -1))
            else:
                outputs.append(output)
        if self.concat:
            return torch.cat(outputs, dim=2)
        return outputs[-1]


class BiAttention(nn.Module):
    def __init__(self, input_size, dropout):
        super().__init__()
        self.dropout = LockedDropout(dropout)
        self.input_linear = nn.Linear(input_size, 1, bias=False)
        self.memory_linear = nn.Linear(input_size, 1, bias=False)

        self.dot_scale = nn.Parameter(torch.Tensor(input_size).uniform_(1.0 / (input_size ** 0.5)))

    def forward(self, input, memory, mask):
        bsz, input_len, memory_len = input.size(0), input.size(1), memory.size(1)

        input = self.dropout(input)
        memory = self.dropout(memory)

        input_dot = self.input_linear(input)
        memory_dot = self.memory_linear(memory).view(bsz, 1, memory_len)
        cross_dot = torch.bmm(input * self.dot_scale, memory.permute(0, 2, 1).contiguous())
        att = input_dot + memory_dot + cross_dot
        att = att - 1e30 * (1 - mask[:, None])

        weight_one = F.softmax(att, dim=-1)
        output_one = torch.bmm(weight_one, memory)
        weight_two = F.softmax(att.max(dim=-1)[0], dim=-1).view(bsz, 1, input_len)
        output_two = torch.bmm(weight_two, input)

        return torch.cat([input, output_one, input * output_one, output_two * output_one], dim=-1)
